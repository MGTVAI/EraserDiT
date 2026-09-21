"""CPU-only HTTP contract tests; no weights or worker processes required."""
import tempfile
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from config.service_args import ServiceArgs
from config.service_contracts.eraserdit import ERASERDIT_SERVICE_CONTRACT
from entrypoints.http_server import create_http_server_app
from entrypoints.server.artifacts import TaskArtifactManager
from entrypoints.server.task import ServiceError, TaskStatus
from entrypoints.server.task_store import TaskStore


class SchedulerStub:
    def __init__(self, store):
        self.store = store
        self.heartbeat = time.time()
        self.ready = True
        self.full = False

    def health_snapshot(self):
        return {"ready": self.ready, "last_all_rank_heartbeat_at": self.heartbeat}

    def submit(self, record):
        if self.full:
            raise ServiceError("queue_full", "queue is full", status_code=429)
        self.store.create(record)

    def stats(self):
        return {"counts": self.store.status_counts()}

    def cancel_or_purge(self, task_id):
        record = self.store.get(task_id)
        if record.is_terminal:
            self.store.purge(task_id)
            return "purged", {"id": task_id, "deleted": True, "remote_result_deleted": False}
        self.store.transition(task_id, TaskStatus.CANCELLED)
        return "cancelled", self.store.snapshot(task_id)


class ServiceAPITest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        inputs = root / "inputs"
        inputs.mkdir()
        self.video = inputs / "video.mp4"
        self.video.write_bytes(b"video")
        self.mask = inputs / "mask.mp4"
        self.mask.write_bytes(b"mask")
        args = ServiceArgs(task_root=str(root / "tasks"), input_allowed_roots=(str(inputs),), max_upload_bytes=32)
        self.store = TaskStore(args.task_root, terminal_ttl_seconds=60, max_terminal_tasks=128)
        self.scheduler = SchedulerStub(self.store)
        app = create_http_server_app(
            service_args=args, scheduler=self.scheduler, task_store=self.store,
            artifact_manager=TaskArtifactManager(args.task_root, max_upload_bytes=32),
            server_summary={"pipeline": "EraserDiTErasePipeline"},
            model_summary={"id": "eraserdit", "capability": "eraserdit_video_erase"},
            service_contract=ERASERDIT_SERVICE_CONTRACT,
        )
        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        self.payload = {"video_path": str(self.video), "mask_path": str(self.mask)}

    def create(self, **values):
        response = self.client.post("/v1/videos", json={**self.payload, **values})
        self.assertEqual(response.status_code, 202, response.text)
        return response.json()["id"]

    def test_eraser_and_model_selection(self):
        response = self.client.post("/v1/videos/eraser", json={**self.payload, "model": "eraserdit"})
        self.assertEqual(response.status_code, 202, response.text)
        record = self.store.get(response.json()["id"])
        self.assertNotIn("model", record.request_payload)
        self.assertEqual(record.request_payload["num_inference_steps"], 50)
        self.assertEqual(self.client.post("/v1/videos", json={**self.payload, "model": "missing"}).status_code, 404)

    def test_progress_cancel_delete(self):
        task_id = self.create()
        url = f"/v1/videos/{task_id}"
        self.assertEqual(self.client.get(url + "/progress").json()["status"], "queued")
        self.assertEqual(self.client.get(url + "/content").status_code, 409)
        self.assertEqual(self.client.delete(url).json()["status"], "cancelled")
        self.assertTrue(self.client.delete(url).json()["deleted"])
        self.assertEqual(self.client.get(url).status_code, 404)

    def test_completed_content(self):
        task_id = self.create()
        self.store.transition(task_id, TaskStatus.RUNNING)
        self.store.begin_result_publication(task_id)
        output = self.store.get(task_id).task_dir / "outputs" / "result.mp4"
        output.write_bytes(b"test-result")
        self.store.set_result_storage(task_id, mode="local", local_path=output, url=None, fallback=False)
        self.store.transition(task_id, TaskStatus.COMPLETED)
        response = self.client.get(f"/v1/videos/{task_id}/content")
        self.assertEqual(response.content, b"test-result")
        self.assertEqual(response.headers["content-type"], "video/mp4")
        output.unlink()
        self.assertEqual(self.client.get(f"/v1/videos/{task_id}/content").status_code, 410)

    def test_pagination(self):
        ids = [self.create() for _ in range(3)]
        first = self.client.get("/v1/videos?limit=2&order=asc").json()
        self.assertTrue(first["has_more"])
        self.assertEqual(first["first_id"], ids[0])
        last = self.client.get("/v1/videos", params={"limit": 1, "order": "asc", "after": first["last_id"]}).json()
        self.assertFalse(last["has_more"])
        self.assertEqual(last["last_id"], ids[-1])
        for params in ({"limit": 0}, {"limit": 101}, {"order": "bad"}, {"after": "missing"}):
            self.assertEqual(self.client.get("/v1/videos", params=params).status_code, 422)

    def test_errors_and_cleanup(self):
        for values in ({"video_path": "/missing.mp4"}, {"unknown": 1}, {"infer_len": 9, "overlap": 9}, {"num_inference_steps": "5"}):
            response = self.client.post("/v1/videos", json={**self.payload, **values})
            self.assertEqual(response.status_code, 422, response.text)
            self.assertIn("error", response.json())
        self.scheduler.full = True
        self.assertEqual(self.client.post("/v1/videos", json=self.payload).status_code, 429)
        self.assertEqual(list(self.store.tasks_root.iterdir()), [])
        self.assertIn("error", self.client.get("/missing-route").json())
        response = self.client.put("/v1/videos")
        self.assertEqual(response.status_code, 405)
        self.assertIn("allow", response.headers)

    def test_allowlist_symlink(self):
        outside = Path(self.tmp.name) / "outside.mp4"
        outside.write_bytes(b"outside")
        link = self.video.parent / "link.mp4"
        link.symlink_to(outside)
        response = self.client.post("/v1/videos", json={**self.payload, "video_path": str(link)})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "input_path_not_allowed")

    def test_health_admission_and_aliases(self):
        for url in ("/health", "/ready", "/server_info", "/get_server_info", "/model_info", "/get_model_info", "/v1/models", "/stats"):
            self.assertEqual(self.client.get(url).status_code, 200)
        self.assertEqual(self.client.get("/server_info").json()["service"], "eraserdit_video_erase")
        self.scheduler.heartbeat = time.time() - 1000
        self.assertEqual(self.client.get("/health").status_code, 503)
        self.assertEqual(self.client.post("/v1/videos", json=self.payload).status_code, 503)
        self.assertEqual(list(self.store.tasks_root.iterdir()), [])

    def test_multipart(self):
        files = {"video": ("v.mp4", b"video"), "mask": ("m.mp4", b"mask")}
        response = self.client.post("/v1/videos/eraser", files=files, data={"parameters": '{"model":"eraserdit", "seed":7}'})
        self.assertEqual(response.status_code, 202, response.text)
        record = self.store.get(response.json()["id"])
        self.assertEqual(record.request_payload["seed"], 7)
        self.assertEqual(record.video_input_path.read_bytes(), b"video")
        response = self.client.post("/v1/videos", files={**files, "video": ("v.mp4", b"x" * 33)})
        self.assertEqual(response.status_code, 429)
        response = self.client.post("/v1/videos", files=[("video", ("a", b"a")), ("video", ("b", b"b")), ("mask", ("m", b"m"))])
        self.assertEqual(response.status_code, 422)
        self.assertEqual(len(list(self.store.tasks_root.iterdir())), 1)

    def test_openapi(self):
        schema = self.client.get("/openapi.json").json()
        for path in ("/v1/videos", "/v1/videos/eraser"):
            content = schema["paths"][path]["post"]["requestBody"]["content"]
            properties = content["application/json"]["schema"]["properties"]
            self.assertEqual(properties["num_inference_steps"]["default"], 50)
            self.assertIn("model", properties)
            self.assertIn("multipart/form-data", content)


if __name__ == "__main__":
    unittest.main()
