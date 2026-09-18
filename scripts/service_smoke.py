#!/usr/bin/env python3
"""End-to-end service acceptance for a local erase pipeline.

Starts nothing itself: point it at a running service and it exercises the full
endpoint set, the task lifecycle and the result download.

    ./inference_server.sh --pipeline-name EraserDiTErasePipeline \
        --model-path <snapshot> --task-root /tmp/mgerase_tasks \
        --input-allowed-root "$PWD/data" &
    python scripts/service_smoke.py --base-url http://127.0.0.1:30000 \
        --video data/10268234.mp4 --mask data/10268234_mask.mp4 \
        --prompt "There is a bridge over the lake."
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
_failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{PASS if ok else FAIL}  {name}{(' — ' + detail) if detail else ''}", flush=True)
    if not ok:
        _failures.append(name)


def request(
    base: str, method: str, path: str, body: dict | None = None
) -> tuple[int, object]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        base + path,
        data=data,
        method=method,
        headers={"content-type": "application/json"} if data else {},
    )
    def _decode(raw: bytes):
        # Binary payloads (the mp4 from /content) are returned as-is; anything
        # else is expected to be JSON.
        try:
            return json.loads(raw)
        except Exception:
            return raw

    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            return response.status, _decode(response.read())
    except urllib.error.HTTPError as error:
        return error.code, _decode(error.read())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--video", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--prompt", default="There is a bridge over the lake.")
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--keep", action="store_true", help="do not delete the task")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    status, health = request(base, "GET", "/health")
    check("GET /health", status == 200, f"status={status}")

    status, models = request(base, "GET", "/v1/models")
    check("GET /v1/models", status == 200 and models["data"], f"status={status}")
    model_id = models["data"][0]["id"] if models.get("data") else None
    capability = models["data"][0]["capability"] if models.get("data") else None
    check("model card carries a capability", bool(capability), str(capability))

    status, info = request(base, "GET", "/server_info")
    check(
        "GET /server_info reports effective acceleration",
        status == 200 and "effective_acceleration" in info,
        f"status={status}",
    )
    check("GET /model_info", request(base, "GET", "/model_info")[0] == 200)
    check("GET /stats", request(base, "GET", "/stats")[0] == 200)
    check(
        f"GET /v1/models/{model_id}",
        request(base, "GET", f"/v1/models/{model_id}")[0] == 200,
    )
    check(
        "unknown model id -> 404 error body",
        request(base, "GET", "/v1/models/nope")[1].get("error", {}).get("code")
        == "model_not_found",
    )

    # strict request contract: an undeclared field must be rejected
    status, payload = request(
        base,
        "POST",
        "/v1/videos",
        {
            "video_path": str(Path(args.video).resolve()),
            "mask_path": str(Path(args.mask).resolve()),
            "prompt": args.prompt,
            "not_a_field": 1,
        },
    )
    check(
        "undeclared field rejected",
        status == 422 and payload.get("error", {}).get("code") == "invalid_request",
        f"status={status}",
    )

    status, created = request(
        base,
        "POST",
        "/v1/videos",
        {
            "video_path": str(Path(args.video).resolve()),
            "mask_path": str(Path(args.mask).resolve()),
            "prompt": args.prompt,
        },
    )
    check("POST /v1/videos", status == 202, f"status={status}")
    if status != 202:
        print(json.dumps(created, indent=2)[:2000])
        return 1
    task_id = created["id"]
    check("created response has a queue position", "queue_position" in created)

    deadline = time.time() + args.timeout_seconds
    final = created
    while time.time() < deadline:
        _, final = request(base, "GET", f"/v1/videos/{task_id}")
        if final["status"] in {"completed", "failed", "cancelled"}:
            break
        time.sleep(10)
    check(
        "task reached a terminal state",
        final["status"] in {"completed", "failed", "cancelled"},
        f"status={final['status']}",
    )
    check("task completed", final["status"] == "completed", str(final.get("error")))
    check(
        "terminal snapshot carries the window counters",
        final.get("window_count") is not None,
        f"object={final.get('object_index')}/{final.get('object_count')} "
        f"window={final.get('window_index')}/{final.get('window_count')}",
    )
    check(
        "terminal snapshot carries metrics",
        bool(final.get("metrics")),
        f"keys={sorted(final.get('metrics', {}))[:6]}",
    )

    if final["status"] == "completed":
        status, content = request(base, "GET", f"/v1/videos/{task_id}/content")
        ok = status == 200 and isinstance(content, bytes) and len(content) > 1024
        check(
            "GET /v1/videos/{id}/content returns an mp4",
            ok,
            f"status={status} bytes={len(content) if isinstance(content, bytes) else 'n/a'}",
        )
        if ok:
            out = Path("/tmp") / f"service_smoke_{task_id}.mp4"
            out.write_bytes(content)
            print(f"    saved {out}")

    status, listed = request(base, "GET", "/v1/videos?limit=10")
    check(
        "GET /v1/videos lists the task",
        status == 200 and any(item["id"] == task_id for item in listed["data"]),
        f"status={status}",
    )

    if not args.keep:
        status, deleted = request(base, "DELETE", f"/v1/videos/{task_id}")
        check(
            "DELETE /v1/videos/{id}",
            deleted.get("deleted") is True,
            f"status={status}",
        )
        check("deleted task is gone", request(base, "GET", f"/v1/videos/{task_id}")[0] == 404)

    print()
    if _failures:
        print(f"{len(_failures)} check(s) failed: {', '.join(_failures)}")
        return 1
    print("all service checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
