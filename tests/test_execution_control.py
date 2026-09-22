"""Cancellation behavior at the service/execution boundary, without CUDA."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from nodes.control import CancellationToken, RequestCancelled, service_checkpoint


class ExecutionControlTests(unittest.TestCase):
    def test_no_token_does_not_communicate(self):
        with patch("nodes.control.dist.all_reduce") as collective:
            service_checkpoint(SimpleNamespace(extra={}), None, phase="prepare")
        collective.assert_not_called()

    def test_local_cancellation_and_server_fallback(self):
        token = CancellationToken("request")
        batch = SimpleNamespace(extra={"service_cancellation_token": token})
        with patch("nodes.control.time.time", return_value=100):
            service_checkpoint(batch, None, phase="prepare")
        self.assertEqual(token.last_checkpoint_at, 100)
        token.request()
        requested_at = token.requested_at
        token.request()
        self.assertEqual(token.requested_at, requested_at)
        args = SimpleNamespace(_service_cancellation_token=token)
        for request, server in ((batch, None), (SimpleNamespace(extra={}), args)):
            with self.assertRaisesRegex(RequestCancelled, "request cancelled at denoise"):
                service_checkpoint(request, server, phase="denoise")

    def test_peer_cancellation_uses_control_group(self):
        token = CancellationToken("request")
        group = object()
        args = SimpleNamespace(parallel_context=SimpleNamespace(
            enabled=True, control_process_group=group, local_rank=0,
        ))
        batch = SimpleNamespace(extra={"service_cancellation_token": token})

        def peer_cancel(value, *, op, group):
            self.assertEqual(value.device.type, "cpu")
            self.assertEqual(value.dtype, torch.int64)
            self.assertEqual(value.item(), 0)
            value.fill_(1)

        with patch("nodes.control.dist.is_available", return_value=True), \
             patch("nodes.control.dist.is_initialized", return_value=True), \
             patch("nodes.control.dist.get_backend", return_value="gloo"), \
             patch("nodes.control.dist.all_reduce", side_effect=peer_cancel) as collective:
            with self.assertRaises(RequestCancelled):
                service_checkpoint(batch, args, phase="window")
        self.assertEqual(collective.call_count, 1)
        self.assertIs(collective.call_args.kwargs["group"], group)
        self.assertEqual(collective.call_args.kwargs["op"], torch.distributed.ReduceOp.MAX)

    def test_parallel_cancellation_requires_initialized_process_group(self):
        args = SimpleNamespace(parallel_context=SimpleNamespace(enabled=True))
        batch = SimpleNamespace(extra={"service_cancellation_token": CancellationToken("request")})
        with patch("nodes.control.dist.is_available", return_value=True), \
             patch("nodes.control.dist.is_initialized", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "requires torch.distributed"):
                service_checkpoint(batch, args, phase="prepare")

    def test_service_compatibility_exports_preserve_identity(self):
        from entrypoints.server import control

        self.assertIs(control.CancellationToken, CancellationToken)
        self.assertIs(control.RequestCancelled, RequestCancelled)
        self.assertIs(control.service_checkpoint, service_checkpoint)


if __name__ == "__main__":
    unittest.main()
