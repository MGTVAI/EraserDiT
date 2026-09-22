"""Check contract selection and parallel plan compatibility after relocation."""

import unittest
from unittest.mock import patch

from config.parallel import AccelerationConfig, ParallelMode, ResolvedAccelerationPlan
from config.service_contracts.eraserdit import ERASERDIT_SERVICE_CONTRACT
from distributed.parallel_state import initialize_parallel_context
from parallel import planner
from pipelines.registry import PipelineRegistry
from pipelines.service_contract import resolve_service_contract


class AssemblyContractTests(unittest.TestCase):
    def test_model_contract_selection(self):
        for name, expected in (
            ("EraserDiTErasePipeline", ERASERDIT_SERVICE_CONTRACT),
            (None, ERASERDIT_SERVICE_CONTRACT),
        ):
            with self.subTest(pipeline=name):
                self.assertIs(resolve_service_contract(name), expected)

    def test_only_eraserdit_is_registered(self):
        self.assertEqual(PipelineRegistry.names(), ["EraserDiTErasePipeline"])
        with self.assertRaisesRegex(ValueError, "Unsupported pipeline"):
            resolve_service_contract("LTX095ErasePipeline")

    def test_default_protocol_uses_eraserdit_schema(self):
        from entrypoints.server.protocol import LocalVideoCreateRequest
        from config.service_contracts.eraserdit import EraserDiTLocalVideoCreateRequest
        self.assertIs(LocalVideoCreateRequest, EraserDiTLocalVideoCreateRequest)

    def test_scheduler_resolves_without_retired_model_aliases(self):
        from models.registry import ModelRegistry
        from models.schedulers.flow_match import FlowMatchEulerDiscreteScheduler
        cls, name = ModelRegistry.resolve_model_cls("FlowMatchEulerDiscreteScheduler")
        self.assertIs(cls, FlowMatchEulerDiscreteScheduler)
        scheduler = cls(num_train_timesteps=1000)
        self.assertEqual(scheduler.num_train_timesteps, 1000)

    def test_removed_quantization_modes_are_rejected(self):
        from config.server_args import ServerArgs
        for mode in ("fp8_w8a8", "fp8_w8a8_triton_selective", "int8_w8a8_viditq"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                ServerArgs(transformer_quantization=mode)
        with self.assertRaises(ValueError):
            ServerArgs(text_encoder_quantization="int8_w8a8_viditq")

    def test_runtime_plans_complete_example_video(self):
        from config.eraserdit import EraserDiTEraseSamplingParams
        from pipelines.runtime.windowing.planner import build_window_specs
        specs = build_window_specs(EraserDiTEraseSamplingParams(num_frames=145))
        self.assertEqual(len(specs), 2)
        self.assertEqual(specs[0].load_start, 0)
        self.assertEqual(specs[-1].load_end, 145)

    def test_invalid_model_contract_selection(self):
        with self.assertRaisesRegex(ValueError, "Unsupported pipeline"):
            resolve_service_contract("MissingPipeline")
        pipeline = type("WithoutContract", (), {})
        with patch.object(PipelineRegistry, "resolve", return_value=(pipeline, "WithoutContract")):
            with self.assertRaisesRegex(ValueError, "does not declare a service contract"):
                resolve_service_contract("WithoutContract")

    def test_parallel_types_keep_compatibility_identity(self):
        self.assertIs(planner.AccelerationConfig, AccelerationConfig)
        self.assertIs(planner.ParallelMode, ParallelMode)
        self.assertIs(planner.ResolvedAccelerationPlan, ResolvedAccelerationPlan)

    def test_auto_plan_mapping(self):
        for world_size, degrees in ((1, (1, 1, 1)), (2, (2, 1, 2)), (4, (2, 2, 4))):
            plan = planner.resolve_acceleration_plan(
                AccelerationConfig(parallel_mode=ParallelMode.AUTO), world_size=world_size,
            )
            self.assertIsInstance(plan, ResolvedAccelerationPlan)
            self.assertEqual((plan.sp_degree, plan.cfg_degree, plan.vae_degree), degrees)
            self.assertEqual(plan.enabled, world_size > 1)

    def test_disabled_plan_can_initialize_without_collectives(self):
        plan = planner.resolve_acceleration_plan(AccelerationConfig(), world_size=1)
        context = initialize_parallel_context(plan, global_rank=0, local_rank=0)
        self.assertIs(context.plan, plan)
        self.assertFalse(context.enabled)
        self.assertEqual(context.groups, {})


if __name__ == "__main__":
    unittest.main()
