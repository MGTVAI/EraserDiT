"""Check contract selection and parallel plan compatibility after relocation."""

import unittest
from unittest.mock import patch

from config.parallel import AccelerationConfig, ParallelMode, ResolvedAccelerationPlan
from config.service_contracts.eraserdit import ERASERDIT_SERVICE_CONTRACT
from config.service_contracts.ltx095 import LTX095_SERVICE_CONTRACT
from distributed.parallel_state import initialize_parallel_context
from parallel import planner
from pipelines.registry import PipelineRegistry
from pipelines.service_contract import resolve_service_contract


class AssemblyContractTests(unittest.TestCase):
    def test_model_contract_selection(self):
        for name, expected in (
            ("EraserDiTErasePipeline", ERASERDIT_SERVICE_CONTRACT),
            ("LTX095ErasePipeline", LTX095_SERVICE_CONTRACT),
            (None, LTX095_SERVICE_CONTRACT),
        ):
            with self.subTest(pipeline=name):
                self.assertIs(resolve_service_contract(name), expected)

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
