# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experiment inputs, tolerant export reload, release formats and relative stagnation."""

import json
import unittest
from dataclasses import replace

import numpy as np

from newton_calibration import (
    Asset,
    BatchEnvironment,
    Decision,
    Environment,
    Evidence,
    Experiment,
    Template,
)
from newton_calibration._src.data import identity
from newton_calibration.examples import OscillatorAsset, make_tuning
from newton_calibration.tests.unittest_utils import TempPathTestCase


class _Probe(Environment):
    def __init__(self):
        self.applied_parameters = {}

    def run(self, times, *, seed, deadline):
        raise NotImplementedError


class _BatchProbe(BatchEnvironment):
    applied_parameters = ()

    def run(self, times, *, seed, deadline):
        raise NotImplementedError


def _template():
    template = Template(
        numeric_settings={"dt": 0.1},
        discretization={},
        fixed_parameters={},
        experiment_settings={"e": {"seed": 0}},
        precision_tolerance=1.0,
        refinements=(),
        qualification_mode="repeatability",
    )
    return template.freeze()


def _experiment(inputs=None):
    return Experiment("e", "g", np.zeros((3, 1)), np.array([0.0, 0.1, 0.2]), inputs=inputs)


class Float32Export(OscillatorAsset):
    """A native asset format that stores single-precision values."""

    def reload(self, path):
        loaded = super().reload(path)
        loaded.baseline_parameters = {k: float(np.float32(v)) for k, v in loaded.baseline_parameters.items()}
        return loaded


class TextExport(OscillatorAsset):
    def export(self, parameter_values, path):
        path = path.with_suffix(".txt")
        path.write_text(json.dumps(dict(parameter_values)))
        return path

    def reload(self, path):
        return TextExport(json.loads(path.read_text()))


def _state(history):
    return {
        "analysis_reports": {},
        "fatal_error": None,
        "cancelled": False,
        "elapsed_s": 0.0,
        "budget_usage": {},
        "best_candidate": {"objective": history[-1]},
        "optimizer_snapshot": {},
        "batch_history": [{"purpose": "train", "best_objective": x} for x in history],
    }


class TestExperimentInputs(unittest.TestCase):
    def test_inputs_are_validated_and_fingerprinted(self):
        """Validate experiment inputs and include them in the evidence fingerprint."""
        times, reference = np.array([0.0, 1.0]), np.zeros((2, 1))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            Experiment("e", "g", reference, times, inputs={"u": [np.nan]}).normalized()
        with self.assertRaisesRegex(ValueError, "nonempty"):
            Experiment("e", "g", reference, times, inputs={}).normalized()

        def fingerprint(command):
            evidence = Evidence([Experiment("e", "g", reference, times, inputs={"u": command})])
            evidence.prepare()
            return evidence.evidence_id

        self.assertEqual(fingerprint([1.0, 2.0]), fingerprint([1.0, 2.0]))
        self.assertNotEqual(fingerprint([1.0, 2.0]), fingerprint([1.0, 2.5]))
        self.assertNotEqual(
            identity(Experiment("e", "g", reference, times)),
            identity(Experiment("e", "g", reference, times, inputs={"u": [1.0]})),
        )

    def test_template_passes_inputs_only_when_declared(self):
        """Pass inputs only to assets whose environment factories declare them."""
        received = []

        class LegacyAsset(Asset):
            def create_environment(self, template, parameter_values, experiment_settings):
                received.append("legacy")
                return _Probe()

            def create_batch_environment(self, template, parameter_values, experiment_settings):
                received.append("legacy batch")
                return _BatchProbe()

        class InputAsset(Asset):
            def create_environment(self, template, parameter_values, experiment_settings, *, inputs):
                received.append(sorted(inputs))
                return _Probe()

            def create_batch_environment(self, template, parameter_values, experiment_settings, *, inputs):
                received.append(("batch", sorted(inputs)))
                return _BatchProbe()

        template = _template()
        plain, with_inputs = _experiment().normalized(), _experiment({"u": np.ones(3)}).normalized()
        template.instantiate(LegacyAsset({}), {"a": 1.0}, plain)
        template.instantiate_batch(LegacyAsset({}), [{"a": 1.0}], plain)
        template.instantiate(InputAsset({}), {"a": 1.0}, with_inputs)
        template.instantiate_batch(InputAsset({}), [{"a": 1.0}], with_inputs)
        self.assertEqual(received, ["legacy", "legacy batch", ["u"], ("batch", ["u"])])


class TestExportAndRelease(TempPathTestCase):
    def test_reload_comparison_uses_rollout_readback_tolerance(self):
        """Compare reloaded parameters within the rollout readback tolerance."""
        strict = make_tuning("oscillator", self.tmp_path / "strict", optimizer="grid")
        strict.asset = Float32Export(strict.asset.baseline_parameters)
        strict.rollout.parameter_rtol = strict.rollout.parameter_atol = 0.0
        self.assertEqual(strict.run().acceptance, "run_failed")
        self.assertTrue(any("reloaded" in e for e in strict.state.errors))

        tolerant = make_tuning("oscillator", self.tmp_path / "tolerant", optimizer="grid")
        tolerant.asset = Float32Export(tolerant.asset.baseline_parameters)
        tolerant.rollout.parameter_rtol = 1e-6
        result = tolerant.run()
        self.assertEqual(result.acceptance, "accepted", tolerant.state.errors)
        roundtrip = tolerant.state.analysis_reports["export_roundtrip"]
        self.assertNotEqual(roundtrip["reloaded"], roundtrip["selected"])
        self.assertEqual(result.asset_path.name, "asset.json")

    def test_release_keeps_the_exported_format(self):
        """Release the asset in the file format its export produced."""
        tuning = make_tuning("oscillator", self.tmp_path / "text", optimizer="grid")
        tuning.asset = TextExport(tuning.asset.baseline_parameters)
        result = tuning.run()
        self.assertEqual(result.acceptance, "accepted", tuning.state.errors)
        self.assertEqual(result.asset_path.name, "asset.txt")
        manifest = json.loads((result.asset_path.parent / "manifest.json").read_text())
        self.assertEqual(manifest["asset"], "asset.txt")


class TestRelativeStagnation(unittest.TestCase):
    def test_relative_improvement_tolerance_stops_on_small_relative_progress(self):
        """Stop on stagnation when relative progress falls below the tolerance."""
        request = {"purpose": "train", "count": 4, "episodes_per_trial": 1}
        absolute = Decision(target=0.0, patience=2)
        relative = replace(absolute, relative_improvement_tolerance=1e-3)
        slow = _state([10.0, 9.999, 9.998])
        self.assertEqual(absolute.next_action(slow, request).action, "continue")
        self.assertEqual(relative.next_action(slow, request).reason, "stagnation")
        self.assertEqual(relative.next_action(_state([10.0, 9.0, 8.0]), request).action, "continue")
        with self.assertRaises(ValueError):
            Decision(relative_improvement_tolerance=-1.0)


if __name__ == "__main__":
    unittest.main()
