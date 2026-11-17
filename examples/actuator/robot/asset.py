# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""File-backed robot assets and actuator calibration environments."""

import hashlib
import importlib.metadata
import json
import shutil
import warnings
from pathlib import Path

import numpy as np

from newton_calibration import Asset

from .bindings import inspect_model
from .resources import freeze_resources, verify_resources


class AssetRobot(Asset):
    """Freeze and reconstruct an imported fixed-base robot with SolverMuJoCo.

    Supports inspection, resource delivery and configured actuator environments.
    """

    def __init__(self, baseline_parameters, *, config, resource_root):
        if baseline_parameters and "controller" not in config:
            raise ValueError("robot parameters require a controller configuration")
        super().__init__(baseline_parameters, config=config)
        self._resource_root = Path(resource_root).resolve()

    @classmethod
    def from_file(cls, source, *, bundle_dir, gravity, package_map=None, obj_materials="reject", provenance=None):
        """Freeze a local MJCF/URDF and record explicit gravity [m/s²]."""
        gravity = np.asarray(gravity, dtype=float)
        if gravity.shape != (3,) or not np.isfinite(gravity).all():
            raise ValueError("gravity must be a finite 3-vector")
        resources = freeze_resources(source, bundle_dir, package_map=package_map, obj_materials=obj_materials)
        versions = {}
        for package in ("newton", "warp-lang", "mujoco", "mujoco-warp", "trimesh", "pillow", "scipy"):
            distribution = importlib.metadata.distribution(package)
            versions[package] = {
                "version": distribution.version,
                "direct_url": json.loads(distribution.read_text("direct_url.json") or "null"),
            }
        config = {
            "model": "imported-robot/1",
            "solver": "SolverMuJoCo",
            "device": "cpu",
            "gravity": gravity.tolist(),
            "resources": resources,
            "versions": versions,
            "import_options": {
                "collapse_fixed_joints": False,
                "ignore_inertial_definitions": False,
                "parse_mujoco_options": False,
                "ctrl_direct": True,
            },
            "urdf_policy": "passive_damping_no_actuators",
        }
        if provenance is not None:
            config["source_provenance"] = provenance
        return cls({}, config=config, resource_root=bundle_dir).load_or_create()

    @property
    def bundle_name(self):
        """Use a content-addressed relative directory for exported resources."""
        return self.config.get("resource_bundle", "robot-" + self.asset_id)

    def with_controller(self, controller, baseline_parameters):
        """Attach an explicit sampled controller without changing frozen resources."""
        # Keep resource identity independent of fitted parameters on reload.
        config = json.loads(json.dumps(self.config, default=dict))
        config.update(controller=controller, resource_bundle=self.bundle_name)
        return type(self)(baseline_parameters, config=config, resource_root=self._resource_root)

    def load_or_create(self):
        verify_resources(self._resource_root, self.config["resources"]["files"])
        return self

    def with_device(self, device):
        """Select CPU or an explicit CUDA device in the reconstruction config."""
        if device != "cpu" and not (device.startswith("cuda:") and device[5:].isdigit()):
            raise ValueError("robot device must be cpu or cuda:<index>")
        config = json.loads(json.dumps(self.config, default=dict))
        config.update(device=device, resource_bundle=self.bundle_name)
        return type(self)(self.baseline_parameters, config=config, resource_root=self._resource_root)

    def _build_builder(self):
        import newton  # noqa: PLC0415 -- optional physics backend

        self.load_or_create()
        for package, expected in self.config["versions"].items():
            distribution = importlib.metadata.distribution(package)
            current = {
                "version": distribution.version,
                "direct_url": json.loads(distribution.read_text("direct_url.json") or "null"),
            }
            if current != dict(expected):
                raise ValueError(f"robot backend version changed: {package}")
        if (self.config["model"], self.config["solver"]) != (
            "imported-robot/1",
            "SolverMuJoCo",
        ):
            raise ValueError("unsupported robot construction configuration")
        options = dict(self.config["import_options"])
        if options != {
            "collapse_fixed_joints": False,
            "ignore_inertial_definitions": False,
            "parse_mujoco_options": False,
            "ctrl_direct": True,
        }:
            raise ValueError("unsupported robot import options")
        if self.config["urdf_policy"] != "passive_damping_no_actuators":
            raise ValueError("unsupported URDF drive policy")
        builder = newton.ModelBuilder(gravity=tuple(self.config["gravity"]))
        resources = self.config["resources"]
        source = self._resource_root / resources["entry"]
        if resources["entry"] not in resources["files"]:
            raise ValueError("model entry is not covered by resource hashes")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if resources["format"] == "mjcf":
                builder.add_mjcf(str(source), **options)
            elif resources["format"] == "urdf":
                builder.add_urdf(
                    str(source), floating=False, collapse_fixed_joints=False, ignore_inertial_definitions=False
                )
                # Newton imports URDF passive damping as target_kd and creates
                # position drives. URDF has no gains: restore passive semantics.
                builder.joint_damping[:] = builder.joint_target_kd
                builder.joint_target_kd[:] = [0.0] * len(builder.joint_target_kd)
                builder.joint_target_ke[:] = [0.0] * len(builder.joint_target_ke)
                builder.joint_target_mode[:] = [newton.JointTargetMode.NONE] * len(builder.joint_target_mode)
            else:
                raise ValueError("unsupported robot format")
            if caught:
                raise ValueError("robot import warning: " + "; ".join(str(item.message) for item in caught))
        if any(kind not in (newton.JointType.FIXED, newton.JointType.REVOLUTE) for kind in builder.joint_type):
            raise ValueError("only fixed and revolute joints are supported")
        if len(builder.body_label) != len(resources["names"]["bodies"]):
            raise ValueError("body count changed during import")
        return builder

    def build_model(self):
        """Build a CPU inspection model with actual names and physical indices."""
        import newton  # noqa: PLC0415

        builder = self._build_builder()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = builder.finalize(device="cpu")
            solver = newton.solvers.SolverMuJoCo(
                model, use_mujoco_cpu=True, integrator="implicitfast", disable_contacts=True, update_data_interval=1
            )
            if caught:
                raise ValueError("robot solver conversion warning: " + "; ".join(str(item.message) for item in caught))
        report = inspect_model(model, solver, self.config["resources"]["names"])
        return model, solver, report

    def create_environment(self, template, parameter_values, experiment_settings):
        from .environment import EnvironmentRobot  # noqa: PLC0415 -- optional simulation adapter

        if "controller" not in self.config:
            raise ValueError("attach a controller before creating robot rollouts")
        if template.numeric_settings["device"] != self.config["device"]:
            raise ValueError("template and asset devices differ")
        if self.supports_batch:
            from .batch import EnvironmentRobotGPU  # noqa: PLC0415

            return EnvironmentRobotGPU(self, template.numeric_settings, parameter_values, experiment_settings)
        return EnvironmentRobot(self, template.numeric_settings, parameter_values, experiment_settings)

    @property
    def supports_batch(self):
        return self.config["device"].startswith("cuda:")

    def create_batch_environment(self, template, parameter_values, experiment_settings):
        from .batch import BatchEnvironmentRobotGPU  # noqa: PLC0415

        if not self.supports_batch or template.numeric_settings["device"] != self.config["device"]:
            raise ValueError("robot batches require matching CUDA asset and template devices")
        return BatchEnvironmentRobotGPU(self, template.numeric_settings, parameter_values, experiment_settings)

    def _copy_resources(self, parent):
        self.load_or_create()
        target = Path(parent) / self.bundle_name
        if target.exists():
            verify_resources(target, self.config["resources"]["files"])
        else:
            shutil.copytree(self._resource_root, target)
            verify_resources(target, self.config["resources"]["files"])

    def export(self, parameter_values, path):
        if set(parameter_values) != set(self.baseline_parameters):
            raise ValueError("exported robot parameter names must match the asset")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._copy_resources(path.parent)
        return super().export(parameter_values, path)

    def reload(self, path):
        loaded = super().reload(path)
        loaded._resource_root = Path(path).resolve().parent / loaded.bundle_name
        return loaded.load_or_create()

    @classmethod
    def from_export(cls, path):
        """Reconstruct an exported asset without access to its original source."""
        record = json.loads(Path(path).read_text())
        asset = cls(record["parameters"], config=record["config"], resource_root=Path(path).parent)
        return asset.reload(path)

    def stage_release(self, candidate_path, release_path):
        loaded = self.reload(candidate_path)
        release_path = Path(release_path)
        release_path.parent.mkdir(parents=True, exist_ok=True)
        loaded._copy_resources(release_path.parent)
        super().stage_release(candidate_path, release_path)
        # Check bytes as well as resource hashes; publication cannot change the candidate.
        if (
            hashlib.sha256(Path(candidate_path).read_bytes()).digest()
            != hashlib.sha256(release_path.read_bytes()).digest()
        ):
            raise ValueError("candidate changed during release staging")
