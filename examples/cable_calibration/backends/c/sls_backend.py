"""SLS binding and measured prehistory for the pinned Newton extension."""

import newton
import numpy as np
import warp as wp

# Pinned solver adapter: use its strain measure rather than approximate angles.
from newton._src.solvers.vbd.rigid_vbd_kernels import (
    compute_geometric_rod_kappa_cached_z,
)

from .newton_backend import CableAsset, CableEnvironment


@wp.kernel
def measure_bend(
    parent: wp.array[int],
    child: wp.array[int],
    xp: wp.array[wp.transform],
    xc: wp.array[wp.transform],
    rest: wp.array[wp.vec3],
    twist: wp.array[float],
    q: wp.array[wp.transform],
    out: wp.array[wp.vec3],
):
    j = wp.tid()
    p = xp[j]
    if parent[j] >= 0:
        p = q[parent[j]] * p
    c = q[child[j]] * xc[j]
    k = compute_geometric_rod_kappa_cached_z(
        wp.transform_get_rotation(p),
        wp.transform_get_rotation(c),
        rest[j],
        twist[j],
    )
    out[j] = wp.vec3(k[0], k[1], 0.0)


def integrate_history(kappa, times, stiffness, tau, initial=None):
    """Exact Maxwell update for piecewise-linear observed strain."""
    moment = np.zeros_like(kappa[0]) if initial is None else initial.copy()
    for before, after, dt in zip(kappa[:-1], kappa[1:], np.diff(times), strict=True):
        decay = np.exp(-dt / tau)
        moment = decay * moment + stiffness[:, None] * tau / dt * (
            -np.expm1(-dt / tau)
        ) * (after - before)
    return moment


class SLSCableEnvironment(CableEnvironment):
    def _configure_builder(self, builder):
        newton.solvers.SolverVBD.register_custom_attributes(builder)

    def _configure_model(self, parts):
        if not hasattr(self.model, "vbd") or not hasattr(
            self.model.vbd, "visco_bend_ke"
        ):
            raise RuntimeError("Launch with the pinned vendor Newton on PYTHONPATH")
        ke, tau = np.zeros(self.model.joint_count), np.zeros(self.model.joint_count)
        for part in parts:
            if part.parameters.get("clamp_rotational_compliance", 0) != 0:
                raise ValueError("SLS adapter supports the frozen rigid mount")
            ids = part.joints[: part.n]
            ke[ids] = part.parameters["sls_transient_rigidity"] / part.h
            ke[ids[0]] *= 2
            tau[ids] = part.parameters["sls_relaxation_time"]
        self.model.vbd.visco_bend_ke.assign(ke)
        self.model.vbd.visco_bend_tau.assign(tau)

    def _readback(self):
        super()._readback()
        ids = self.joints[: self.n]
        factors = np.full(self.n, self.h)
        factors[0] /= 2
        actual_ke = self.model.vbd.visco_bend_ke.numpy()[ids] * factors
        actual_tau = self.model.vbd.visco_bend_tau.numpy()[ids]
        for name, values in (
            ("sls_transient_rigidity", actual_ke),
            ("sls_relaxation_time", actual_tau),
        ):
            if not np.allclose(values, self.parameters[name], rtol=2e-6, atol=1e-8):
                raise RuntimeError(f"SLS actual binding mismatch: {name}")
            self.applied_parameters[name] = float(values.mean())

    def _initialize_material_history(self, parts):
        if not self.solver.enable_viscoelasticity:
            return
        m, solver = self.model, self.solver
        moment = np.zeros((m.joint_count, 3))
        last_kappa = np.zeros_like(moment)
        out = wp.zeros(m.joint_count, dtype=wp.vec3, device=self.device)
        initial_q = self.state_0.body_q.numpy().copy()
        initial_qd = self.state_0.body_qd.numpy().copy()
        for part in parts:
            if part.settings["mode"] != "dynamic":
                continue
            history = part.settings["sls_history"]
            if not all(frame["valid"] for frame in history):
                raise ValueError("SLS prehistory contains invalid observations")
            kappa = []
            ids = part.joints[: part.n]
            for frame in history:
                part._set_initial_curve(np.asarray(frame["centerline"]))
                wp.launch(
                    measure_bend,
                    dim=m.joint_count,
                    inputs=[
                        m.joint_parent,
                        m.joint_child,
                        m.joint_X_p,
                        m.joint_X_c,
                        solver.joint_rod_rest_kb_local,
                        solver.joint_rod_rest_twist,
                        self.state_0.body_q,
                    ],
                    outputs=[out],
                    device=self.device,
                )
                kappa.append(out.numpy()[ids])
            kappa = np.asarray(kappa)
            stiffness = m.vbd.visco_bend_ke.numpy()[ids]
            initial = (
                part.settings.get("sls_history_initial_stress_factor", 0.0)
                * stiffness[:, None]
                * kappa[0]
            )
            moment[ids] = integrate_history(
                kappa,
                [f["time_s"] for f in history],
                stiffness,
                part.parameters["sls_relaxation_time"],
                initial,
            )
            last_kappa[ids] = kappa[-1]
        for state in (self.state_0, self.state_1):
            state.body_q.assign(initial_q)
            state.body_qd.assign(initial_qd)
        if self.settings["mode"] == "dynamic":
            solver.body_q_prev.assign(initial_q)
            solver.joint_visco_moment_prev.assign(moment)
            solver.joint_visco_kappa_prev.assign(last_kappa)
            solver._rigid_pose_rebaseline_mask.zero_()


class SLSCableAsset(CableAsset):
    environment_type = SLSCableEnvironment

    def create_environment(self, template, parameter_values, experiment_settings):
        return self.environment_type(
            template, parameter_values, experiment_settings, self.config
        )
