"""Compose a fresh real Tuning from a portable stage configuration.

Construction does not simulate. CLI execution requires --run.
"""

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

from task_tuning import (
    CMAES,
    Decision,
    Evidence,
    GridSearch,
    Parameter,
    Parameters,
    Play,
    Template,
    Tuning,
)

from .verify import CASE, load_experiments


def check_solver(source, stage):
    source = Path(source).expanduser().resolve()
    expected = json.loads((CASE / "solver/expected-hashes.json").read_text())
    hashes = expected["sls" if stage == "C" else "base"]
    for name, digest in hashes.items():
        path = source / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Wrong Newton VBD source for {stage}: {path}")
    sys.path.insert(0, str(source))
    newton = importlib.import_module("newton")
    if not Path(newton.__file__).resolve().is_relative_to(source):
        raise RuntimeError("Another Newton is already imported; start a fresh Python process.")
    return source, hashes


def build_tuning(stage, *, newton_source, workspace="workspace/cable", device=None):
    if stage not in ("A", "B", "Bstar", "C"):
        raise ValueError(f"Unknown stage: {stage}")
    source, solver_hashes = check_solver(newton_source, stage)
    config_path = CASE / "configs" / f"{stage}.json"
    config = json.loads(config_path.read_text())
    package = f"{__package__}.backends.{stage.lower()}"
    backend = importlib.import_module(
        package + (".sls_backend" if stage == "C" else ".newton_backend")
    )
    rollout = importlib.import_module(package + ".rollout")
    validation = importlib.import_module(package + ".validation")
    if stage in ("Bstar", "C"):
        evaluator_type = importlib.import_module(
            package + ".staged_evaluator"
        ).DistalDynamicEvaluator
    else:
        evaluator_type = validation.CableEvaluator
    parameters = Parameters([Parameter(**p) for p in config["parameters"]])
    data = config["template"]
    if device is not None:
        data["numeric_settings"]["device"] = device
    asset_type = backend.SLSCableAsset if stage == "C" else backend.CableAsset
    experiments = load_experiments()
    sources = {
        "historical_run": config["historical_run_id"],
        "portable_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "processed_data_sha256": hashlib.sha256(
            (CASE / "data/experiments.json").read_bytes()
        ).hexdigest(),
        "adapter_hashes": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((CASE / "backends" / stage.lower()).glob("*.py"))
        },
        "solver_vbd_hashes": solver_hashes,
        "interpretation": "New run; historical precision and acceptance are not reused.",
        "validation": "With-plug data are historical diagnostics, not an independent holdout.",
    }
    return Tuning(
        evidence=Evidence(
            [experiments[k] for k in config["train"]],
            [experiments[k] for k in config["validation"]],
            sources=sources,
        ),
        asset=asset_type(
            parameters.as_dict(parameters.baseline),
            config={**config["asset_config"], "newton_source": str(source / "newton/__init__.py")},
        ),
        evaluator=evaluator_type(),
        parameters=parameters,
        template=Template(**data),
        rollout=rollout.NewtonRollout(capacity=8),
        optimizer=GridSearch(levels=17, batch_size=8)
        if stage == "Bstar"
        else CMAES(batch_size=8, sigma=0.2, seed=42),
        decision=Decision(**config["decision"]),
        validator=validation.HistoricalValidator(max_objective=1.0, max_error=3.0),
        play=Play(),
        workspace=workspace,
        orthogonalize=config["orthogonalize"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("A", "B", "Bstar", "C"), required=True)
    parser.add_argument("--newton-source", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path("workspace/cable"))
    parser.add_argument(
        "--device", help="Optional new device setting; changes numerical conditions"
    )
    parser.add_argument("--run", action="store_true", help="Start qualification and a NEW search")
    args = parser.parse_args()
    tuning = build_tuning(
        args.stage, newton_source=args.newton_source, workspace=args.workspace, device=args.device
    )
    print(
        f"stage={args.stage}, parameters={tuning.parameters.names}, "
        f"run_directory={tuning.run_directory}"
    )
    if not args.run:
        print("Constructed only. No simulation performed. Add --run to start a new calibration.")
        return 0
    import warp as wp

    cache = tuning.workspace / ".warp_cache"
    cache.mkdir(parents=True, exist_ok=True)
    wp.config.kernel_cache_dir = str(cache)
    result = tuning.run()
    print(result)
    return 0 if result.acceptance == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
