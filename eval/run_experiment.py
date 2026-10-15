"""Run a config over the dev set or the FROZEN subset.

    uv run python -m eval.run_experiment --config configs/eval/a1.yaml --set dev \\
        --seeds 0 1 2 --workers 2 --out out/experiments/a1-dev

``--set frozen`` refuses to start unless docs/PREREGISTRATION.md is filled in and
committed and the working tree is clean (see eval/prereg.py). Re-running the same
command resumes: finished (instance, seed) pairs are skipped.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

import docker

from agent.config import REPO_ROOT, ExperimentConfig
from agent.instance import Instance
from eval.data import SetName, load_set
from eval.experiment import ExperimentError, run_experiment
from eval.prereg import PreregistrationError, PreregStatus, require_preregistered
from eval.swebench_adapter import ensure_image, to_instance


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--set", dest="set_name", choices=["dev", "frozen"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--instance-ids", nargs="+", help="dev set only: restrict to these ids")
    args = ap.parse_args(argv)
    set_name = cast(SetName, args.set_name)

    prereg: PreregStatus | None = None
    if set_name == "frozen":
        if args.instance_ids:
            ap.error("--instance-ids is not allowed on the frozen subset")
        try:
            prereg = require_preregistered(REPO_ROOT)
        except PreregistrationError as exc:
            print(exc, file=sys.stderr)
            return 3

    config = ExperimentConfig.from_yaml(args.config)
    data = load_set(set_name)
    records = data.records
    if args.instance_ids:
        wanted = set(args.instance_ids)
        records = [r for r in records if r["instance_id"] in wanted]
        if len(records) != len(wanted):
            ap.error("some --instance-ids are not in the dev set")
    instances = [to_instance(r) for r in records]
    client = docker.from_env()

    def prepare(inst: Instance) -> None:
        if inst.image:
            ensure_image(inst.image, client)

    try:
        report = run_experiment(
            config=config,
            set_name=set_name,
            data_file=data.path.name,
            data_sha256=data.sha256,
            instances=instances,
            seeds=args.seeds,
            workers=args.workers,
            out_dir=args.out,
            prereg=prereg,
            prepare=prepare,
        )
    except ExperimentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"ran {report.ran}, skipped {report.skipped_done} finished, "
        f"not run (budget) {len(report.not_run)}, env errors {report.env_errors}, "
        f"invalid traces {report.invalid_traces}, aborted by cap {report.aborted_by_cap}, "
        f"spent ${report.spent_usd:.2f}  ->  {report.out_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
