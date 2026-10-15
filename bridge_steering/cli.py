"""Single CLI entry point.

  python -m bridge_steering pipeline --config X.yaml
  python -m bridge_steering pipeline --config X.yaml --only steer_collect,steer_extract,aut_infer
  python -m bridge_steering pipeline --config X.yaml --resume
  python -m bridge_steering pipeline --config X.yaml --dry-run

  # Standalone stage — same code path as `pipeline --only`
  python -m bridge_steering steer-collect    --config X.yaml
  python -m bridge_steering aut-infer        --config X.yaml
  python -m bridge_steering aut-ocsai        --config X.yaml
  python -m bridge_steering aut-judge        --config X.yaml
  python -m bridge_steering mac-infer        --config X.yaml
  python -m bridge_steering mac-judge        --config X.yaml
  python -m bridge_steering aggregate        --config X.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import RunConfig, load_run_config
from .pipeline import PipelineError, run_pipeline
from .repro import build_run_context
from .stages import DEFAULT_ORDER, STAGES


# Subcommand name -> stage name (CLI uses dashes; STAGES uses underscores).
_STANDALONE: dict[str, str] = {
    "steer-collect":          "steer_collect",
    "steer-extract":          "steer_extract",
    "aut-infer":              "aut_infer",
    "aut-brainstorm-select":  "aut_brainstorm_select",
    "aut-ocsai":              "aut_ocsai",
    "aut-novelty":            "aut_novelty",
    "aut-judge":              "aut_judge",
    "mac-infer":              "macgyver_infer",
    "mac-novelty":            "macgyver_novelty",
    "mac-judge":              "macgyver_judge",
    "aggregate":              "aggregate",
}


def _parse_only_csv(s: str | None) -> list[str] | None:
    if not s:
        return None
    items = [t.strip() for t in s.split(",") if t.strip()]
    unknown = [t for t in items if t not in STAGES]
    if unknown:
        raise SystemExit(
            f"--only: unknown stage(s) {unknown}. Allowed: {sorted(STAGES)}"
        )
    return items


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m bridge_steering",
                                description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pipeline = sub.add_parser("pipeline", help="run multiple stages")
    pipeline.add_argument("--config", "-c", type=Path, required=True)
    pipeline.add_argument("--only", type=str, default=None,
                          help="comma-separated stage names")
    pipeline.add_argument("--with-deps", action="store_true",
                          help="auto-run missing upstream prerequisites")
    pipeline.add_argument("--resume", action="store_true",
                          help="skip stages whose outputs are present + sha-verified")
    pipeline.add_argument("--dry-run", action="store_true")

    for cli_name in _STANDALONE:
        sp = sub.add_parser(cli_name, help=f"run only {_STANDALONE[cli_name]}")
        sp.add_argument("--config", "-c", type=Path, required=True)
        sp.add_argument("--with-deps", action="store_true")
        sp.add_argument("--resume", action="store_true")
        sp.add_argument("--dry-run", action="store_true")

    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = load_run_config(args.config)
    ctx = build_run_context(cfg)

    if args.cmd == "pipeline":
        only = _parse_only_csv(args.only)
        kw = dict(only=only, with_deps=args.with_deps,
                  resume=args.resume, dry_run=args.dry_run)
    else:
        only = [_STANDALONE[args.cmd]]
        kw = dict(only=only, with_deps=args.with_deps,
                  resume=args.resume, dry_run=args.dry_run)

    try:
        run_pipeline(ctx, **kw)
    except PipelineError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
