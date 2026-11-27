"""Launch supervised hanging through the deployed ``robotwin-simulator`` app.

Unlike ``modal run modal_apps/simulator.py::supervised_hanging``, this module
does not create an ephemeral Modal App. It spawns a detached GPU Function call,
records its call ID locally, and downloads the resulting run artifacts. A lost
terminal can therefore resume the call instead of cancelling the episode.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time


APP_NAME = "robotwin-simulator"
FUNCTION_NAME = "run_supervised_hanging"
REPO = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-state", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--max-actions", type=int, default=900)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--time-limit-minutes", type=int, default=30)
    parser.add_argument("--max-turns", type=int, default=80)
    parser.add_argument("--output", default="")
    parser.add_argument("--complementary-view", action="store_true")
    parser.add_argument("--perception-view", default="default")
    parser.add_argument(
        "--hand-back-to-policy", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--policy-grace-actions", type=int, default=0)
    parser.add_argument("--skip-observations", action="store_true")
    parser.add_argument(
        "--grasp-recovery", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--grasp-retries", type=int, default=2)
    parser.add_argument("--empty-hand-opening", type=float, default=0.19)
    parser.add_argument("--development-oracle-fallback", action="store_true")
    return parser


def supervised_options(args: argparse.Namespace):
    from development.hanging.supervised import SupervisedOptions

    options = SupervisedOptions(
        args.model,
        args.seed,
        args.max_actions,
        args.chunk_size,
        args.time_limit_minutes,
        args.max_turns,
        complementary_view=args.complementary_view,
        perception_view=args.perception_view,
        hand_back_to_policy=args.hand_back_to_policy,
        policy_grace_actions=args.policy_grace_actions,
        grasp_recovery=args.grasp_recovery,
        grasp_retries=args.grasp_retries,
        empty_hand_opening=args.empty_hand_opening,
        development_oracle_fallback=args.development_oracle_fallback,
    )
    options.validate()
    return options


def _state_path(target: Path) -> Path:
    return target.with_name(f"{target.name}.modal-call.json")


def _write_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.part")
    temporary.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _call_may_still_be_running(error: BaseException) -> bool:
    """Distinguish a lost waiter from a terminal remote function failure."""
    if isinstance(error, (KeyboardInterrupt, SystemExit, GeneratorExit)):
        return True
    try:
        from modal import exception as modal_error
    except ImportError:
        return isinstance(error, (TimeoutError, ConnectionError))
    terminal = (
        modal_error.ExecutionError,
        modal_error.FunctionTimeoutError,
        modal_error.OutputExpiredError,
        modal_error.RemoteError,
    )
    if isinstance(error, terminal):
        return False
    transient = (
        ConnectionError,
        TimeoutError,
        modal_error.ConnectionError,
        modal_error.ServiceError,
        modal_error.TimeoutError,
    )
    return isinstance(error, transient)


def _finish_call(state_path, state, call, *, wait_for_volume, download_run):
    """Wait for a detached call, persist its result, and resume artifact download."""
    print(f"Waiting for detached Modal call {state['call_id']}", flush=True)
    try:
        report = call.get()
    except BaseException as exc:
        detached = _call_may_still_be_running(exc)
        state.update(
            status="detached" if detached else "failed",
            local_error_type=type(exc).__name__,
        )
        _write_state(state_path, state)
        if detached:
            print(
                "The remote call remains detached. Resume with:\n"
                f"  conda run -n robotwin python -m modal_apps.deployed_supervised_hanging "
                f"--resume-state {state_path}",
                flush=True,
            )
        else:
            print(
                f"The remote call failed ({type(exc).__name__}); it cannot be resumed.",
                flush=True,
            )
        raise

    state.update(status="completed", result_status=report.get("status"))
    state.pop("local_error_type", None)
    _write_state(state_path, state)
    print(json.dumps(report, indent=2))
    if wait_for_volume(state["run_id"]):
        download_run(
            state["run_id"],
            Path(state["output"]),
            skip_observations=state["skip_observations"],
            resume=True,
        )
    print(
        f"Video, RGB-D and report: {state['output']}; "
        f"remote robotwin-runs/{state['run_id']}"
    )
    return report


def launch(args, *, spawn, wait_for_volume, download_run, now=None, nonce=None):
    """Spawn one deployed episode; dependencies are injectable for CPU tests."""
    options = supervised_options(args)
    now = time.gmtime() if now is None else now
    nonce = time.time_ns() if nonce is None else nonce
    run_id = f"supervised_hanging_seed{args.seed}_{time.strftime('%Y%m%dT%H%M%SZ', now)}_{nonce}"
    target = Path(args.output).resolve() if args.output else REPO / "data/runs" / run_id
    if target.exists():
        raise ValueError("supervised_output_already_exists")
    state_path = _state_path(target)
    if state_path.exists():
        raise ValueError("supervised_call_state_already_exists")

    print(
        f"Invoking deployed {APP_NAME}/{FUNCTION_NAME}; seed={args.seed}; "
        f"actions<={args.max_actions}; limit={args.time_limit_minutes} min",
        flush=True,
    )
    call = spawn(run_id, options.__dict__)
    state = {
        "schema_version": 1,
        "status": "spawned",
        "app": APP_NAME,
        "function": FUNCTION_NAME,
        "call_id": call.object_id,
        "run_id": run_id,
        "output": str(target),
        "skip_observations": args.skip_observations,
        "options": options.__dict__,
    }
    _write_state(state_path, state)
    print(f"Remote run id: {run_id}", flush=True)
    print(f"Detached call state: {state_path}", flush=True)
    return _finish_call(
        state_path,
        state,
        call,
        wait_for_volume=wait_for_volume,
        download_run=download_run,
    )


def resume(state_path, *, lookup_call, wait_for_volume, download_run):
    state_path = Path(state_path).resolve()
    state = json.loads(state_path.read_text())
    if state.get("schema_version") != 1 or not all(
        state.get(key) for key in ("call_id", "run_id", "output")
    ):
        raise ValueError("invalid_supervised_call_state")
    if state.get("status") == "failed":
        raise ValueError("supervised_remote_call_failed")
    call = lookup_call(state["call_id"])
    return _finish_call(
        state_path,
        state,
        call,
        wait_for_volume=wait_for_volume,
        download_run=download_run,
    )


def main(argv=None):
    import modal
    from modal_apps.simulator import _download_run, _wait_for_volume

    args = build_parser().parse_args(argv)
    if args.resume_state is not None:
        resume(
            args.resume_state,
            lookup_call=modal.FunctionCall.from_id,
            wait_for_volume=_wait_for_volume,
            download_run=_download_run,
        )
        return
    if not args.model:
        raise SystemExit("--model is required unless --resume-state is supplied")
    deployed_function = modal.Function.from_name(APP_NAME, FUNCTION_NAME)
    launch(
        args,
        spawn=deployed_function.spawn,
        wait_for_volume=_wait_for_volume,
        download_run=_download_run,
    )


if __name__ == "__main__":
    main()
