"""Replay a successful natural acquisition, verify entry, then test fresh handoff."""

import argparse
from functools import partial
import json
from pathlib import Path

from development.checkpoints.acquisition_entry import AcquisitionEntry, replay_entry, HistoryPrimedVLA
from development.checkpoints.bank import write_json_atomic
from harness.jsonl import append_jsonl
from harness.evaluation.manifest import runtime_identity
from harness.skills.schemas import VLAParameters, EpisodeBudget


def main(argv=None):
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--source-runtime', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    # All source validation precedes native simulator or policy construction.
    entry = AcquisitionEntry(args.episode, args.source_runtime, root=root)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    bounds = VLAParameters(max_chunks=32, chunk_size=8, max_policy_actions=256)
    write_json_atomic(output/'manifest.json', dict(schema_version=1,
        condition='recorded_acquisition_then_history_primed_handoff', seed=entry.seed,
        source_episode=str(entry.episode), source_hashes=entry.hashes,
        history_strategy='reset then infer/discard at replayed chunk boundaries before fresh handoff',
        full_state_restored=False, remote_random_state_restored=False,
        remote_weights_attested=False, handoff=bounds.model_dump(mode='json')))
    write_json_atomic(output/'runtime_identity.json', runtime_identity(root))
    from development.simulator.local_simulator import LocalSimulator
    from development.demos.episode_video import EpisodeVideo, RecordingEnvironment
    from development.simulator.skill_oracle import oracle_predicates
    from harness.skills import SkillExecutor
    from harness.vla.modal_client import ModalVLAClient
    from harness.planner.handoff import HandoffPlanner
    from harness.planner.scripted import GraspRetrySettings
    from harness.runtime.orchestrator import SkillOrchestrator
    from harness.perception.observations import ObservationStore
    simulator = video = backend = None
    summary = dict(outcome='entry_rejected', milestone_reached=False, task_success=False)
    try:
        video = EpisodeVideo(output)
        simulator = LocalSimulator(root/'RoboTwin')
        environment = RecordingEnvironment(simulator, video)
        environment.policy_kind = 'replay_acquisition'
        predicates = oracle_predicates(simulator.task, log=partial(append_jsonl, output/'oracle.jsonl'))
        restored, history = replay_entry(entry, environment, predicates,
            log=partial(append_jsonl, output/'replay_events.jsonl'))
        write_json_atomic(output/'restore.json', restored)
        if restored['accepted']:
            environment.policy_kind = 'live_handoff'
            backend = HistoryPrimedVLA(ModalVLAClient(), history,
                log=partial(append_jsonl, output/'history_events.jsonl'))
            executor = SkillExecutor(environment, backend, predicates=predicates,
                allow_development_oracle=True, log=partial(append_jsonl, output/'skill_events.jsonl'),
                budget=EpisodeBudget(max_skill_calls=4, max_policy_actions=264, max_vla_chunks=33,
                                     max_analytic_ticks=1))
            controller = SkillOrchestrator(executor,
                HandoffPlanner(GraspRetrySettings(stage_before_first_vla=False, max_retries=0), bounds),
                completion_predicate='handoff_confirmed', max_turns=2,
                episode_id=output.name, seed=entry.seed,
                log=partial(append_jsonl, output/'assessments.jsonl'),
                save_observation=ObservationStore(output/'observations').save)
            summary = controller.run()
        summary['restore'] = restored
    except Exception as exc:
        summary.update(outcome='runner_error', error_type=type(exc).__name__, milestone_reached=False)
    finally:
        if backend is not None:
            summary['history_bootstrap'] = dict(calls=backend.warm_calls,
                discarded_predictions=backend.discarded_predictions, ready=backend.ready,
                executed_predictions=0, remote_random_state_restored=False)
        if video is not None:
            try: video.close()
            except Exception as exc: summary['video_cleanup_error_type'] = type(exc).__name__
            summary['video'] = video.metadata()
        if simulator is not None:
            try: simulator.close()
            except Exception as exc: summary['cleanup_error_type'] = type(exc).__name__
        write_json_atomic(output/'summary.json', summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary.get('outcome') == 'handoff_confirmed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
