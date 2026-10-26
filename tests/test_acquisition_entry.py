import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from development.checkpoints.acquisition_entry import AcquisitionEntry, replay_entry, HistoryPrimedVLA
from harness.evaluation.manifest import runtime_identity
from harness.planner.scripted import GraspRetryPlanner, GraspRetrySettings
from harness.predicates.hanging_mug import HangingMugPredicates
from harness.predicates.registry import PredicateRegistry
from harness.perception.observations import ObservationStore
from harness.runtime.orchestrator import SkillOrchestrator
from tests.test_analytic_skills import executor
from tests.test_skill_orchestrator import AcquisitionEnvironment


ROOT=Path(__file__).resolve().parents[1]


def source(directory):
    directory.mkdir()
    env=AcquisitionEnvironment(1)
    oracle=[]
    def sample(obs):
        value=env.evidence(obs)
        oracle.append(dict(event_type='predicate_oracle_evidence',privileged=value.model_dump(mode='json')))
        return value
    gates=PredicateRegistry(HangingMugPredicates(sample,source='development_oracle'))
    runtime,_,events=executor(env,predicates=gates,oracle=True)
    summary=SkillOrchestrator(runtime,GraspRetryPlanner(GraspRetrySettings(stage_before_first_vla=False)),
        episode_id='entry',seed=3,log=lambda e:None,
        save_observation=ObservationStore(directory/'observations').save).run()
    (directory/'manifest.json').write_text(json.dumps({'seed':3,'completion_predicate':'mug_lifted'}))
    (directory/'summary.json').write_text(json.dumps(summary))
    for name,rows in [('skill_events.jsonl',events),('oracle.jsonl',oracle)]:
        (directory/name).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    identity=directory/'runtime.json'
    identity.write_text(json.dumps(runtime_identity(ROOT)))
    return AcquisitionEntry(directory,identity,root=ROOT)


class ReplayEnvironment(AcquisitionEnvironment):
    def __init__(self, *, lifts=True, drift=False):
        super().__init__(1 if lifts else None)
        self.resets=0
        self.drift=drift
    def reset(self,seed):
        self.resets+=1
        observation=self.observe()
        if self.drift:
            return type(observation)(observation.sequence,observation.images,(2.,)*14)
        return observation


class EntryTests(unittest.TestCase):
    def test_complete_replay_verifies_lift_and_collects_only_chunk_boundary_observations(self):
        with tempfile.TemporaryDirectory() as tmp:
            entry=source(Path(tmp)/'source')
            env=ReplayEnvironment()
            gates=PredicateRegistry(HangingMugPredicates(env.evidence,source='development_oracle'))
            report,history=replay_entry(entry,env,gates,log=lambda e:None)
            self.assertTrue(report['accepted'])
            self.assertFalse(report['full_state_restored'])
            self.assertEqual((env.resets,report['replayed_actions']),(1,2))
            self.assertEqual([o.sequence for o,n in history],[0])

    def test_drift_or_missing_lift_never_accepts_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            entry=source(Path(tmp)/'source')
            for env in (ReplayEnvironment(drift=True),ReplayEnvironment(lifts=False)):
                gates=PredicateRegistry(HangingMugPredicates(env.evidence,source='development_oracle'))
                report,_=replay_entry(entry,env,gates,log=lambda e:None)
                self.assertFalse(report['accepted'])
                if env.drift:self.assertEqual(env.policy_steps,0)

    def test_unconfirmed_prefix_and_analytic_prefix_are_rejected(self):
        for kind in ('unconfirmed','analytic'):
            with tempfile.TemporaryDirectory() as tmp:
                entry=source(Path(tmp)/'source')
                p=entry.episode/'skill_events.jsonl'
                events=[json.loads(x) for x in p.read_text().splitlines()]
                if kind=='unconfirmed':
                    events=[e for e in events if not (e.get('event_type')=='action_confirmed' and e['observation']==2)]
                else:events.append({'event_type':'analytic_start'})
                p.write_text(''.join(json.dumps(e)+'\n' for e in events))
                with self.assertRaises(ValueError):
                    AcquisitionEntry(entry.episode,entry.episode/'runtime.json',root=ROOT)

    def test_changed_source_or_simulator_provenance_is_rejected_before_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            entry=source(Path(tmp)/'source')
            identity=entry.episode/'runtime.json'
            data=json.loads(identity.read_text());data['files']['run_inference_chunked.py']='0'*64
            identity.write_text(json.dumps(data))
            with self.assertRaises(ValueError):entry.verify_sources()
            with self.assertRaises(ValueError):AcquisitionEntry(entry.episode,identity,root=ROOT)

    def test_history_predictions_are_discarded_and_live_inference_waits_for_bootstrap(self):
        class Backend:
            def __init__(self):self.events=[]
            def reset_history(self):self.events.append('reset')
            def infer(self,obs,size):
                self.events.append((obs.sequence,size))
                return np.zeros((size,14))
        backend=Backend(); events=[]
        worker=HistoryPrimedVLA(backend,[(SimpleNamespace(sequence=0),8),(SimpleNamespace(sequence=8),8)],log=events.append)
        with self.assertRaises(RuntimeError):worker.infer(SimpleNamespace(sequence=10),8)
        worker.reset_history()
        worker.infer(SimpleNamespace(sequence=10),8)
        self.assertEqual(backend.events,['reset',(0,8),(8,8),(10,8)])
        self.assertEqual((worker.warm_calls,worker.discarded_predictions),(2,16))
        self.assertEqual(len(events),4)
        with self.assertRaises(RuntimeError):worker.reset_history()

    def test_failed_history_reset_is_not_retried(self):
        class Backend:
            def reset_history(self):raise ConnectionError('reset response lost')
        worker=HistoryPrimedVLA(Backend(),[],log=lambda e:None)
        with self.assertRaises(ConnectionError):worker.reset_history()
        with self.assertRaises(RuntimeError):worker.reset_history()
        self.assertFalse(worker.ready)
