import copy
import unittest
from development.sampling.replay import Inspector,fit_support,replay,gate,join_guards
from harness.sampling import Config

def episode(seed=0):
    return dict(run=f'seed{seed}',seed=seed,split='development',end=160,
                phase_entries=[dict(phase='grasp',lo=0,hi=0)],events=[],unknown=[])

def rows(n=160):
    return [dict(action_index=i,sequence=i,command_state=[0]*6+[1]+[0]*6+[1]) for i in range(n+1)]

class ReplayTests(unittest.TestCase):
    def test_recorded_contact_stops_before_semantic_inspection(self):
        trace=rows();guards=[dict(action_index=i,arm_overlap=i==32) for i in range(1,161)]
        r=replay(episode(),join_guards(trace,guards),Config(mode='uniform',max_gap=16))
        self.assertEqual(r['inspections'],[0,16])
        self.assertEqual(r['stops'],[dict(action=32,reason='hard_failure')])
        with self.assertRaises(ValueError):join_guards(trace,guards[:-1])

    def test_future_labels_cannot_change_past_samples(self):
        a=episode();b=copy.deepcopy(a)
        b['events']=[dict(name='late_failure',kind='failure',lo=96,hi=104,end=160)]
        config=Config(mode='uniform',max_gap=32,cheap_triggers=False)
        ra=replay(a,rows(),config);rb=replay(b,rows(),config)
        self.assertEqual([t for t in ra['inspections'] if t<96],[t for t in rb['inspections'] if t<96])

    def test_inspection_identity_and_interval_delay(self):
        e=episode();e['events']=[dict(name='drop',kind='failure',lo=33,hi=39,end=160)]
        r=replay(e,rows(),Config(mode='uniform',max_gap=16,cheap_triggers=False))
        event=r['events'][0]
        self.assertEqual((event['detected_at'],event['delay_low'],event['delay_high']),(48,9,15))

    def test_short_transient_can_be_missed(self):
        e=episode();e['events']=[dict(name='transient',kind='failure',lo=17,hi=17,end=20)]
        self.assertTrue(replay(e,rows(),Config(mode='uniform',max_gap=16,cheap_triggers=False))['events'][0]['missed'])

    def test_reject_heldout_and_duplicate_seeds_in_fit(self):
        e=episode();e['split']='test'
        with self.assertRaises(ValueError):fit_support([e])
        with self.assertRaises(ValueError):fit_support([episode(),episode()])

    def test_support_requires_three_seeds_and_any_event_vetoes(self):
        self.assertFalse(any(fit_support([episode(0),episode(1)])[0].values()))
        samples=[episode(i) for i in range(3)]
        self.assertIn(0,fit_support(samples)[0]['grasp'])
        samples[2]['events']=[dict(name='early',kind='failure',lo=12,hi=20,end=100)]
        self.assertNotIn(0,fit_support(samples)[0]['grasp'])

    def test_phase_origin_lag_and_censoring_not_treated_as_safe(self):
        samples=[episode(i) for i in range(3)]
        for e in samples:
            e['phase_entries'].append(dict(phase='handoff',lo=80,hi=96))
            e['end']=110
        self.assertEqual(fit_support(samples)[0]['handoff'],())

    def test_occluded_entry_does_not_supply_narrow_phase_age_support(self):
        samples=[episode(i) for i in range(4)]
        for e in samples:
            e['end']=700;e['phase_entries'].append(dict(phase='handoff',lo=10,hi=10))
            e['unknown']=[[10,200]]
            e['events']=[dict(name='drop',kind='failure',lo=440,hi=450,end=700)]
        self.assertEqual(fit_support(samples)[0]['handoff'],())

    def test_persistent_failure_duration_not_admitted_as_clear_support(self):
        samples=[episode(i) for i in range(3)]
        samples[0]['events']=[dict(name='drop',kind='failure',lo=1,hi=2,end=160)]
        self.assertEqual(fit_support(samples)[0]['grasp'],())

    def test_unknown_window_forces_one_chunk(self):
        e=episode();e['unknown']=[[32,90]]
        r=replay(e,rows(),Config(mode='uniform',max_gap=32,cheap_triggers=False))
        self.assertEqual([x['action'] for x in r['unknown_rechecks']],[32,48,64,80])
        self.assertTrue(all(x['next_due']-x['action']==16 for x in r['unknown_rechecks']))

    def test_gate_rejects_late_failure(self):
        e=episode();e['events']=[dict(name='drop',kind='failure',lo=33,hi=33,end=160)]
        a=replay(e,rows(),Config(mode='uniform',max_gap=16,cheap_triggers=False))
        b=replay(e,rows(),Config(mode='uniform',max_gap=32,cheap_triggers=False))
        self.assertFalse(gate([a],[b])['passed'])

if __name__=='__main__':unittest.main()
