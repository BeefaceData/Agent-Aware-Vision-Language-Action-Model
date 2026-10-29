import unittest
from harness.sampling import Action,Config,Evidence,InspectionScheduler

class SamplingTests(unittest.TestCase):
    def inspect(self,s,t,e=Evidence('grasp'),seq=None):
        seq=t if seq is None else seq
        self.assertEqual(s.tick(t,seq).action,Action.INSPECT)
        return s.observe(t,seq,e)

    def test_uniform_exact_boundaries(self):
        s=InspectionScheduler(Config(mode='uniform',max_gap=16,cheap_triggers=False))
        self.inspect(s,0)
        for t in range(1,16):self.assertEqual(s.tick(t,t).action,Action.SKIP)
        self.inspect(s,16);self.assertEqual(s.inspections,2)

    def test_supported_table_is_frozen_and_detached(self):
        table={'grasp':[0,1]};c=Config(supported_bins=table)
        table['grasp'].append(2)
        self.assertEqual(c.supported_bins['grasp'],(0,1))
        with self.assertRaises(TypeError):c.supported_bins['grasp']=(0,1,2)

    def test_unknown_and_retry_override_sparse_interval(self):
        for evidence in (Evidence('hang','unknown'),Evidence('hang',retry=True),Evidence(None)):
            s=InspectionScheduler(Config(mode='uniform',max_gap=64))
            self.inspect(s,0,evidence);self.assertEqual(s.next_due,16)

    def test_supported_bins_required_for_entire_gap(self):
        s=InspectionScheduler(Config(supported_bins={'grasp':(0,)}))
        self.inspect(s,0);self.inspect(s,16);self.assertEqual(s.next_due,32)
        s=InspectionScheduler(Config(supported_bins={'grasp':(0,1,2)}))
        self.inspect(s,0);self.inspect(s,16);self.assertEqual(s.next_due,48)

    def test_phase_clock_starts_at_observed_transition(self):
        s=InspectionScheduler(Config(supported_bins={'grasp':(0,1,2)}))
        self.inspect(s,0);self.inspect(s,16);self.inspect(s,48,Evidence('handoff'))
        self.assertEqual((s.phase_start,s.next_due),(48,64))

    def test_no_unsampled_labels_in_tick_interface(self):
        s=InspectionScheduler(Config())
        with self.assertRaises(TypeError):s.tick(0,0,phase='hang')
        with self.assertRaises(RuntimeError):s.observe(0,0,Evidence('hang'))

    def test_hard_guard_preempts_due_duplicate_and_pending(self):
        s=InspectionScheduler(Config())
        self.assertEqual(s.tick(0,0).action,Action.INSPECT)
        self.assertEqual(s.tick(0,0,hard_failure=True).action,Action.STOP)
        self.assertEqual(s.inspections,0)

    def test_gripper_trigger_and_cooldown(self):
        s=InspectionScheduler(Config(mode='uniform',max_gap=64))
        s.tick(0,0,grippers=(1,1));s.observe(0,0,Evidence('grasp'))
        self.assertEqual(s.tick(1,1,grippers=(0,1)).action,Action.SKIP)
        self.assertEqual(s.next_due,4)
        self.assertEqual(s.tick(4,4,grippers=(0,1)).action,Action.INSPECT)

    def test_stale_sequence_never_causes_duplicate_model_request(self):
        s=InspectionScheduler(Config());self.inspect(s,0)
        self.assertEqual(s.tick(16,0).action,Action.STOP)
        self.assertEqual(s.inspections,1)

    def test_failure_stops_without_scheduling_more(self):
        s=InspectionScheduler(Config());self.inspect(s,0,Evidence('grasp',failure=True))
        self.assertEqual(s.tick(1,1).action,Action.STOP)

    def test_reject_invalid_config_and_clock(self):
        for kw in ({'base_gap':0},{'max_gap':True},{'max_gap':8},{'supported_bins':{'secret':(0,)}}):
            with self.assertRaises(ValueError):Config(**kw)
        s=InspectionScheduler(Config());self.inspect(s,0)
        s.tick(3,3)
        with self.assertRaises(ValueError):s.tick(2,2)

    def test_persistent_failure_delay_bound_for_every_offset(self):
        for onset in range(1,65):
            s=InspectionScheduler(Config(mode='uniform',max_gap=16,cheap_triggers=False))
            detected=None
            for t in range(100):
                if s.tick(t,t).action==Action.INSPECT:
                    if s.observe(t,t,Evidence('grasp',failure=t>=onset)).action==Action.STOP:
                        detected=t;break
            self.assertLessEqual(detected-onset,15)

    def test_preset_and_uniform_are_separate_cadences(self):
        s=InspectionScheduler(Config(mode='preset',cheap_triggers=False))
        self.inspect(s,0,Evidence('handoff'));self.assertEqual(s.next_due,128)
        self.inspect(s,128,Evidence('handoff'));self.assertEqual(s.next_due,144)

if __name__=='__main__':unittest.main()
