"""Causal inspection scheduling, independent of provider, policy and simulator.

This schedules semantic inspections only. It neither skips policy conditioning
frames nor weakens per-action execution guards. An inspector supplies evidence
ONLY after tick requests inspection. Phase time starts when a phase is observed,
not when an evaluator later says it began. No probability guarantee is implied
by a development-derived safe-age table.
"""
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping

PHASES = ('grasp', 'handoff', 'hang')

class Action(str, Enum):
    SKIP='skip'
    INSPECT='inspect'
    STOP='stop'

@dataclass(frozen=True)
class Decision:
    action: Action
    reason: str

@dataclass(frozen=True)
class Evidence:
    phase: str | None
    quality: str = 'clear'
    failure: bool = False
    retry: bool = False

    def __post_init__(self):
        if self.phase not in (*PHASES, None) or self.quality not in ('clear','unknown'):
            raise ValueError('invalid_inspection_evidence')
        if type(self.failure) is not bool or type(self.retry) is not bool:
            raise ValueError('invalid_evidence_flags')

@dataclass(frozen=True)
class Config:
    mode: str = 'bounded_hybrid'
    base_gap: int = 16
    max_gap: int = 32
    event_cooldown: int = 4
    age_bin: int = 32
    # Only bins admitted on development data; serializable, no runtime oracle.
    supported_bins: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    cheap_triggers: bool = True

    def __post_init__(self):
        if self.mode not in ('uniform','preset','bounded_hybrid'):
            raise ValueError('invalid_sampling_mode')
        for value in (self.base_gap,self.max_gap,self.event_cooldown,self.age_bin):
            if type(value) is not int or value<1:raise ValueError('invalid_sampling_interval')
        if self.max_gap<self.base_gap or self.event_cooldown>self.base_gap:
            raise ValueError('invalid_sampling_bounds')
        if type(self.cheap_triggers) is not bool:raise ValueError('invalid_trigger_flag')
        if any(k not in PHASES or any(type(v) is not int or v<0 for v in bins)
               for k,bins in self.supported_bins.items()):
            raise ValueError('invalid_supported_bins')
        object.__setattr__(self,'supported_bins',MappingProxyType({
            k:tuple(sorted(set(v))) for k,v in self.supported_bins.items()}))

class InspectionScheduler:
    """Two-stage interface prevents accidental access to unsampled labels."""
    def __init__(self, config: Config):
        self.config=config
        self.phase=None; self.phase_start=0; self.next_due=0
        self.last_action=-1; self.last_inspection=-1; self.last_sequence=-1
        self.gripper_sides=None; self.pending=None; self.stopped=False
        self.inspections=0; self.guard_checks=0; self.cheap_checks=0

    def tick(self, action: int, sequence: int, *, grippers=None, hard_failure=False):
        if type(action) is not int or type(sequence) is not int or action<0 or sequence<0:
            raise ValueError('invalid_clock')
        if action<self.last_action:raise ValueError('nonmonotonic_action_clock')
        if type(hard_failure) is not bool:raise ValueError('invalid_guard_flag')
        self.guard_checks+=1
        if hard_failure:
            self.stopped=True;self.pending=None
            return Decision(Action.STOP,'hard_failure')
        if self.stopped:return Decision(Action.STOP,'already_stopped')
        if self.pending is not None:raise RuntimeError('inspection_result_required')
        if action==self.last_action:return Decision(Action.SKIP,'duplicate_tick')
        self.last_action=action
        if self.config.cheap_triggers and grippers is not None:
            if len(grippers)!=2 or any(not 0<=v<=1 for v in grippers):
                raise ValueError('invalid_gripper_commands')
            self.cheap_checks+=1
            sides=tuple(v>=0.5 for v in grippers)
            if self.gripper_sides is not None and sides!=self.gripper_sides:
                self.next_due=min(self.next_due,max(action,self.last_inspection+self.config.event_cooldown))
            self.gripper_sides=sides
        if action<self.next_due:return Decision(Action.SKIP,'not_due')
        if sequence<=self.last_sequence:
            self.stopped=True
            return Decision(Action.STOP,'stale_observation_at_due_inspection')
        self.pending=(action,sequence)
        return Decision(Action.INSPECT,'due')

    def observe(self, action: int, sequence: int, evidence: Evidence):
        if self.pending!=(action,sequence):raise RuntimeError('unsolicited_or_mismatched_inspection')
        if not isinstance(evidence,Evidence):raise ValueError('typed_evidence_required')
        self.pending=None;self.inspections+=1
        self.last_inspection=action;self.last_sequence=sequence
        if evidence.failure:
            self.stopped=True
            return Decision(Action.STOP,'observed_failure')
        changed=evidence.phase is not None and evidence.phase!=self.phase
        if changed or evidence.retry:
            self.phase=evidence.phase or self.phase;self.phase_start=action
        c=self.config;gap=c.base_gap
        if evidence.quality=='unknown' or evidence.phase is None or evidence.retry:
            gap=c.base_gap
        elif c.mode=='uniform':gap=c.max_gap
        elif c.mode=='preset':
            gap={'grasp':64,'handoff':128,'hang':64}[self.phase] if changed else c.base_gap
        elif not changed:
            age=action-self.phase_start
            # Every bin traversed by an expanded interval must have support.
            bins=set(range(age//c.age_bin,(age+c.max_gap)//c.age_bin+1))
            if bins.issubset(set(c.supported_bins.get(self.phase,()))):gap=c.max_gap
        self.next_due=action+gap
        return Decision(Action.SKIP,'uncertain_recheck' if evidence.quality=='unknown' else 'scheduled')
