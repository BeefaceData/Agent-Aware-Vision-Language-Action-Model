"""Causal alarm-only shadow replay with interval-censored public-video labels."""
import argparse
import hashlib
import json
from pathlib import Path
from harness.sampling import Action,Config,Evidence,InspectionScheduler,PHASES

def intersects(a,b):return a[0]<=b[1] and b[0]<=a[1]

def validate_episode(ep):
    if ep['split'] not in ('development','test'):raise ValueError('invalid_split')
    if type(ep['seed']) is not int or type(ep['end']) is not int or ep['end']<0:
        raise ValueError('invalid_episode')
    entries=ep['phase_entries']
    if not entries or entries[0]!={'phase':'grasp','lo':0,'hi':0}:
        raise ValueError('initial_phase_required')
    if any(e['phase'] not in PHASES or not 0<=e['lo']<=e['hi']<=ep['end'] for e in entries):
        raise ValueError('invalid_phase_bounds')
    if entries!=sorted(entries,key=lambda x:x['lo']):raise ValueError('unordered_phases')
    names=set()
    for e in ep['events']:
        if e['name'] in names:raise ValueError('duplicate_event')
        names.add(e['name'])
        if e['kind'] not in ('milestone','failure','retry') or not 0<=e['lo']<=e['hi']<=e['end']<=ep['end']:
            raise ValueError('invalid_event_bounds')
    if any(not 0<=lo<=hi<=ep['end'] for lo,hi in ep.get('unknown',[])):
        raise ValueError('invalid_unknown_bounds')

class Inspector:
    """The only label access in a replay; called only for requested inspections."""
    def __init__(self, episode):
        validate_episode(episode);self.episode=episode;self.calls=[];self.seen=set()

    def inspect(self,t):
        self.calls.append(t);ep=self.episode
        ambiguous=any(lo<=t<=hi for lo,hi in ep.get('unknown',[]))
        ambiguous|=any(e['lo']<=t<e['hi'] for e in ep['events'])
        ambiguous|=any(e['lo']<=t<e['hi'] for e in ep['phase_entries'])
        phase='grasp'
        for e in ep['phase_entries']:
            if e['hi']<=t:phase=e['phase']
        observed=[];new_retry=False;failure=False
        if not ambiguous:
            for event in ep['events']:
                if event['hi']<=t<=event['end']:
                    observed.append(event['name'])
                    if event['kind'] in ('failure','retry') and event['name'] not in self.seen:
                        new_retry=True
                    failure|=event['kind']=='failure'
                    self.seen.add(event['name'])
        # Alarm-only shadow analysis: leave the recorded baseline untouched.
        # Active failure forces conservative rechecks rather than sparse sensing.
        return Evidence(None if ambiguous else phase,
                        'unknown' if ambiguous or failure else 'clear',retry=new_retry),observed

def fit_support(episodes, *, age_bin=32,min_seeds=3,max_age=1200):
    """Conservative all-development veto with phase-origin lag envelopes."""
    if any(ep['split']!='development' for ep in episodes):raise ValueError('held_out_fit_forbidden')
    if len({ep['seed'] for ep in episodes})!=len(episodes):raise ValueError('duplicate_seed_in_fit')
    for ep in episodes:validate_episode(ep)
    admitted={p:[] for p in PHASES};audit=[]
    for phase in PHASES:
        for b in range(max_age//age_bin+1):
            support=[];veto=[]
            for ep in episodes:
                entries=ep['phase_entries'];entry=next((e for e in entries if e['phase']==phase),None)
                if entry is None:continue
                i=entries.index(entry)
                end=entries[i+1]['lo']-1 if i+1<len(entries) else ep['end']
                origin_hi=entry['hi']+(0 if i==0 else 32)
                # Occlusion can postpone the first observed phase far beyond a
                # single sampling gap. Such an entry supplies no narrow origin
                # support; conservatively let its clock start anywhere to end.
                obscured=any(intersects((entry['lo'],origin_hi),x) for x in ep.get('unknown',[]))
                obscured|=any(e['lo']<origin_hi and e['hi']>entry['hi'] for e in ep['events'])
                if obscured:origin_hi=max(origin_hi,end)
                span=(entry['lo']+b*age_bin,origin_hi+(b+1)*age_bin)
                if span[0]>ep['end']:continue
                # Possible onset/unknown windows are expanded by one max gap.
                excluded=[(max(0,e['lo']-32),(e['end'] if e['kind']=='failure' else e['hi'])+32)
                          for e in ep['events']]
                excluded += [(max(0,lo-32),hi+32) for lo,hi in ep.get('unknown',[])]
                excluded += [(max(0,e['lo']-32),e['hi']+32) for e in entries[1:]]
                if any(intersects(span,x) for x in excluded):veto.append(ep['seed'])
                elif span[0]>=entry['hi'] and span[1]<=end:support.append(ep['seed'])
            ok=len(support)>=min_seeds and not veto
            if ok:admitted[phase].append(b)
            audit.append(dict(phase=phase,bin=b,support_seeds=support,veto_seeds=veto,admitted=ok))
    return {k:tuple(v) for k,v in admitted.items()},audit

def configurations(support):
    return {
        'uniform16':Config(mode='uniform',max_gap=16,cheap_triggers=False),
        'uniform32':Config(mode='uniform',max_gap=32,cheap_triggers=False),
        'uniform64':Config(mode='uniform',max_gap=64,cheap_triggers=False),
        'preset':Config(mode='preset',cheap_triggers=False),
        'hybrid32':Config(supported_bins=support,cheap_triggers=False),
        'hybrid32_trigger':Config(supported_bins=support,cheap_triggers=True),
        'uniform16_trigger':Config(mode='uniform',max_gap=16,cheap_triggers=True),
    }

def config_dict(c):
    return dict(mode=c.mode,base_gap=c.base_gap,max_gap=c.max_gap,event_cooldown=c.event_cooldown,
                age_bin=c.age_bin,supported_bins=dict(c.supported_bins),cheap_triggers=c.cheap_triggers)

def join_guards(rows,guards):
    if [g['action_index'] for g in guards]!=[r['action_index'] for r in rows[1:]]:
        raise ValueError('incomplete_guard_trace')
    if any(type(g['arm_overlap']) is not bool for g in guards):raise ValueError('invalid_guard_trace')
    return [dict(rows[0],hard_failure=False)]+[
        dict(r,hard_failure=g['arm_overlap']) for r,g in zip(rows[1:],guards)]

def replay(ep,rows,config):
    inspector=Inspector(ep);scheduler=InspectionScheduler(config)
    inspections=[];first={};reasons=[];unknown=[]
    for row in rows:
        t=row['action_index']
        if t>ep['end']:break
        state=row['command_state']
        decision=scheduler.tick(t,row['sequence'],grippers=(state[6],state[13]),
                                hard_failure=row.get('hard_failure',False))
        if decision.action==Action.STOP:
            reasons.append(dict(action=t,reason=decision.reason));break
        if decision.action==Action.INSPECT:
            evidence,seen=inspector.inspect(t)
            inspections.append(t)
            for name in seen:first.setdefault(name,t)
            scheduler.observe(t,row['sequence'],evidence)
            if evidence.quality=='unknown':unknown.append(dict(action=t,next_due=scheduler.next_due))
    event_results=[]
    for event in ep['events']:
        d=first.get(event['name'])
        event_results.append(dict(**event,detected_at=d,missed=d is None,
            endpoint_censored=event['end']==ep['end'] and ep['end']-event['hi']<16,
            delay_low=None if d is None else d-event['hi'],
            delay_high=None if d is None else d-event['lo']))
    return dict(run=ep['run'],seed=ep['seed'],split=ep['split'],end=ep['end'],
        inspections=inspections,inspection_count=len(inspections),events=event_results,
        unknown_rechecks=unknown,guard_checks=scheduler.guard_checks,cheap_checks=scheduler.cheap_checks,
        stops=reasons,meaning='shadow inspection opportunities; no model decisions or trajectory changes')

def gate(reference,candidate):
    failures=[]
    if len(reference)!=len(candidate):raise ValueError('unmatched_episode_count')
    for ref,cand in zip(reference,candidate):
        if ref['seed']!=cand['seed']:raise ValueError('unmatched_episodes')
        if len(ref['events'])!=len(cand['events']):raise ValueError('unmatched_event_count')
        for r,c in zip(ref['events'],cand['events']):
            if r['name']!=c['name']:raise ValueError('unmatched_events')
            if c['missed'] and not r['missed']:
                failures.append(f"seed{ref['seed']}:{r['name']}:extra_miss")
            if c['kind']=='failure' and c['end']-c['hi']>=16 and c['missed']:
                failures.append(f"seed{ref['seed']}:{r['name']}:persistent_failure_missed")
            if r['kind']=='failure' and c['detected_at'] is not None and r['detected_at'] is not None:
                if c['delay_high']>r['delay_high']:failures.append(f"seed{ref['seed']}:{r['name']}:later_failure")
        if any(x['next_due']-x['action']>16 for x in cand['unknown_rechecks']):failures.append('unknown_gap')
    return dict(passed=not failures,violations=failures,
                inspections=sum(x['inspection_count'] for x in candidate))

def main():
    p=argparse.ArgumentParser();p.add_argument('--annotations',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--frozen',type=Path)
    p.add_argument('--protocol',type=Path,required=True)
    args=p.parse_args();episodes=json.loads(args.annotations.read_text())['episodes']
    args.out.mkdir(parents=True,exist_ok=True)
    if args.frozen:
        frozen=json.loads(args.frozen.read_text());support=frozen['support'];selected=frozen['selected']
        if any(e['split']!='test' for e in episodes):raise ValueError('heldout_only_evaluation')
    else:
        support,audit=fit_support(episodes)
        (args.out/'support-audit.json').write_text(json.dumps(audit,indent=2))
    configs=configurations(support);results={name:[] for name in configs}
    if args.frozen:
        actual_hash=hashlib.sha256(Path(__import__('harness.sampling',fromlist=['x']).__file__).read_bytes()).hexdigest()
        if actual_hash!=frozen['scheduler_sha256']:raise ValueError('frozen_scheduler_changed')
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=frozen['replay_sha256']:
            raise ValueError('frozen_replay_changed')
        if hashlib.sha256(args.protocol.read_bytes()).hexdigest()!=frozen['protocol_sha256']:
            raise ValueError('frozen_protocol_changed')
        actual_configs=json.loads(json.dumps({n:config_dict(c) for n,c in configs.items()}))
        if actual_configs!=frozen['configurations']:raise ValueError('frozen_configuration_changed')
        if set(frozen['development_seeds']) & {e['seed'] for e in episodes}:raise ValueError('heldout_seed_leakage')
    for ep in episodes:
        rows=[json.loads(x) for x in (args.data/ep['run']/'trace.jsonl').read_text().splitlines()]
        if [r['action_index'] for r in rows]!=list(range(ep['end']+1)):raise ValueError('incomplete_trace')
        guards=[json.loads(x) for x in (args.data/ep['run']/'guards.jsonl').read_text().splitlines()]
        rows=join_guards(rows,guards)
        for name,c in configs.items():results[name].append(replay(ep,rows,c))
    gates={name:gate(results['uniform16'],result) for name,result in results.items()}
    if not args.frozen:
        # Sparse uniform64 is stress/reference, not an adoption candidate.
        eligible=[n for n in ('uniform16','uniform32','hybrid32','hybrid32_trigger','uniform16_trigger') if gates[n]['passed']]
        priority={name:i for i,name in enumerate(('uniform16','uniform32','hybrid32','uniform16_trigger','hybrid32_trigger'))}
        def selection_key(name):
            delays=[e['delay_high'] for r in results[name] for e in r['events'] if e['delay_high'] is not None]
            return gates[name]['inspections'],max(delays,default=0),priority[name]
        selected=min(eligible,key=selection_key) if eligible else 'uniform16'
        frozen=dict(selected=selected,support=support,configurations={n:config_dict(c) for n,c in configs.items()},
                    feasible=bool(eligible),selection_note='retain conservative reference if no candidate passes',
                    development_seeds=[e['seed'] for e in episodes],
                    annotation_sha256=hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
                    replay_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    protocol_sha256=hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
                    scheduler_sha256=hashlib.sha256(Path(__import__('harness.sampling',fromlist=['x']).__file__).read_bytes()).hexdigest())
        (args.out/'frozen-design.json').write_text(json.dumps(frozen,indent=2))
    result=dict(selected=selected,gates=gates,results=results,held_out=bool(args.frozen))
    (args.out/'replay-results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(selected=selected,gates=gates),indent=2))

if __name__=='__main__':main()
