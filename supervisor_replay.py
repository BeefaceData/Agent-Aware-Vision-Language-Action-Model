"""Versioned, offline supervisor fixtures keyed by complete request identity."""

from copy import deepcopy

from episode_harness import ActionProposal, SupervisorResponseError
from supervisor_response import SupervisorResponseDecoder


class RecordedSupervisor:
    """Use as ``run_episode(supervisor_decider=adapter)``.

    The JSON-compatible bundle has schema_version=1 and a fixtures list. Each
    entry has episode_id, observation_sequence, proposal_id, and either response
    (a structured response) or error (the literal 'provider_error'). Identities
    are exact: no positional matching, implicit rebinding, or provider fallback.
    Repeated calls for the same identity return the same detached decision.
    """

    def __init__(self, bundle: dict, decoder: SupervisorResponseDecoder | None = None):
        if (type(bundle) is not dict or set(bundle) != {'schema_version', 'fixtures'} or
                type(bundle['schema_version']) is not int or bundle['schema_version'] != 1 or
                type(bundle['fixtures']) is not list):
            raise SupervisorResponseError('invalid recorded supervisor bundle version or fields')
        self._decoder = decoder or SupervisorResponseDecoder()
        self._fixtures = {}
        for entry in deepcopy(bundle['fixtures']):
            identity_fields = {'episode_id', 'observation_sequence', 'proposal_id'}
            if (type(entry) is not dict or
                    set(entry) not in (identity_fields | {'response'}, identity_fields | {'error'})):
                raise SupervisorResponseError('invalid recorded supervisor fixture fields')
            key = self._identity(entry['episode_id'], entry['observation_sequence'],
                                 entry['proposal_id'])
            if key in self._fixtures:
                raise SupervisorResponseError('duplicate recorded supervisor request identity')
            if 'error' in entry and entry['error'] != 'provider_error':
                raise SupervisorResponseError('unsupported recorded supervisor error')
            self._fixtures[key] = entry

    @staticmethod
    def _identity(episode, sequence, proposal):
        if (type(episode) is not str or not episode or
                type(sequence) is not int or sequence < 0 or
                type(proposal) is not str or not proposal):
            raise SupervisorResponseError('invalid recorded supervisor request identity')
        return episode, sequence, proposal

    def __call__(self, proposal: ActionProposal):
        key = self._identity(proposal.observation.episode_id,
                             proposal.observation.sequence, proposal.proposal_id)
        entry = self._fixtures.get(key)
        if entry is None:
            raise SupervisorResponseError('missing recorded supervisor request identity')
        if 'error' in entry:
            raise SupervisorResponseError('recorded supervisor provider_error')
        return self._decoder.decode(deepcopy(entry['response']), proposal)
