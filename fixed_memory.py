"""Read-only decision memory bound to one externally pinned snapshot."""

from copy import deepcopy
from pathlib import Path

from decision_memory import DecisionMemory
from intervention_memory import InterventionMemory
from memory_snapshot import MemorySnapshot
from recorded_replay import TraceError


class FixedMemory:
    """Pass as ChronologicalVlmAdapter's decision_memory for fixed evaluation.

    The callback supplies only current task, control, progress and diagnosis.
    Membership and budgets always come from the verified snapshot. Append is
    explicitly rejected; other stores may record episodes without admitting them
    to this view. This is an API boundary, not filesystem access control.
    """

    def __init__(self, path, *, expected_reference, query):
        if not callable(query):
            raise ValueError('memory query callback required')
        self._path = Path(path).resolve()
        self._snapshot = MemorySnapshot(self._path, expected_reference=expected_reference)
        self._query = query

    @property
    def reference(self):
        return self._snapshot.reference

    def append(self, *, trace, attempt, supervisor, context):
        """Reject admission, including otherwise valid completed episodes."""
        raise TraceError('fixed-memory evaluation is read-only')

    def prepare(self, proposal):
        """Verify sealed evidence on each decision and return detached provenance."""
        query = deepcopy(self._query(deepcopy(proposal)))
        if type(query) is not dict or set(query) != {
                'task', 'robot_capabilities', 'progress_context', 'failure_category'}:
            raise ValueError('fixed-memory query requires only current applicability fields')
        document = self._snapshot.read()
        query.update({key: document['retrieval'][key] for key in (
            'max_entries', 'max_summary_bytes', 'max_context_bytes')})
        store = InterventionMemory(self._path.parent / document['store'])
        result = DecisionMemory(store, document['records'], lambda _: query,
                                snapshot=self.reference).prepare(proposal)
        # Detect manifest or source drift during retrieval as well as before it.
        self._snapshot.read()
        return result
