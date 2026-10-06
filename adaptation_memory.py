"""Episode-boundary admission for separately declared adaptation experiments."""

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from threading import RLock

from fixed_memory import FixedMemory
from intervention_memory import InterventionMemory
from memory_snapshot import MemorySnapshot
from recorded_replay import TraceError, load_recorded_replay


class AdaptationMemory:
    """Keep retrieval fixed during an episode; publish admission only after sealing.

    Pass this view as the chronological adapter's decision_memory. The first
    prepare begins an episode, or the host can call begin_episode explicitly.
    A separate recorder/store retains evidence. complete_episode verifies its
    pinned terminal trace and selected record pins before sealing a new view.
    This API does not authorize an experiment or certify data split permissions.
    """

    def __init__(self, path, *, expected_reference, query):
        self._path = Path(path).resolve()
        self._query = query
        self._view = FixedMemory(self._path, expected_reference=expected_reference, query=query)
        self._starting = self._view.reference
        self._episode = None
        self._lock = RLock()

    @property
    def reference(self):
        with self._lock:
            return self._view.reference

    def _document(self):
        return MemorySnapshot(self._path, expected_reference=self.reference).read()

    @staticmethod
    def _completed(document):
        metadata = document['metadata']
        if 'completed_episodes' in metadata:
            return deepcopy(metadata['completed_episodes'])
        # The previous format retained only its most recent completion.
        previous = metadata.get('adaptation')
        return ([] if previous is None else [{key: previous[key] for key in
                ('episode_id', 'trace_sha256', 'admitted')}])

    def begin_episode(self, episode_id):
        """Require a fresh attempt identity and preserve the current membership."""
        with self._lock:
            if type(episode_id) is not str or not episode_id.strip():
                raise TraceError('adaptation requires an episode identity')
            if self._episode is not None:
                raise TraceError('complete the active adaptation episode first')
            document = self._document()
            if any(item['episode_id'] == episode_id for item in self._completed(document)):
                raise TraceError('episode cannot retrieve its own experience or repeat exposure')
            store = InterventionMemory(self._path.parent / document['store'])
            for ref in document['records']:
                record = store.read(ref['record_id'], expected_sha256=ref['sha256'])
                if record['episode_id'] == episode_id:
                    raise TraceError('episode cannot retrieve its own experience')
            self._episode = episode_id

    def prepare(self, proposal):
        """Retrieve only the pinned pre-episode membership, never pending writes."""
        with self._lock:
            episode_id = proposal.observation.episode_id
            if self._episode is None:
                self.begin_episode(episode_id)
            if episode_id != self._episode:
                raise TraceError('complete the active adaptation episode first')
            return self._view.prepare(proposal)

    def complete_episode(self, *, trace, references, snapshot_path):
        """Admit selected records from this sealed episode, returning lineage.

        trace is (manifest_path, expected_sha256). references are record pins
        already written by InterventionMemory, not caller-authored outcomes.
        Empty admission still records an exposure and its verified terminal trace.
        A failed validation/write leaves retrieval on the old snapshot. Preserve
        the returned pin separately; restart explicitly from that pinned manifest.
        A verified retry of a retained completion returns replayed=True and the
        unchanged current pin without beginning an episode or writing a snapshot.
        """
        with self._lock:
            try:
                trace_path, digest = trace
                trace_path = Path(trace_path).resolve()
                raw = trace_path.read_bytes()
                if trace_path.name != 'manifest.json' or sha256(raw).hexdigest() != digest:
                    raise TraceError('adaptation requires a pinned sealed trace')
                replay = load_recorded_replay(trace_path.parent)
                episode_id = replay.evidence()['source_episode_id']
                if self._episode is not None and episode_id != self._episode:
                    raise TraceError('sealed trace does not match active episode')
                document = self._document()
                store = InterventionMemory(self._path.parent / document['store'])
                refs = deepcopy(list(references))
                for ref in refs:
                    record = store.read(ref['record_id'], expected_sha256=ref['sha256'])
                    source = record['provenance']['trace']
                    if (record['episode_id'] != episode_id or
                            source['sha256'] != digest or
                            (store.directory / source['path']).resolve() != trace_path):
                        raise TraceError('admission record does not match completed episode')
                if trace_path.read_bytes() != raw:
                    raise TraceError('episode trace changed during admission')
                refs.sort(key=lambda ref: ref['record_id'])
                if len({ref['record_id'] for ref in refs}) != len(refs):
                    raise TraceError('duplicate admission record pin')
                completed = self._completed(document)
                receipt = {'episode_id': episode_id, 'trace_sha256': digest,
                           'admitted': refs}
                previous = next((item for item in completed
                                 if item['episode_id'] == episode_id), None)
                if previous is not None:
                    if previous != receipt:
                        raise TraceError('conflicting completed episode admission')
                    # Reverify evidence above even on a retry. No publication,
                    # exposure increment, or retrieval mutation is necessary.
                    return deepcopy(dict(receipt, mode='adaptation',
                        starting_snapshot=self._starting, before=self.reference,
                        after=self.reference, replayed=True))
                if self._episode is None:
                    raise TraceError('no active adaptation episode')
                before = self.reference
                lineage = {'mode': 'adaptation', 'starting_snapshot': self._starting,
                    'before': before, 'episode_id': self._episode,
                    'trace_sha256': digest,
                    'admitted': sorted(refs, key=lambda ref: ref['record_id'])}
                snapshot = MemorySnapshot.freeze(snapshot_path, store,
                    document['records'] + refs,
                    metadata={'adaptation': lineage,
                              'completed_episodes': completed + [receipt]},
                    retrieval=document['retrieval'])
                view = FixedMemory(snapshot_path, expected_reference=snapshot.reference,
                                   query=self._query)
            except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
                raise TraceError(f'invalid adaptation admission: {exc}') from exc
            self._path = Path(snapshot_path).resolve()
            self._view = view
            self._episode = None
            return deepcopy(dict(lineage, after=snapshot.reference))
