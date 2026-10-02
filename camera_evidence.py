"""Record camera frames with an index linking videos to observations.

The caller supplies video writers and frame conversion, so replay does not
import simulator, image encoding, NumPy, or model packages. This recorder owns
the writers it creates and the index file; close it after success or failure.
"""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from episode_harness import ObservationPacket


class FrameWriter(Protocol):
    def append_data(self, frame: Any) -> None: ...
    def close(self) -> None: ...


class CameraEvidenceRecorder:
    """Save each available view and index absent views without substitution."""

    def __init__(self, main_path: Path, wrist_path: Path, index_path: Path,
                 writer_factory: Callable[[Path], FrameWriter],
                 convert_frame: Callable[[Any], Any]):
        self._paths = {'main': main_path, 'wrist': wrist_path}
        self._index_path = index_path
        self._writer_factory = writer_factory
        self._convert_frame = convert_frame
        self._writers: dict[str, FrameWriter] = {}
        self._counts = {'main': 0, 'wrist': 0}
        self._resources = ExitStack()
        self._index = self._resources.enter_context(index_path.open('w', encoding='utf-8'))
        self._closed = False
        self._finalized = False

    def record(self, packet: ObservationPacket) -> None:
        if self._closed:
            raise RuntimeError('camera evidence recorder is closed')
        refs = {ref.camera: ref for ref in packet.frame_references}
        if (len(packet.frame_references) != 2 or set(refs) != {'main', 'wrist'} or
            not isinstance(packet.observation, Mapping)):
            raise ValueError('camera evidence requires main and wrist references')
        pixels = packet.observation.get('pixels')
        if not isinstance(pixels, Mapping):
            pixels = {}
        for camera in ('main', 'wrist'):
            ref = refs[camera]
            expected_key = 'pixels.image' if camera == 'main' else 'pixels.image2'
            if (ref.image_key != expected_key or
                ref.availability not in ('available', 'missing') or
                (ref.synchronization in ('verified', 'co_observed') and
                 ref.observation_sequence != packet.sequence)):
                raise ValueError('camera reference does not match observation')
            index = None
            path = self._paths[camera]
            if ref.availability == 'available':
                key = ref.image_key.split('.', 1)[1]
                if pixels.get(key) is None:
                    raise ValueError('available camera reference has no frame')
                if camera not in self._writers:
                    writer = self._writer_factory(path)
                    self._writers[camera] = writer
                    self._resources.callback(writer.close)
                index = self._counts[camera]
                self._writers[camera].append_data(
                    self._convert_frame(pixels[key]))
                self._counts[camera] += 1
            elif pixels.get(ref.image_key.split('.', 1)[1]) is not None:
                raise ValueError('missing camera reference has a frame')
            self._index.write(json.dumps({
                'episode_id': packet.episode_id,
                'observation_sequence': packet.sequence,
                'camera': camera,
                'source_key': ref.image_key,
                'frame_index': index,
                'video_path': str(path.resolve()) if index is not None else None,
                'capture_observation_sequence': ref.observation_sequence,
                'captured_at': (ref.captured_at.isoformat()
                                if ref.captured_at is not None else None),
                'captured_monotonic': ref.captured_monotonic,
                'availability': ref.availability,
                'synchronization': ref.synchronization,
                'time_basis': ref.time_basis,
            }) + '\n')
        self._index.flush()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._resources.close()
            self._finalized = True

    @property
    def artifacts(self) -> dict[str, str]:
        if not self._finalized:
            raise RuntimeError('camera evidence artifacts are not finalized')
        result = {'frames_path': str(self._index_path.resolve())}
        for camera, key in (('main', 'video_path'),
                            ('wrist', 'wrist_video_path')):
            if self._counts[camera]:
                result[key] = str(self._paths[camera].resolve())
        return result
