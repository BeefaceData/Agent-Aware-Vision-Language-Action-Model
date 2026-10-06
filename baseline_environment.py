"""Inspect the invocation's environment without exporting paths or credentials."""

import hashlib
from importlib import metadata
import json
import os
import platform


def capture_environment(path, device):
    """Write a new runtime manifest, including failures, before model loading.

    Package metadata and installed controller source bytes are inspected; no
    simulator is constructed. This is not a controller calibration or a lockfile.
    """
    if device not in {'cpu', 'cuda'}:
        raise ValueError('device must be cpu or cuda')
    # Exclusive creation protects evidence from earlier invocations.
    with path.open('x', encoding='utf-8') as stream:
        record = {
            'schema_version': 1,
            'provenance': 'inspected-invocation-runtime',
            'historical_environment': {
                'source': 'README.md: Tested environment',
                'provenance': 'documented-history-not-current-inspection',
                'python': '3.10', 'lerobot': '0.4.3', 'torch': '2.7.1',
                'gymnasium': '1.3.0', 'hf-libero': '0.1.4',
            },
            'python': {'version': platform.python_version(),
                       'implementation': platform.python_implementation()},
            'platform': platform.system(),
            'device': {'requested': device, 'resolved': None},
            'render_backend': os.environ.get('MUJOCO_GL', 'egl'),
            'dependencies': [],
            'controller': {'distribution': 'robosuite', 'version': None,
                           'source_sha256': {}, 'status': 'unavailable'},
            'preflight': {'status': 'failed'},
        }
        try:
            record['dependencies'] = sorted(
                ({'name': dist.metadata['Name'], 'version': dist.version}
                 for dist in metadata.distributions() if dist.metadata['Name']),
                key=lambda row: (row['name'].lower(), row['version']))
            try:
                installed = metadata.version('lerobot')
            except metadata.PackageNotFoundError:
                installed = None
            if installed != '0.4.3':
                raise RuntimeError(
                    f'Unsupported LeRobot version {installed!r}; this runner requires '
                    'lerobot==0.4.3. Activate the declared baseline environment or '
                    'install "lerobot[smolvla,libero]==0.4.3" in that environment.')
            controller = metadata.distribution('robosuite')
            identity = record['controller']
            identity['version'] = controller.version
            for source in controller.files or ():
                name = str(source).replace('\\', '/')
                if name.startswith('robosuite/controllers/') and name.endswith('.py'):
                    identity['source_sha256'][name] = hashlib.sha256(
                        controller.locate_file(source).read_bytes()).hexdigest()
            if not identity['source_sha256']:
                raise RuntimeError('Cannot identify installed robosuite controller sources; '
                                   'install a distribution with readable controller files.')
            identity['status'] = 'installed-source-inspected'
            import torch

            record['device']['torch_cuda_build'] = torch.version.cuda
            if device == 'cuda':
                if not torch.cuda.is_available():
                    raise RuntimeError('CUDA unavailable in this interpreter; activate a '
                                       'CUDA-capable environment or select --device cpu.')
                index = torch.cuda.current_device()
                record['device'].update(resolved=f'cuda:{index}',
                                        name=torch.cuda.get_device_name(index))
            else:
                record['device']['resolved'] = 'cpu'
            record['preflight'] = {'status': 'passed'}
        except Exception as exc:
            # Unexpected exceptions may contain private local paths. Retain their
            # type only; our own actionable RuntimeErrors contain no such paths.
            record['preflight'].update(error_type=type(exc).__name__)
            if type(exc) is RuntimeError and str(exc).startswith((
                    'Unsupported LeRobot', 'Cannot identify installed', 'CUDA unavailable')):
                record['preflight']['diagnostic'] = str(exc)
            raise
        finally:
            json.dump(record, stream, indent=2, sort_keys=True)
            stream.write('\n')
    return record
