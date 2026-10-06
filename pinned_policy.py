"""Resolve and verify the frozen baseline before constructing any model.

The checked-in lock is trusted configuration. Hub/cache contents are verified
against it; a caller-supplied repository or a mutable branch is never a pin.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


POLICY_REPOSITORY = 'HuggingFaceVLA/smolvla_libero'
POLICY_REVISION = '6721902bc4d61e50a3bfdb11dfb4cb626f05d102'
BACKBONE_REPOSITORY = 'HuggingFaceTB/SmolVLM2-500M-Instruct'
BACKBONE_REVISION = '7b375e1b73b11138ff12fe22c8f2822d8fe03467'
ASSET_LOCK = Path(__file__).with_name('policy-assets.lock.json')


def _verify(directory, spec):
    if directory.name != spec['revision'] or directory.parent.name != 'snapshots':
        raise ValueError('asset resolution did not return the pinned snapshot')
    expected = set(spec['files'])
    extras = {p.name for p in directory.iterdir() if p.is_file()} - expected
    if extras - {'README.md', '.gitattributes'}:
        raise ValueError('unexpected files in pinned snapshot')
    for name, entry in spec['files'].items():
        path = directory / name
        if not path.is_file() or path.stat().st_size != entry['size']:
            raise ValueError(f'missing or size-mismatched policy asset: {name}')
        if entry['algorithm'] == 'git-blob-sha1':
            digest = hashlib.sha1(f"blob {entry['size']}\0".encode())
        elif entry['algorithm'] == 'sha256':
            digest = hashlib.sha256()
        else:
            raise ValueError('unsupported asset digest algorithm')
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != entry['digest']:
            raise ValueError(f'digest-mismatched policy asset: {name}')


@dataclass(frozen=True)
class PinnedPolicyAssets:
    """Verified local sources; loading rechecks bytes before calling LeRobot."""

    policy_directory: Path
    backbone_directory: Path

    def identity(self):
        lock = json.loads(ASSET_LOCK.read_text(encoding='utf-8'))
        for name, repo, revision, directory in (
            ('policy', POLICY_REPOSITORY, POLICY_REVISION, self.policy_directory),
            ('backbone', BACKBONE_REPOSITORY, BACKBONE_REVISION, self.backbone_directory),
        ):
            spec = lock['assets'][name]
            if (spec['repo_id'], spec['revision']) != (repo, revision):
                raise ValueError('asset lock does not match the frozen baseline')
            _verify(directory, spec)
        return {'verification': 'file-digests-verified', **lock}

    def load(self, device, *, config_loader=None, policy_loader=None,
             processor_factory=None):
        """Load strict weights and matching processors from verified snapshots.

        Optional loaders are trusted dependency injection for offline contract
        tests; production uses the version-checked runner's LeRobot classes.
        """
        self.identity()
        if config_loader is None:
            from lerobot.configs.policies import PreTrainedConfig
            config_loader = PreTrainedConfig.from_pretrained
        if policy_loader is None:
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            policy_loader = SmolVLAPolicy.from_pretrained
        if processor_factory is None:
            from lerobot.policies.factory import make_pre_post_processors
            processor_factory = make_pre_post_processors
        policy_path, backbone_path = str(self.policy_directory), str(self.backbone_directory)
        config = config_loader(policy_path, local_files_only=True)
        if config.vlm_model_name != BACKBONE_REPOSITORY:
            raise ValueError('policy references an unexpected backbone')
        config.device = device
        config.vlm_model_name = backbone_path
        policy = policy_loader(policy_path, config=config,
                               local_files_only=True, strict=True)
        policy.to(device).eval()
        policy.requires_grad_(False)
        preprocessor, postprocessor = processor_factory(
            policy.config, policy_path,
            preprocessor_overrides={
                'device_processor': {'device': device},
                'tokenizer_processor': {'tokenizer_name': backbone_path},
            },
        )
        return policy, preprocessor, postprocessor


def resolve_policy_assets(repository=POLICY_REPOSITORY, *, snapshot_download=None):
    """Fail closed on unavailable, unpinned, partial or corrupted assets.

    Only the declared files are downloaded. Existing HF snapshots can be reused;
    their names alone are insufficient evidence of identity.
    """
    if repository != POLICY_REPOSITORY:
        raise ValueError('the baseline requires the declared frozen policy repository')
    if snapshot_download is None:
        from huggingface_hub import snapshot_download
    lock = json.loads(ASSET_LOCK.read_text(encoding='utf-8'))
    directories = []
    for name, repo, revision in (
        ('policy', POLICY_REPOSITORY, POLICY_REVISION),
        ('backbone', BACKBONE_REPOSITORY, BACKBONE_REVISION),
    ):
        spec = lock['assets'][name]
        if (spec['repo_id'], spec['revision']) != (repo, revision):
            raise ValueError('asset lock does not match the frozen baseline')
        directory = Path(snapshot_download(
            repo_id=repo, revision=revision, allow_patterns=list(spec['files']),
        )).absolute()
        _verify(directory, spec)
        directories.append(directory)
    assets = PinnedPolicyAssets(*directories)
    return assets
