# Baseline runtime environment

Every baseline invocation creates `environment.json` in its new output directory
before importing the simulator or resolving model assets. Use the existing runner:

```bash
python run_smolvla_episode.py --device cuda --output-dir outputs/baseline-new
```

The manifest contains inspected Python version/implementation, OS, every installed
Python distribution's name and version, selected MuJoCo render backend, requested
and resolved device, CUDA build and selected GPU name when applicable, and the
installed robosuite version plus SHA-256 hashes of its controller Python sources.
`MUJOCO_GL` defaults to `egl`; an explicit setting is preserved. This records the
selected backend, not proof that rendering succeeded. Controller source identity
does not establish the instantiated controller's settings, calibration or bounds.

The historical README environment is labeled separately and never substituted for
runtime inspection. Package versions do not form a complete reproducible system
lock: OS libraries, drivers, FFmpeg binaries and editable-install changes outside
the hashed controller sources remain outside this record. Package origins, local
installation paths, environment-variable dumps and authentication tokens are not
collected. Model file identity remains in `policy-assets.json`.

An unsupported or missing LeRobot installation fails before model loading, with
the required `lerobot==0.4.3` and environment repair guidance retained in the
manifest. Missing controller sources or an unavailable requested CUDA device also
fail preflight. Failed preflight is a pre-start failure, not a policy task outcome;
it can leave only `environment.json`, without an episode or `result.json`.
Unexpected inspection failures retain their exception type without private paths.
Existing manifests are never overwritten. A passed preflight establishes inspected
environment identity only; it does not prove all imports or live inference work.

Offline verification:

```bash
python -m unittest discover -s tests -p test_baseline_environment.py -v
python -m unittest discover -s tests
```

These fixtures exercise CPU/CUDA selection, historical/runtime separation,
controller digests, unsupported versions, missing sources, artifact preservation
and runner startup rejection. No live models, datasets or robot commands are used.
