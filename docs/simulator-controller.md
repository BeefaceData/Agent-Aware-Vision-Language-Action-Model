# Simulator controller evidence

Status: verified on 2026-10-05T10:09:21.168998+00:00. [Controller report](evidence/controller-wsl-20261005T100916Z.json) contains six passing translation probes.
[Installed package versions](evidence/controller-environment-20261005T100916Z.txt) describe the controller-only environment;
recreate that subset with `uv pip install --no-deps -r <package-file>`.
The inference/training dependency sets of LeRobot and LIBERO are intentionally not installed.

The local setup is complete. The failed installation omitted `termcolor` and
`future`; installing `termcolor==3.3.0` and `future==1.0.0` resolved the probe's
import failures. Both are included in the recorded package versions above.
Inspect the setup status from PowerShell:

```powershell
Get-Content .agent/controller-setup-status.json
Get-Content .agent/controller-setup.log -Tail 20
```

`ready` means the runtime probe passed and its report was saved. `failed` keeps
the error in the status and log; it does not mark #67 complete or enable a
conversion. The setup does not launch Ralph or change GitHub issues.

Issue #67 needs the installed controller's command scale and reference frame.
`inspect_simulator_controller.py` captures those facts from a live simulator
controller without loading a policy, downloading policy weights, or executing
an episode.

The local verification environment is WSL Ubuntu at
`/home/almon/.venvs/neotix-simulator`. It is a new environment; it does not
reconstruct the unrecorded environment behind the README's historical results.

From this repository in PowerShell:

```powershell
wsl -d Ubuntu -- /home/almon/.venvs/neotix-simulator/bin/python inspect_simulator_controller.py --output docs/evidence/controller-new.json
```

Choose a new output filename for each capture. The probe refuses to overwrite
evidence. LIBERO's dataset-path configuration is kept inside this environment,
unless `LIBERO_CONFIG_PATH` is explicitly supplied.

The probe calls LIBERO's `ControlEnv`, the base of the `OffScreenRenderEnv`
used by LeRobot 0.4.3, for LIBERO-10 task 0. It keeps the same robot/controller
defaults and disables cameras and rendering. It verifies relative control,
loads no initial-state dataset and executes no settling steps.
This environment supports controller inspection;
LeRobot is installed for source provenance, not as a complete inference stack.
It verifies the controller's end-effector position against the MuJoCo world
site position, then checks positive and negative translation commands on all
three axes. It records both the resulting world-space goal residuals and the
inverse conversion to native commands, together with versions and installed
source hashes.

This measures **command-to-goal mapping**, not achieved physical displacement.
The verified translation scale is 0.05 metres per native command unit on each
world axis: a +0.2 command requests a +0.01 metre goal offset, and converting
a declared world-frame metre residual to native units divides it by 0.05.
It does not establish camera calibration, arbitrary frame transforms, real
robot compatibility, correction safety or task performance. A runtime with
different controller settings must be verified again before enabling a
conversion. Conversion implementation and its public tests remain part of #67.
