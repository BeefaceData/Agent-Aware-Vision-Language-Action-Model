# Agent-Augmented Robotic Manipulation

The domain is improving a pretrained robot policy's task execution through supervision, bounded interventions, and experience retained across attempts.

## Language

**Frozen VLA policy**:
A vision-language-action policy whose learned parameters remain unchanged while the supervisory system is developed and evaluated.
_Avoid_: Self-training VLA, online policy training

**Supervisor**:
The vision-language-model-based decision maker that assesses task execution and requests a correction when warranted.
_Avoid_: Overseer, robot policy

**Harness**:
The execution environment surrounding the policy and supervisor, coordinating observations, proposed actions, permitted interventions, and recorded outcomes.
_Avoid_: Supervisor model

**Supervised system**:
The combined robot-execution system consisting of the frozen VLA, harness, supervisor, and configured memory. Its task performance can change while the constituent models' learned parameters remain fixed.
_Avoid_: Retrained VLA, improved model weights

**Intervention**:
A bounded change to the policy's proposed execution requested by the supervisor.
_Avoid_: Policy retraining

**Episode**:
One attempt to complete a manipulation task, from its initial configuration to success or another termination condition.
_Avoid_: Control step, action chunk

**Action horizon**:
The maximum number of executed robot actions permitted in one episode, including corrective actions.
_Avoid_: Action chunk, elapsed-time limit

**Rollout trace**:
The recorded evidence of observations, decisions, actions, and outcomes from a task attempt.
_Avoid_: Supplied training demonstration

**Cross-episode memory**:
Recorded experience, including intervention outcomes, that remains available to inform the supervisor after an episode ends.
_Avoid_: Updated policy weights, current-episode context

**Memory-based self-improvement**:
The intended improvement of supervisor decisions through accumulated, outcome-linked experience while both the VLA and supervisor model weights remain unchanged. Evidence of improvement requires comparison with the same supervisor without that experience.
_Avoid_: Model training, automatic harness modification

**Fixed-memory evaluation**:
An evaluation in which the supervisor's persistent experience is held at a declared development-memory snapshot throughout the measured episodes.
_Avoid_: Online adaptation experiment

**Adaptation evaluation**:
An evaluation that permits memory updates from a declared starting snapshot and measures performance against the amount of experience consumed.
_Avoid_: Fixed-memory evaluation, model fine-tuning

**Failure signal**:
Observable evidence that a task attempt may be stalled or deviating from its intended progress; it can be ambiguous and does not itself establish failure.
_Avoid_: Ground-truth task outcome

**Failure diagnosis**:
The supervisor's evidence-backed assessment of what is going wrong during an episode, including an explicit uncertain assessment when the available observations are insufficient.
_Avoid_: Evaluator verdict

**Recovery tool**:
A predefined, bounded corrective procedure the supervisor may request to recover from an execution problem.
_Avoid_: Generated robot program, instruction rewrite

**Action adjustment**:
A bounded numerical change to a proposed robot action requested by the supervisor.
_Avoid_: Policy update, model training

**Transfer probe**:
An evaluation of the developed supervised system on the client task before task-specific tuning, with declared frozen elements and necessary integration changes.
_Avoid_: Guaranteed zero-shot improvement
