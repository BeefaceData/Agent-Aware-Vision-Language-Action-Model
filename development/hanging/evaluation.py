"""Score labeled hanging evidence without a simulator, network, or robot actions.

Inputs are assessments, not inferred contact from TCP alignment. Public and
development-oracle trials are deliberately separate. Native benchmark success
is recorded but never substitutes for evidence of sustained support.
"""

from typing import Annotated, Literal

from pydantic import Field, StrictBool, StrictFloat, model_validator

from harness.schemas import Model, NonnegativeInt


Seconds = Annotated[StrictFloat, Field(ge=0)]
PositiveSeconds = Annotated[StrictFloat, Field(gt=0)]
EvidenceMode = Literal["public", "development_oracle"]


class Assessment(Model):
    value: StrictBool | None
    sequence: NonnegativeInt
    source: EvidenceMode
    artifact: Annotated[str, Field(min_length=1)]
    reason: Annotated[str, Field(min_length=1)]


class SupportSample(Model):
    sequence: NonnegativeInt
    elapsed_seconds: Seconds
    supported: StrictBool | None
    source: EvidenceMode
    artifact: Annotated[str, Field(min_length=1)]


class HangCriteria(Model):
    # Initial experimental criteria, not established physical tolerances.
    observation_window_seconds: PositiveSeconds = 1.0
    minimum_samples: Annotated[int, Field(strict=True, ge=2)] = 3
    maximum_sample_gap_seconds: PositiveSeconds = 0.5


class HangTrial(Model):
    schema_version: Literal[1] = 1
    run_id: Annotated[str, Field(min_length=1)]
    seed: NonnegativeInt
    evidence_mode: EvidenceMode
    entry: Assessment
    approach: Assessment | None = None
    seat_alignment: Assessment | None = None
    contact: Assessment | None = None
    release: Assessment | None = None
    retreat: Assessment | None = None
    support_samples: tuple[SupportSample, ...] = ()
    native_success: StrictBool | None = None
    criteria: HangCriteria = Field(default_factory=HangCriteria)

    @model_validator(mode="after")
    def consistent_evidence(self):
        assessments = [self.entry, self.approach, self.seat_alignment,
                       self.contact, self.release, self.retreat]
        if any(item.source != self.evidence_mode for item in
               [*(a for a in assessments if a is not None), *self.support_samples]):
            raise ValueError("mixed_public_and_oracle_evidence")
        previous = self.entry.sequence
        for item in (self.approach, self.seat_alignment, self.contact,
                     self.release, self.retreat):
            if item is not None:
                if item.sequence < previous:
                    raise ValueError("stage_evidence_out_of_order")
                previous = item.sequence
        previous_sample = None
        for sample in self.support_samples:
            if sample.sequence < previous:
                raise ValueError("support_sample_precedes_stage_evidence")
            if previous_sample is not None and (
                    sample.sequence <= previous_sample.sequence
                    or sample.elapsed_seconds <= previous_sample.elapsed_seconds):
                raise ValueError("support_samples_not_strictly_ordered")
            previous_sample = sample
        return self


class StageResult(Model):
    status: Literal["passed", "failed", "unknown"]
    reason: str
    sequence: NonnegativeInt | None = None


class HangReport(Model):
    schema_version: Literal[1] = 1
    run_id: str
    seed: NonnegativeInt
    evidence_mode: EvidenceMode
    criteria: HangCriteria
    valid_entry: StageResult
    stages: dict[str, StageResult]
    outcome: Literal["invalid_entry", "entry_unknown", "hang_verified", "hang_failed", "hang_unknown"]
    native_success: StrictBool | None


def stage_result(assessment):
    if assessment is None:
        return StageResult(status="unknown", reason="evidence_missing")
    status = "unknown" if assessment.value is None else (
        "passed" if assessment.value else "failed")
    return StageResult(status=status, reason=assessment.reason, sequence=assessment.sequence)


def sustained_hang(trial):
    for name in ("release", "retreat"):
        assessment = getattr(trial, name)
        if assessment is None or assessment.value is None:
            return StageResult(status="unknown", reason=f"{name}_unverified")
        if not assessment.value:
            return StageResult(status="failed", reason=f"{name}_failed",
                               sequence=assessment.sequence)

    samples = trial.support_samples
    for sample in samples:
        if sample.supported is False:
            return StageResult(status="failed", reason="support_lost",
                               sequence=sample.sequence)
    if any(sample.supported is None for sample in samples):
        return StageResult(status="unknown", reason="support_ambiguous")
    criteria = trial.criteria
    if len(samples) < criteria.minimum_samples:
        return StageResult(status="unknown", reason="insufficient_support_samples")
    if samples[-1].elapsed_seconds - samples[0].elapsed_seconds < criteria.observation_window_seconds:
        return StageResult(status="unknown", reason="support_window_too_short")
    if any(right.elapsed_seconds - left.elapsed_seconds > criteria.maximum_sample_gap_seconds
           for left, right in zip(samples, samples[1:])):
        return StageResult(status="unknown", reason="support_observation_gap")
    return StageResult(status="passed", reason="support_observed_after_release_and_retreat",
                       sequence=samples[-1].sequence)


def evaluate(trial: HangTrial) -> HangReport:
    entry = stage_result(trial.entry)
    stages = {name: stage_result(getattr(trial, name)) for name in
              ("approach", "seat_alignment", "contact", "release", "retreat")}
    if entry.status != "passed":
        stages["sustained_hang"] = StageResult(status="unknown", reason="valid_entry_required")
        outcome = "invalid_entry" if entry.status == "failed" else "entry_unknown"
    else:
        hang = sustained_hang(trial)
        stages["sustained_hang"] = hang
        outcome = {"passed": "hang_verified", "failed": "hang_failed",
                   "unknown": "hang_unknown"}[hang.status]
    return HangReport(run_id=trial.run_id, seed=trial.seed,
        evidence_mode=trial.evidence_mode, criteria=trial.criteria, valid_entry=entry,
        stages=stages, outcome=outcome, native_success=trial.native_success)


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path, help="JSON file matching HangTrial")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trial = HangTrial.model_validate_json(args.evidence.read_text(encoding="utf-8-sig"))
    report = evaluate(trial)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(report.model_dump_json(indent=2) + "\n")
    print(report.outcome)


if __name__ == "__main__":
    main()
