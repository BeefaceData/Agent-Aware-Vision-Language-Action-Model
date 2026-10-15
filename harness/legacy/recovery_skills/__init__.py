"""Legacy recovery candidates retained for demonstrations and old supervision.

These are not the planner's six skills in harness.skills. File existence is not
verification; candidate-only admission restrictions remain unchanged.
"""


def candidate_skills():
    # Fixed allowlist: configuration cannot import arbitrary Python modules.
    from .retry_grasp import RetryGrasp
    from .backoff_reapproach_rack import BackoffReapproachRack

    return (RetryGrasp(), BackoffReapproachRack())
