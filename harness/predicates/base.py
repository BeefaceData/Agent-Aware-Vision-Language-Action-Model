from typing import Literal
from pydantic import StrictBool, StrictStr
from harness.schemas import Model, NonnegativeInt


class PredicateResult(Model):
    name: StrictStr
    value: StrictBool | None
    source: Literal["runtime", "perception", "development_oracle", "unavailable"]
    observation_sequence: NonnegativeInt
    reason: StrictStr
