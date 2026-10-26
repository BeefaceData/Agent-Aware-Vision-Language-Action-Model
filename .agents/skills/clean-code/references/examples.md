# Before / after examples

Short illustrations of the guidelines in `SKILL.md`. Python is used here, but
the ideas apply to any language. These are examples, not templates. Always adapt
to the conventions of the code you are working in.

## Separate decisions from effects

Before: the decision is buried inside I/O, so testing it needs a live client.

```python
def maybe_release(client, arm):
    state = client.get_state(arm)
    if state["contact"] and not state["table_contact"]:
        return False
    client.open_gripper(arm)
    return True
```

After: the rule is a pure function that tests can call with plain values.

```python
def release_allowed(contact: bool, table_contact: bool) -> bool:
    """Open only an empty hand or one whose object is supported."""
    return not contact or table_contact


def maybe_release(client, arm):
    state = client.get_state(arm)
    if not release_allowed(state["contact"], state["table_contact"]):
        return False
    client.open_gripper(arm)
    return True
```

## Flatten control flow with guard clauses

Before:

```python
def parse_seed(value):
    if value is not None:
        if isinstance(value, int):
            if value >= 0:
                return value
            else:
                raise ValueError("seed_negative")
        else:
            raise ValueError("seed_not_int")
    else:
        raise ValueError("seed_missing")
```

After:

```python
def parse_seed(value):
    if value is None:
        raise ValueError("seed_missing")
    if not isinstance(value, int):
        raise ValueError("seed_not_int")
    if value < 0:
        raise ValueError("seed_negative")
    return value
```

## Name the numbers

Before:

```python
if abs(mug_z - start_z) > 0.03:
    lifted = True
```

After:

```python
LIFT_THRESHOLD_M = 0.03  # Above sensor noise; below a typical lift height.

if abs(mug_z - start_z) > LIFT_THRESHOLD_M:
    lifted = True
```

## Inject dependencies

Before: the function reads the wall clock itself, so its timeout path is hard to
test.

```python
import time

def timed_out(started):
    return time.monotonic() - started > 30
```

After: the clock and limit are parameters with sensible defaults.

```python
import time

def timed_out(started, *, limit_seconds=30.0, clock=time.monotonic):
    return clock() - started > limit_seconds
```

Test:

```python
def test_timed_out_after_limit():
    assert timed_out(0.0, limit_seconds=30.0, clock=lambda: 31.0)
```

## A test that reads like a specification

```python
def test_release_refused_while_holding_unsupported_mug():
    assert release_allowed(contact=True, table_contact=False) is False
```

The name states the behavior, the body is one call and one check, and there is
no logic to debug in the test itself.
