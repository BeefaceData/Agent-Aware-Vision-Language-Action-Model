"""Shared plumbing for structured, single-turn Opus perception requests.

Each grounder sends public RGB images plus a system prompt, forces exactly one
tool call, and validates the returned arguments itself. Responses are 2D image
proposals only; metric geometry always comes from local RGB-D.
"""

import base64
from io import BytesIO

import numpy as np


def require_messages_client(client, model):
    """Fail at construction time rather than on the first perception request."""
    if not callable(getattr(getattr(client, "messages", None), "create", None)):
        raise ValueError("anthropic_messages_client_required")
    if not isinstance(model, str) or not model:
        raise ValueError("anthropic_model_required")


def rgb_view(observation, camera, *, missing_error, format_error):
    """Return one camera's image, requiring nonempty HxWx3 uint8 RGB."""
    image = observation.images.get(camera)
    if image is None:
        raise ValueError(missing_error)
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or not image.size:
        raise ValueError(format_error)
    return image


def png_bytes(image) -> bytes:
    from PIL import Image

    stream = BytesIO()
    Image.fromarray(image).save(stream, format="PNG", optimize=True)
    return stream.getvalue()


def image_block(png: bytes) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                        "data": base64.b64encode(png).decode("ascii")}}


def request_tool_input(client, *, model, system, tool, content, max_tokens):
    """Force one call of ``tool`` and return its input dict, or None if absent."""
    response = client.messages.create(
        model=model, max_tokens=max_tokens, system=system, tools=[tool],
        tool_choice={"type": "tool", "name": tool["name"]},
        messages=[{"role": "user", "content": content}])
    return next((item.input for item in response.content
                 if getattr(item, "type", None) == "tool_use"
                 and getattr(item, "name", None) == tool["name"]), None)


def unit_interval(value, error) -> float:
    """Parse a finite number in [0, 1]; booleans are rejected, not coerced."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(error)
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(error) from None
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(error)
    return value


def bounded_text(value, limit, error) -> str:
    """Require a string and clip it so logs stay bounded."""
    if not isinstance(value, str):
        raise ValueError(error)
    return value[:limit]
