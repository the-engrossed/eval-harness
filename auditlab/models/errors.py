"""Model-call failures, each tagged with the failure origin it scores as (SPEC.md §9.8)."""

from __future__ import annotations

from auditlab.domain.enums import FailureOrigin, Node


class ProviderError(RuntimeError):
    """The provider did not deliver: HTTP error, timeout, connection failure, malformed reply.

    Origin `infra`: never counts against the model's accuracy. The T12 runner must catch
    this per node and record a failed step with this failure_origin, not stop the whole run.
    """

    failure_origin = FailureOrigin.INFRA

    def __init__(self, provider: str, model_id: str, node: Node, message: str) -> None:
        super().__init__(f"{provider} {model_id} at {node.value}: {message}")
        self.provider = provider
        self.model_id = model_id
        self.node = node


class ModelOutputError(ValueError):
    """The provider returned, but the output is not valid JSON or fails the node's schema.

    Origin `agent`: the model's output was wrong, so it counts toward accuracy. `raw_text`
    is exactly what the model returned. The T12 runner must catch this per node and record
    a failed step with this failure_origin, not stop the whole run.
    """

    failure_origin = FailureOrigin.AGENT

    def __init__(self, model_id: str, node: Node, raw_text: str, reason: str) -> None:
        super().__init__(f"{model_id} at {node.value}: output rejected: {reason}")
        self.model_id = model_id
        self.node = node
        self.raw_text = raw_text


class PromptTooLargeError(RuntimeError):
    """The prompt exceeds the model's context window, so the call is never made.

    Origin `agent`: the prompt is built from the model's own upstream output (for example
    an over-broad sample selection), and a model that cannot take this case's input fails
    it. Checked before the call because some servers (Ollama) silently truncate instead.
    """

    failure_origin = FailureOrigin.AGENT

    def __init__(self, model_id: str, node: Node, tokens: int, window: int) -> None:
        super().__init__(
            f"{model_id} at {node.value}: prompt too large: {tokens} tokens (estimated) "
            f"> context window {window}"
        )
        self.model_id = model_id
        self.node = node
        self.tokens = tokens
        self.window = window


class RecordedFailure(RuntimeError):
    """Replay of a call that failed while it was recorded (a cache failure entry).

    Carries the recorded origin and the exact recorded message, so a replayed run scores
    the failure exactly as the recording did. Only ReplayModel raises it.
    """

    def __init__(self, origin: FailureOrigin, message: str) -> None:
        super().__init__(message)
        self.failure_origin = origin
