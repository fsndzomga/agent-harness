"""Base provider types and protocol.

Defines the **completion logging contract** for the harness.

There are two layers:

1. ``Provider`` — a structural Protocol used for type-checking. Any object
   with a ``complete()`` method that returns ``CompletionResponse`` satisfies
   this.

2. ``BaseProvider`` — an abstract base class that **every concrete provider
   should extend**.  ``BaseProvider.complete()`` wraps the subclass's
   ``_do_complete()`` with automatic trace logging so usage/cost data is
   *always* captured.  New providers that forget to call the logger will
   still work because the base class handles it.

For agents that use **external frameworks** (smolagents, langchain, …) and
therefore bypass harness providers entirely, see:

* ``Agent.emit_completion()``  — if the agent extends ``Agent``
* ``harness.logging.emit_completion()`` — standalone helper for any context
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Any, Protocol, runtime_checkable, TYPE_CHECKING

if TYPE_CHECKING:
    from ..logging import TraceLogger, StdoutLogger


@dataclass
class Message:
    """A message in a conversation."""
    role: Literal["user", "assistant", "system"]
    content: str
    tool_calls: list[dict] | None = None
    tool_results: list[dict] | None = None


@dataclass
class Usage:
    """Token usage information."""
    input_tokens: int
    output_tokens: int
    total_tokens: int = field(init=False)
    
    def __post_init__(self):
        self.total_tokens = self.input_tokens + self.output_tokens


@dataclass
class CompletionRequest:
    """Request for a completion."""
    messages: list[Message]
    tools: list[dict] | None = None
    max_tokens: int = 4096
    temperature: float = 0.0
    response_format: dict | None = None  # For structured outputs


@dataclass
class CompletionResponse:
    """Response from a completion request."""
    message: Message
    usage: Usage
    raw_request: dict      # Exact request sent to API
    raw_response: dict     # Exact response from API
    provider: str          # e.g., "litellm/anthropic"
    latency_ms: int
    model: str = ""


@runtime_checkable
class Provider(Protocol):
    """Protocol for LLM providers (structural typing)."""
    
    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Make a completion request."""
        ...


class BaseProvider(ABC):
    """Abstract base class for LLM providers with automatic completion logging.

    Subclass this and implement ``_do_complete()``.  The base ``complete()``
    method wraps every call with automatic trace logging — **you cannot
    forget to log completions**.

    ``_do_complete()`` must return a ``CompletionResponse`` with
    ``raw_request`` and ``raw_response`` populated; these are what the
    harness uses to extract usage/cost from traces.

    Example::

        class MyProvider(BaseProvider):
            def _do_complete(self, request):
                resp = call_my_api(request)
                return CompletionResponse(
                    message=..., usage=...,
                    raw_request={...}, raw_response={...},
                    provider="my-provider", latency_ms=42,
                )
    """

    def __init__(
        self,
        model: str,
        logger: "TraceLogger | StdoutLogger | None" = None,
    ):
        self.model = model
        self.logger = logger

    # ------------------------------------------------------------------
    # Public API — do NOT override
    # ------------------------------------------------------------------

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Make a completion request **and** log it automatically.

        Do NOT override this method.  Override ``_do_complete()`` instead.
        """
        result = self._do_complete(request)

        # Auto-log — this is the whole point of the ABC
        if self.logger:
            self.logger.log_completion(
                provider=result.provider,
                request=result.raw_request,
                response=result.raw_response,
                latency_ms=result.latency_ms,
            )

        return result

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    @abstractmethod
    def _do_complete(self, request: CompletionRequest) -> CompletionResponse:
        """Implement the actual API call.

        **Must** return a ``CompletionResponse`` with ``raw_request`` and
        ``raw_response`` populated — the harness reads these for
        usage / cost tracking.
        """
        ...
