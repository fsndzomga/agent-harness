"""Base provider types and protocol."""

from dataclasses import dataclass, field
from typing import Literal, Any, Protocol, runtime_checkable


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
    """Protocol for LLM providers."""
    
    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Make a completion request."""
        ...
