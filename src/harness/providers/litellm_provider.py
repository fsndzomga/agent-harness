"""LiteLLM provider - default provider with broad model support."""

import time
from typing import TYPE_CHECKING

from .base import BaseProvider, Message, CompletionRequest, CompletionResponse, Usage

if TYPE_CHECKING:
    from ..logging import TraceLogger, StdoutLogger


class LiteLLMProvider(BaseProvider):
    """
    Provider using LiteLLM for broad model support.
    
    Extends ``BaseProvider`` so completion logging is automatic — the base
    ``complete()`` wraps ``_do_complete()`` and logs every call.

    LiteLLM auto-detects provider from model string:
    - "claude-sonnet-4-5-20250514" -> Anthropic
    - "gpt-4o" -> OpenAI
    - "gemini/gemini-pro" -> Google
    - etc.
    """
    
    def __init__(
        self,
        model: str,
        logger: "TraceLogger | StdoutLogger | None" = None,
    ):
        super().__init__(model, logger)
        self._litellm = None
    
    @property
    def litellm(self):
        """Lazy import of litellm."""
        if self._litellm is None:
            try:
                import litellm
                self._litellm = litellm
                # Disable litellm's own logging
                litellm.suppress_debug_info = True
            except ImportError:
                raise ImportError(
                    "litellm is required for LiteLLMProvider. "
                    "Install it with: pip install litellm"
                )
        return self._litellm
    
    def _do_complete(self, request: CompletionRequest) -> CompletionResponse:
        """Make a completion request via LiteLLM.

        Logging is handled by ``BaseProvider.complete()`` — do NOT log here.
        """
        
        # Build raw request
        raw_request = {
            "model": self.model,
            "messages": [
                {"role": m.role, "content": m.content}
                for m in request.messages
            ],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        
        if request.tools:
            raw_request["tools"] = request.tools
        
        if request.response_format:
            raw_request["response_format"] = request.response_format
        
        # Make request
        start = time.monotonic()
        response = self.litellm.completion(**raw_request)
        latency_ms = int((time.monotonic() - start) * 1000)
        
        # Get raw response
        raw_response = response.model_dump()
        
        # Try to get cost from litellm
        try:
            cost = self.litellm.completion_cost(completion_response=response)
            raw_response["_cost_usd"] = cost
        except Exception:
            pass  # Cost calculation may fail for some models
        
        # Normalize to our format
        choice = response.choices[0]
        message = Message(
            role="assistant",
            content=choice.message.content or "",
            tool_calls=getattr(choice.message, 'tool_calls', None),
        )
        
        usage = Usage(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )
        
        provider_name = f"litellm/{self._get_provider()}"
        
        return CompletionResponse(
            message=message,
            usage=usage,
            raw_request=raw_request,
            raw_response=raw_response,
            provider=provider_name,
            latency_ms=latency_ms,
            model=self.model,
        )
    
    def _get_provider(self) -> str:
        """Detect underlying provider from model string."""
        try:
            _, provider, *_ = self.litellm.get_llm_provider(self.model)
            return provider
        except Exception:
            return "unknown"
