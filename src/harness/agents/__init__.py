"""Agent base classes for the v2 architecture.

The ``AgentBase`` class defines the unified agent protocol — one method
per execution archetype.  Agent implementations subclass ``AgentBase``
and implement the methods for the modes they support.
"""

from .base import AgentBase

__all__ = ["AgentBase"]
