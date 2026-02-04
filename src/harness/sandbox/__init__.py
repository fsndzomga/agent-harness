"""
Sandbox module - isolation backends for agent execution.

Available tiers:
- NONE: Direct subprocess, no isolation
- VENV: Isolated Python virtual environment
- FIREJAIL: Linux filesystem/network restrictions (requires firejail)
- DOCKER: Full container isolation (requires Docker)

The harness auto-detects available isolation and uses the best available,
unless a benchmark requires a specific tier.
"""

from enum import Enum


class SandboxTier(Enum):
    """Isolation tiers, from least to most isolated."""
    NONE = "none"           # Direct subprocess
    VENV = "venv"           # Isolated venv
    FIREJAIL = "firejail"   # Linux sandboxing
    DOCKER = "docker"       # Full container


# TODO: Implement sandbox backends
# For now, all execution happens in NONE tier (direct subprocess)
