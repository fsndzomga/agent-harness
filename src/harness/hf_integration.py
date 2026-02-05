"""HuggingFace Hub integration for storing run results.

Upload run.json files to a HuggingFace dataset repo for persistent
storage, sharing, and analysis.

Dataset structure on HF:
    {repo_id}/
        README.md              # Dataset card (auto-generated)
        {run_id}.json          # Full run.json files renamed with run_id
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None  # Will raise at usage time


def _get_token() -> str:
    """Get HF token from environment."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise ValueError(
            "HuggingFace token not found. Set HF_TOKEN in your environment or .env file."
        )
    return token


def _get_api(token: str | None = None):
    """Get HfApi instance."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for HF integration. "
            "Install with: pip install huggingface-hub"
        )
    return HfApi(token=token or _get_token())


def ensure_dataset_repo(
    repo_id: str,
    token: str | None = None,
    private: bool = False,
) -> str:
    """
    Create or ensure a HuggingFace dataset repo exists.

    Args:
        repo_id: HF repo ID (e.g., "username/agent-harness-runs")
        token: HF API token (defaults to HF_TOKEN env var)
        private: Whether the dataset should be private

    Returns:
        The repo URL as a string.
    """
    api = _get_api(token)
    url = api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        exist_ok=True,
        private=private,
    )
    _ensure_readme(api, repo_id)
    return str(url)


def _ensure_readme(api, repo_id: str) -> None:
    """Upload a dataset card README if one doesn't exist."""
    try:
        if api.file_exists(repo_id=repo_id, filename="README.md", repo_type="dataset"):
            return
    except Exception:
        pass  # file_exists may not be available in older versions

    readme = f"""---
license: mit
task_categories:
  - text-generation
tags:
  - agent-evaluation
  - benchmark
  - agent-harness
pretty_name: Agent Harness Runs
---

# Agent Harness Runs

This dataset contains evaluation run results from [agent-harness](https://github.com/fsndzomga/agent-harness).

Each file is a full `run.json` named `{{run_id}}.json`.

## Loading

```python
import json
from huggingface_hub import hf_hub_download

# Download a specific run
path = hf_hub_download("{repo_id}", "my_run_id.json", repo_type="dataset")
run = json.load(open(path))
```

```python
# List all run files
from huggingface_hub import HfApi
api = HfApi()
files = api.list_repo_files("{repo_id}", repo_type="dataset")
run_files = [f for f in files if f.endswith(".json")]
```
"""
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(readme)
        f.flush()
        try:
            api.upload_file(
                path_or_fileobj=f.name,
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message="Initialize dataset card",
            )
        finally:
            os.unlink(f.name)


def push_run(
    run_dir: Path,
    repo_id: str,
    token: str | None = None,
    private: bool = False,
) -> dict[str, str]:
    """
    Push a run.json to a HuggingFace dataset repo.

    Uploads the full run.json as {run_id}.json at the repo root.

    Args:
        run_dir: Path to the run directory containing run.json
        repo_id: HF dataset repo ID (e.g., "username/agent-harness-runs")
        token: HF API token (defaults to HF_TOKEN env var)
        private: Whether to create the repo as private (if new)

    Returns:
        Dict with "run_id", "repo_url", "file_url" keys.
    """
    run_json_path = Path(run_dir) / "run.json"
    if not run_json_path.exists():
        raise FileNotFoundError(f"No run.json found at {run_json_path}")

    run_data = json.loads(run_json_path.read_text())
    run_id = run_data.get("run_id", run_json_path.parent.name)

    api = _get_api(token)

    # Ensure repo exists
    repo_url = api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        exist_ok=True,
        private=private,
    )
    _ensure_readme(api, repo_id)

    # Upload full run.json as {run_id}.json
    api.upload_file(
        path_or_fileobj=str(run_json_path),
        path_in_repo=f"{run_id}.json",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Add run {run_id}",
    )

    file_url = f"https://huggingface.co/datasets/{repo_id}/blob/main/{run_id}.json"

    return {
        "run_id": run_id,
        "repo_url": str(repo_url),
        "file_url": file_url,
    }


def list_runs(
    repo_id: str,
    token: str | None = None,
) -> list[dict]:
    """
    List all runs from a HuggingFace dataset repo.

    Lists .json files in the repo and downloads each one.
    """
    api = _get_api(token)

    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    except Exception:
        return []

    # Filter to .json files (exclude README.md etc.)
    run_files = [f for f in files if f.endswith(".json")]

    if not run_files:
        return []

    if hf_hub_download is None:
        raise ImportError("huggingface_hub required. Install with: pip install huggingface-hub")

    runs = []
    tk = token or _get_token()
    for filename in run_files:
        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                token=tk,
            )
            run_data = json.loads(Path(local_path).read_text())
            runs.append(run_data)
        except Exception:
            continue

    return runs


def pull_run(
    run_id: str,
    repo_id: str,
    output_dir: Path | None = None,
    token: str | None = None,
) -> Path:
    """
    Download a run.json from HuggingFace.

    Args:
        run_id: The run ID to download
        repo_id: HF dataset repo ID
        output_dir: Where to save (defaults to current dir)
        token: HF API token

    Returns:
        Path to the downloaded file.
    """
    if hf_hub_download is None:
        raise ImportError("huggingface_hub required. Install with: pip install huggingface-hub")

    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"{run_id}.json",
        repo_type="dataset",
        token=token or _get_token(),
    )

    if output_dir:
        import shutil

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        dest = output_dir / f"{run_id}.json"
        shutil.copy2(local_path, dest)
        return dest

    return Path(local_path)
