"""Web tools for the Compact Search Agent.

Provides web_search() and visit_webpage() — thin wrappers around
duckduckgo_search and requests+BeautifulSoup.
"""

from __future__ import annotations

import random
import re
import time

import requests

# Rotate User-Agent strings to reduce bot-detection blocks
_USER_AGENTS = [
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
]


def web_search(query: str, max_results: int = 8) -> str:
    """Search the web using DuckDuckGo.

    Returns a formatted string of results with title, URL, and snippet.
    Falls back to a reformulated query on failure.
    """
    from ddgs import DDGS

    for attempt_query in [query, _reformulate(query)]:
        try:
            ddgs = DDGS()
            results = list(ddgs.text(attempt_query, max_results=max_results))
            if results:
                return _format_results(results)
        except Exception:
            continue

    return "(No results found)"


def visit_webpage(url: str, max_chars: int = 10000) -> str:
    """Fetch a webpage and return its content as plain text (truncated).

    Retries on transient errors (429, 5xx) with exponential backoff.
    Returns the readable text content of the page.
    """
    try:
        from bs4 import BeautifulSoup

        headers = {
            "User-Agent": random.choice(_USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",
        }

        # Retry loop for transient failures
        resp = None
        last_exc = None
        for attempt in range(3):
            try:
                resp = requests.get(url, headers=headers, timeout=20)
                if resp.status_code == 429:
                    # Rate limited — wait and retry
                    wait = 2 ** attempt + random.random()
                    time.sleep(wait)
                    continue
                if resp.status_code >= 500:
                    time.sleep(1)
                    continue
                resp.raise_for_status()
                break
            except requests.exceptions.RequestException as exc:
                last_exc = exc
                if attempt < 2:
                    time.sleep(1)
                    continue

        if resp is None or resp.status_code != 200:
            err = last_exc or f"HTTP {resp.status_code if resp else 'no response'}"
            return f"(Error fetching page: {err})"

        # Parse and strip noisy elements
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup.find_all(
            ["script", "style", "nav", "footer", "noscript", "aside"]
        ):
            tag.decompose()
        # Remove Wikipedia language/interwiki links and sidebars
        for tag in soup.find_all(class_=re.compile(
            r"interlanguage|sidebar|infobox|reflist|navbox|catlinks|mw-editsection",
            re.I,
        )):
            tag.decompose()

        # Try to find main content area
        main = (
            soup.find("main")
            or soup.find("article")
            or soup.find(id="bodyContent")    # Wikipedia
            or soup.find(id="content")
            or soup.find(attrs={"role": "main"})
            or soup.body
            or soup
        )

        # Extract text with newline separators
        content = main.get_text(separator="\n", strip=True)

        # Collapse excessive whitespace
        content = re.sub(r"\n{3,}", "\n\n", content)

        if len(content) > max_chars:
            content = content[:max_chars] + "\n\n... (truncated)"

        return content if content else "(Page is empty)"
    except Exception as exc:
        return f"(Error fetching page: {exc})"


# ── helpers ──────────────────────────────────────────────────────────

def _reformulate(query: str) -> str:
    """Simple query reformulation: remove quotes, shorten."""
    q = query.replace('"', "").replace("'", "")
    words = q.split()
    if len(words) > 6:
        words = words[:6]
    return " ".join(words)


def _format_results(results: list[dict]) -> str:
    """Format DuckDuckGo results into a readable string."""
    lines: list[str] = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "(no title)")
        href = r.get("href", "")
        body = r.get("body", "")
        lines.append(f"{i}. **{title}**\n   URL: {href}\n   {body}")
    return "\n\n".join(lines)
