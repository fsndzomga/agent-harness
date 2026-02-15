"""Web tools for the Compact Search Agent.

Provides web_search() and visit_webpage() — thin wrappers around
duckduckgo_search and requests+BeautifulSoup.
"""

from __future__ import annotations

import re
import requests


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


def visit_webpage(url: str, max_chars: int = 8000) -> str:
    """Fetch a webpage and return its content as plain text (truncated).

    Returns the readable text content of the page.
    """
    try:
        from bs4 import BeautifulSoup

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()

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
