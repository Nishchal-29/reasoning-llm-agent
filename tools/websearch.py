from __future__ import annotations
import json
import logging
import textwrap
from typing import Any, Dict, List, Optional
from duckduckgo_search import DDGS

logger = logging.getLogger(__name__)

MAX_QUERY_LENGTH: int = 300
MAX_RESULTS: int = 5
MAX_OUTPUT_CHARS: int = 4_000
DEFAULT_REGION: str = "wt-wt"  

def _search_ddgs(query: str, max_results: int = MAX_RESULTS) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, region=DEFAULT_REGION, max_results=max_results):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("href", r.get("link", "")),
                "snippet": r.get("body", r.get("snippet", "")),
            })

    return results

def _format_results(results: List[Dict[str, str]]) -> str:
    if not results:
        return "[WebSearch] No results found."

    lines: List[str] = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "Untitled")
        url = r.get("url", "")
        snippet = r.get("snippet", "No description available.")
        lines.append(f"[{i}] {title}")
        if url:
            lines.append(f"    URL: {url}")
        lines.append(f"    {snippet}")
        lines.append("")  

    output = "\n".join(lines).strip()
    if len(output) > MAX_OUTPUT_CHARS:
        output = output[:MAX_OUTPUT_CHARS] + "\n… (results truncated)"
    return output

def search(query: str, max_results: int = MAX_RESULTS) -> str:
    query = query.strip()[:MAX_QUERY_LENGTH]
    if not query:
        return "[WebSearchError] Empty query provided."

    logger.info("Web search: '%s' (max_results=%d)", query, max_results)

    try:
        results = _search_ddgs(query, max_results=max_results)
        formatted = _format_results(results)
        logger.info("Web search returned %d results (%d chars)", len(results), len(formatted))
        return formatted
    except ImportError:
        logger.warning("duckduckgo-search not installed — attempting fallback")
    except Exception as exc:
        logger.warning("DuckDuckGo search failed: %s — attempting fallback", exc)

    error_msg = (
        "[WebSearchError] No search backend available. "
        "Install duckduckgo-search: pip install duckduckgo-search"
    )
    logger.error(error_msg)
    return error_msg

def run(input_str: str) -> str:
    input_str = input_str.strip()
    try:
        payload: Dict[str, Any] = json.loads(input_str)
        query = payload.get("query", payload.get("input", ""))
        max_results = int(payload.get("max_results", MAX_RESULTS))
        return search(query, max_results=max_results)
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    return search(input_str)