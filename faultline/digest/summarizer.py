"""
LLM summarizer for daily digest.

Zero imports from faultline core — fully self-contained.
Takes git commit data and produces a human-readable daily summary.
"""

import json
import logging
import os

logger = logging.getLogger(__name__)

_MODEL = "claude-haiku-4-5-20251001"

_SYSTEM_PROMPT = """\
You are a senior engineering manager's AI assistant. \
You produce concise daily digest reports from git commit data.

## Output format

Return a JSON object with these fields:

- "summary": 2-3 sentence executive summary of what happened today.
- "highlights": array of 3-5 bullet points (strings), each describing a notable change. \
  Focus on WHAT was done and WHY it matters, not technical details.
- "risk_signals": array of 0-3 objects, each with "signal" (string) and "severity" \
  ("low" | "medium" | "high"). Flag things like: high bug-fix ratio, single author \
  touching many areas, large changes without tests.
- "categories": array of objects with "name" (area/feature name), \
  "commit_count" (int), "summary" (1 sentence), "is_bug_fix" (bool — true if >50% \
  of commits in this category are bug fixes).

## Rules

1. Group commits by business purpose, not by file path.
2. Use clear, non-technical language an EM can skim in 30 seconds.
3. If there are no commits, return a summary saying "No commits merged today."
4. Highlight patterns: many bug fixes, single-author areas, large refactors.
5. Keep each bullet point under 100 characters.\
"""

_USER_PROMPT = """\
Generate a daily digest for {repo_name} on {date}.

Branch: {branch}
Total commits: {total_commits}
Bug fixes: {bug_fixes}

Contributors:
{authors}

Directory activity:
{dir_changes}

Commits:
{commits}\
"""


def summarize_with_llm(
    digest_data: dict,
    api_key: str | None = None,
) -> dict | None:
    """Summarizes daily commit data using Claude Haiku.

    Args:
        digest_data: Output from git_reader.get_daily_commits().
        api_key: Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.

    Returns:
        Dict with summary, highlights, risk_signals, categories.
        None if LLM call fails or no API key.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None

    if digest_data["total_commits"] == 0:
        return {
            "summary": f"No commits merged to {digest_data['branch']} on {digest_data['date']}.",
            "highlights": [],
            "risk_signals": [],
            "categories": [],
        }

    try:
        import anthropic
    except ImportError:
        logger.warning("anthropic package not installed")
        return None

    prompt = _format_digest_prompt(digest_data)

    try:
        client = anthropic.Anthropic(api_key=key)
        response = client.messages.create(
            model=_MODEL,
            max_tokens=2048,
            temperature=0,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text.strip()
        # Strip markdown code block if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        return json.loads(text)
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("Digest LLM summarization failed: %s", e)
        return None


def summarize_with_deepseek(
    digest_data: dict,
    api_key: str | None = None,
    model: str = "deepseek-chat",
    base_url: str = "https://api.deepseek.com",
) -> dict | None:
    """Summarizes daily commit data using DeepSeek API (OpenAI-compatible).

    Zero imports from faultline core — fully self-contained.
    """
    key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        return None

    if digest_data["total_commits"] == 0:
        return {
            "summary": f"No commits merged to {digest_data['branch']} on {digest_data['date']}.",
            "highlights": [],
            "risk_signals": [],
            "categories": [],
        }

    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai package not installed (required for DeepSeek)")
        return None

    prompt = _format_digest_prompt(digest_data)

    try:
        client = OpenAI(api_key=key, base_url=base_url)
        response = client.chat.completions.create(
            model=model,
            max_tokens=2048,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT + "\n\nRespond with valid JSON only."},
                {"role": "user", "content": prompt},
            ],
        )

        text = response.choices[0].message.content
        if not text:
            return None
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        return json.loads(text)
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("DeepSeek digest summarization failed: %s", e)
        return None


def _format_digest_prompt(digest_data: dict) -> str:
    """Formats commit data into a prompt string. Shared by all providers."""
    commit_lines = []
    for c in digest_data["commits"][:50]:
        bug_tag = " [BUG FIX]" if c["is_bug_fix"] else ""
        pr_tag = f" (PR #{c['pr_number']})" if c.get("pr_number") else ""
        files_hint = f" [{len(c['files_changed'])} files]" if c["files_changed"] else ""
        commit_lines.append(f"  {c['sha']} {c['author']}: {c['message']}{bug_tag}{pr_tag}{files_hint}")

    author_lines = [f"  {name}: {count} commits" for name, count in digest_data["authors"].items()]
    dir_lines = [f"  {d}: {count} file changes" for d, count in list(digest_data["dir_changes"].items())[:20]]

    return _USER_PROMPT.format(
        repo_name=digest_data["repo_name"],
        date=digest_data["date"],
        branch=digest_data["branch"],
        total_commits=digest_data["total_commits"],
        bug_fixes=digest_data["bug_fixes"],
        authors="\n".join(author_lines) if author_lines else "  (none)",
        dir_changes="\n".join(dir_lines) if dir_lines else "  (none)",
        commits="\n".join(commit_lines) if commit_lines else "  (none)",
    )


def build_digest_report(
    digest_data: dict,
    llm_summary: dict | None = None,
) -> dict:
    """Combines git data with LLM summary into a full digest report.

    Returns a complete JSON-serializable digest ready for the dashboard.
    """
    bug_ratio = (
        digest_data["bug_fixes"] / digest_data["total_commits"]
        if digest_data["total_commits"] > 0
        else 0
    )

    # Auto-generate risk signals if LLM is unavailable
    risk_signals = []
    if llm_summary and llm_summary.get("risk_signals"):
        risk_signals = llm_summary["risk_signals"]
    else:
        if bug_ratio > 0.5 and digest_data["total_commits"] >= 3:
            risk_signals.append({
                "signal": f"High bug-fix ratio: {bug_ratio:.0%} of commits are bug fixes",
                "severity": "high",
            })
        authors = digest_data.get("authors", {})
        if len(authors) == 1 and digest_data["total_commits"] >= 5:
            name = list(authors.keys())[0]
            risk_signals.append({
                "signal": f"Single contributor ({name}) made all {digest_data['total_commits']} commits",
                "severity": "medium",
            })

    return {
        "repo_name": digest_data["repo_name"],
        "repo_path": digest_data["repo_path"],
        "remote_url": digest_data["remote_url"],
        "branch": digest_data["branch"],
        "date": digest_data["date"],
        "total_commits": digest_data["total_commits"],
        "bug_fixes": digest_data["bug_fixes"],
        "bug_fix_ratio": round(bug_ratio, 3),
        "authors": digest_data["authors"],
        "dir_changes": digest_data["dir_changes"],
        "summary": llm_summary.get("summary", "") if llm_summary else "",
        "highlights": llm_summary.get("highlights", []) if llm_summary else [],
        "risk_signals": risk_signals,
        "categories": llm_summary.get("categories", []) if llm_summary else [],
        "commits": digest_data["commits"],
    }
