"""
CLI entry point for daily digest.

Standalone — can be called directly or imported by the Vite API endpoint.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from faultline.digest.git_reader import get_daily_commits
from faultline.digest.summarizer import (
    summarize_with_llm,
    summarize_with_deepseek,
    build_digest_report,
)


def generate_digest(
    repo_path: str = ".",
    date: str | None = None,
    branch: str | None = None,
    api_key: str | None = None,
    use_llm: bool = True,
    provider: str = "anthropic",
) -> dict:
    """Generate a daily digest report.

    Args:
        repo_path: Path to the git repository.
        date: Date in YYYY-MM-DD format. Defaults to today.
        branch: Branch to analyze. Auto-detects main/master.
        api_key: API key for LLM provider.
        use_llm: Whether to use LLM for summarization.
        provider: LLM provider: "anthropic" or "deepseek".

    Returns:
        Complete digest report as a dict.
    """
    if not date:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    digest_data = get_daily_commits(repo_path, date, branch)

    llm_summary = None
    if use_llm and digest_data["total_commits"] > 0:
        if provider == "deepseek":
            llm_summary = summarize_with_deepseek(digest_data, api_key=api_key)
        else:
            llm_summary = summarize_with_llm(digest_data, api_key=api_key)

    return build_digest_report(digest_data, llm_summary)


def main():
    """CLI entry point: python -m faultline.digest <repo_path> [--date YYYY-MM-DD]"""
    import argparse

    parser = argparse.ArgumentParser(description="Generate daily commit digest")
    parser.add_argument("repo_path", nargs="?", default=".", help="Path to git repo")
    parser.add_argument("--date", "-d", default=None, help="Date (YYYY-MM-DD), default: today")
    parser.add_argument("--branch", "-b", default=None, help="Branch name, default: auto-detect")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM summarization")
    parser.add_argument("--provider", default="anthropic", help="LLM provider: anthropic or deepseek")
    parser.add_argument("--api-key", default=None, help="API key for LLM provider")
    parser.add_argument("--output", "-o", default=None, help="Output file path (default: stdout)")
    args = parser.parse_args()

    report = generate_digest(
        repo_path=args.repo_path,
        date=args.date,
        branch=args.branch,
        api_key=args.api_key,
        use_llm=not args.no_llm,
        provider=args.provider,
    )

    output = json.dumps(report, indent=2, ensure_ascii=False)

    if args.output:
        Path(args.output).write_text(output)
        print(f"Saved to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
