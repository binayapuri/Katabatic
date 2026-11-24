"""Check model README completeness across supported and experimental models.

Scans every model package under ``katabatic/models/<model>/`` and
``katabatic/experimental/models/<model>/`` and verifies that a ``README.md``
exists and contains the required documentation sections.

Usage:
    python benchmarks/check_readmes.py              # summary table
    python benchmarks/check_readmes.py --markdown   # Markdown report table
    python benchmarks/check_readmes.py --verbose    # detailed section breakdown
    python benchmarks/check_readmes.py --strict     # exit code 1 if any required section is missing
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Canonical required section categories and the heading keywords that satisfy each.
REQUIRED_SECTIONS: dict[str, tuple[str, ...]] = {
    "overview": (
        "overview",
        "model overview",
        "why this model",
        "how it works",
        "key idea",
        "project structure",
    ),
    "reference": (
        "reference",
        "references",
        "paper",
        "research paper",
        "citation",
        "reference and credit",
    ),
    "parameters": (
        "configuration",
        "hyperparameter",
        "hyperparameters",
        "parameters",
        "parameter guide",
        "constructor parameters",
        "supported variants",
    ),
    "usage": (
        "quick start",
        "quickstart",
        "usage",
        "input",
        "expected files",
        "training",
        "installation",
    ),
    "evaluation_or_notes": (
        "benchmark",
        "evaluation",
        "implementation notes",
        "implementation details",
        "limitations",
        "hardware",
        "runtime",
        "paper alignment",
        "output",
        "katabatic integration",
    ),
}


@dataclass
class ReadmeCheckResult:
    """Completeness check result for a single model README."""

    model_name: str
    tier: str  # "supported" | "experimental"
    readme_path: Path
    exists: bool
    headings: list[str] = field(default_factory=list)
    present_sections: list[str] = field(default_factory=list)
    missing_sections: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if not self.exists:
            return "MISSING_FILE"
        if self.missing_sections:
            return "INCOMPLETE"
        return "OK"


def extract_headings(markdown_text: str) -> list[str]:
    """Return normalized lowercase markdown heading titles (`#` through `####`)."""
    headings: list[str] = []
    in_code_block = False
    for raw_line in markdown_text.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        match = re.match(r"^#{1,4}\s+(.+?)\s*$", stripped)
        if match:
            title = re.sub(r"[`*_]", "", match.group(1)).strip().lower()
            headings.append(title)
    return headings


def check_readme_text(
    markdown_text: str,
    *,
    model_name: str = "model",
    tier: str = "supported",
    readme_path: Path | None = None,
    required_sections: dict[str, tuple[str, ...]] | None = None,
) -> ReadmeCheckResult:
    """Evaluate a markdown string against the required section categories."""
    sections = required_sections or REQUIRED_SECTIONS
    headings = extract_headings(markdown_text)

    present: list[str] = []
    missing: list[str] = []

    for section_key, keywords in sections.items():
        matched = any(
            any(kw in h for kw in keywords)
            for h in headings
        )
        if matched:
            present.append(section_key)
        else:
            missing.append(section_key)

    return ReadmeCheckResult(
        model_name=model_name,
        tier=tier,
        readme_path=readme_path or Path("README.md"),
        exists=True,
        headings=headings,
        present_sections=present,
        missing_sections=missing,
    )


def check_model_dir(
    model_dir: Path,
    *,
    tier: str = "supported",
    required_sections: dict[str, tuple[str, ...]] | None = None,
) -> ReadmeCheckResult:
    """Check ``README.md`` inside a single model package directory."""
    readme_path = model_dir / "README.md"
    model_name = model_dir.name
    sections = required_sections or REQUIRED_SECTIONS

    if not readme_path.is_file():
        return ReadmeCheckResult(
            model_name=model_name,
            tier=tier,
            readme_path=readme_path,
            exists=False,
            headings=[],
            present_sections=[],
            missing_sections=list(sections.keys()),
        )

    text = readme_path.read_text(encoding="utf-8")
    return check_readme_text(
        text,
        model_name=model_name,
        tier=tier,
        readme_path=readme_path,
        required_sections=sections,
    )


def check_all_readmes(repo_root: Path | None = None) -> list[ReadmeCheckResult]:
    """Scan all supported and experimental model directories in the repository."""
    root = repo_root or REPO_ROOT
    results: list[ReadmeCheckResult] = []

    tier_roots = [
        ("supported", root / "katabatic" / "models"),
        ("experimental", root / "katabatic" / "experimental" / "models"),
    ]

    for tier, base_dir in tier_roots:
        if not base_dir.is_dir():
            continue
        for child in sorted(base_dir.iterdir(), key=lambda p: p.name):
            if not child.is_dir() or child.name.startswith("_"):
                continue
            results.append(check_model_dir(child, tier=tier))

    return results


def format_markdown_report(results: list[ReadmeCheckResult]) -> str:
    """Render a Markdown table summarizing README completeness across all models."""
    lines = [
        "| Model | Tier | Status | Present Sections | Missing Sections |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in results:
        present_str = ", ".join(r.present_sections) if r.present_sections else "-"
        missing_str = ", ".join(r.missing_sections) if r.missing_sections else "-"
        lines.append(
            f"| `{r.model_name}` | {r.tier} | {r.status} | {present_str} | {missing_str} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check model README completeness across Katabatic models."
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Print the report as a Markdown table.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Include detected headings for each model README.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if any model README is missing or incomplete.",
    )
    args = parser.parse_args(argv)

    results = check_all_readmes()

    if args.markdown:
        print(format_markdown_report(results))
    else:
        for r in results:
            missing_note = (
                f" (missing: {', '.join(r.missing_sections)})"
                if r.missing_sections
                else ""
            )
            print(f"{r.model_name:<18} [{r.tier:<12}] {r.status}{missing_note}")
            if args.verbose and r.headings:
                print(f"    headings: {', '.join(r.headings)}")

    incomplete = [r for r in results if r.status != "OK"]
    if args.strict and incomplete:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
