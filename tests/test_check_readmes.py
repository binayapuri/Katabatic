"""Unit tests for benchmarks/check_readmes.py."""

from __future__ import annotations

from pathlib import Path

from benchmarks.check_readmes import (
    REQUIRED_SECTIONS,
    check_all_readmes,
    check_model_dir,
    check_readme_text,
    extract_headings,
    format_markdown_report,
    main,
)


def test_extract_headings_ignores_fenced_code_blocks():
    md = (
        "# CTGAN\n"
        "## Overview\n"
        "```python\n"
        "# comment inside code block\n"
        "```\n"
        "### Quick Start\n"
    )
    assert extract_headings(md) == ["ctgan", "overview", "quick start"]


def test_complete_readme_text_passes_all_required_sections():
    md = (
        "# Example Model\n"
        "## Model Overview\n"
        "## Research Paper\n"
        "## Hyperparameters\n"
        "## Quick Start\n"
        "## Implementation Notes\n"
    )
    res = check_readme_text(md, model_name="example")
    assert res.status == "OK"
    assert res.missing_sections == []
    assert set(res.present_sections) == set(REQUIRED_SECTIONS.keys())


def test_incomplete_readme_reports_missing_sections():
    md = "# Partial Model\n## Overview\n"
    res = check_readme_text(md, model_name="partial")
    assert res.status == "INCOMPLETE"
    assert "overview" in res.present_sections
    assert "reference" in res.missing_sections
    assert "parameters" in res.missing_sections


def test_missing_readme_file_is_flagged(tmp_path: Path):
    model_dir = tmp_path / "custom_model"
    model_dir.mkdir()

    res = check_model_dir(model_dir, tier="experimental")
    assert res.exists is False
    assert res.status == "MISSING_FILE"
    assert set(res.missing_sections) == set(REQUIRED_SECTIONS.keys())


def test_all_repository_model_readmes_exist():
    results = check_all_readmes()
    assert len(results) >= 22
    missing_files = [r.model_name for r in results if not r.exists]
    assert missing_files == [], f"Models missing README.md: {missing_files}"


def test_format_markdown_report_and_cli_exit_code(capsys):
    results = check_all_readmes()
    table = format_markdown_report(results)
    assert "| `ctgan` | supported |" in table
    assert "| `great` | supported |" in table
    assert "| `tabddpm` | experimental |" in table

    exit_code = main(["--markdown"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "| Model | Tier | Status |" in captured.out
