#!/usr/bin/env python3
"""Merge main.tex and all \\input'd files into a single flat .tex document.

Recursively resolves \\input{...} commands, strips standalone wrappers
(\\documentclass, \\usepackage, \\begin{document}, \\end{document}) from
included files, and writes the result to merged.tex.

Usage:
    python merge.py                  # writes merged.tex
    python merge.py -o output.tex    # writes to output.tex
"""

import argparse
import re
from pathlib import Path

INPUT_RE = re.compile(r"^(\s*)\\input\{([^}]+)\}")
STRIP_PATTERNS = [
    re.compile(r"^\s*\\documentclass.*"),
    re.compile(r"^\s*\\usepackage.*"),
    re.compile(r"^\s*\\begin\{document\}"),
    re.compile(r"^\s*\\end\{document\}"),
]


def strip_standalone_wrapper(lines: list[str]) -> list[str]:
    """Remove standalone preamble/wrapper lines from an included file."""
    return [l for l in lines if not any(p.match(l) for p in STRIP_PATTERNS)]


def resolve_inputs(tex_path: Path, base_dir: Path, depth: int = 0) -> list[str]:
    """Recursively resolve \\input commands, returning merged lines."""
    lines = tex_path.read_text(encoding="utf-8").splitlines(keepends=True)

    if depth > 0:
        lines = strip_standalone_wrapper(lines)

    result = []
    for line in lines:
        m = INPUT_RE.match(line)
        if m:
            rel_path = m.group(2)
            # Add .tex extension if missing
            if not rel_path.endswith(".tex"):
                rel_path += ".tex"
            child = base_dir / rel_path
            if child.exists():
                result.append(f"% ---- begin {rel_path} ----\n")
                result.extend(resolve_inputs(child, base_dir, depth + 1))
                result.append(f"% ---- end {rel_path} ----\n")
            else:
                # Keep the original \input if file not found
                result.append(f"% WARNING: file not found: {rel_path}\n")
                result.append(line)
        else:
            result.append(line)

    return result


def main():
    parser = argparse.ArgumentParser(description="Merge main.tex into a single flat file")
    parser.add_argument("-o", "--output", default="merged.tex", help="Output file path")
    parser.add_argument("input", nargs="?", default="main.tex", help="Root .tex file")
    args = parser.parse_args()

    root = Path(args.input)
    base_dir = root.parent if root.parent != Path() else Path(".")
    merged = resolve_inputs(root, base_dir)

    out = Path(args.output)
    out.write_text("".join(merged), encoding="utf-8")
    print(f"Wrote {out} ({len(merged)} lines)")


if __name__ == "__main__":
    main()
