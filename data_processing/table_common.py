"""Shared LaTeX table helpers for circuit_table.py and scheduling_table.py."""

import os
import shutil
import subprocess
import sys
import tempfile


def latex_escape(s):
    """Escape special LaTeX characters in a string."""
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]
    for old, new in replacements:
        s = s.replace(old, new)
    return s


def generate_pdf(latex_table: str, pdf_path: str) -> None:
    """
    Wrap *latex_table* in a minimal standalone document and render it to
    *pdf_path*.  Tries pdflatex / xelatex / lualatex first (they compile the
    .tex directly and preserve all LaTeX commands), then falls back to pandoc.
    Raises RuntimeError if no suitable tool is found.
    """
    standalone_doc = (
        r"\documentclass{article}" + "\n"
        r"\usepackage{booktabs}" + "\n"
        r"\usepackage{makecell}" + "\n"
        r"\usepackage{geometry}" + "\n"
        r"\geometry{margin=1in}" + "\n"
        r"\begin{document}" + "\n"
        r"\pagestyle{empty}" + "\n" + latex_table + "\n"
        r"\end{document}" + "\n"
    )

    pdf_path = os.path.abspath(pdf_path)

    # --- try pdflatex / xelatex / lualatex (compile .tex directly) ---
    for engine in ("pdflatex", "xelatex", "lualatex"):
        if not shutil.which(engine):
            continue
        with tempfile.TemporaryDirectory() as tmpdir:
            tex_path = os.path.join(tmpdir, "table.tex")
            with open(tex_path, "w") as f:
                f.write(standalone_doc)
            try:
                subprocess.run(
                    [engine, "-interaction=nonstopmode", "-output-directory", tmpdir, tex_path],
                    check=True,
                    capture_output=True,
                )
                shutil.copy(os.path.join(tmpdir, "table.pdf"), pdf_path)
                print(f"PDF written to {pdf_path} (via {engine})", file=sys.stderr)
                return
            except subprocess.CalledProcessError as e:
                print(f"{engine} failed: {e.stderr.decode()}", file=sys.stderr)

    # --- fall back to pandoc (passes --pdf-engine so it invokes pdflatex) ---
    if shutil.which("pandoc"):
        with tempfile.NamedTemporaryFile(suffix=".tex", mode="w", delete=False) as tmp:
            tmp.write(standalone_doc)
            tex_path = tmp.name
        try:
            subprocess.run(
                ["pandoc", tex_path, "-o", pdf_path, "--pdf-engine=pdflatex", "--from=latex"],
                check=True,
                capture_output=True,
            )
            print(f"PDF written to {pdf_path} (via pandoc)", file=sys.stderr)
            return
        except subprocess.CalledProcessError as e:
            print(f"pandoc failed: {e.stderr.decode()}", file=sys.stderr)
        finally:
            os.unlink(tex_path)

    raise RuntimeError("No PDF renderer found. Install pdflatex, xelatex, lualatex, or pandoc.")
