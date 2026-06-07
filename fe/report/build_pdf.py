"""HTML -> PDF via headless Chrome (Mac/Linux), with weasyprint fallback."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "google-chrome", "chromium", "chromium-browser",
]


def _find_chrome():
    for c in CHROME_CANDIDATES:
        if os.path.exists(c) or shutil.which(c):
            return c
    return None


def html_to_pdf(html_path: Path, pdf_path: Path | None = None) -> Path | None:
    html_path = Path(html_path)
    pdf_path = Path(pdf_path or html_path.with_suffix(".pdf"))
    chrome = _find_chrome()
    if chrome:
        try:
            subprocess.run(
                [chrome, "--headless", "--disable-gpu", "--no-sandbox",
                 f"--print-to-pdf={pdf_path}", "--no-pdf-header-footer",
                 f"file://{html_path.resolve()}"],
                check=True, capture_output=True, timeout=120)
            if pdf_path.exists():
                return pdf_path
        except Exception as e:  # noqa: BLE001
            print(f"  chrome pdf failed: {e}")
    try:
        from weasyprint import HTML
        HTML(filename=str(html_path)).write_pdf(str(pdf_path))
        return pdf_path
    except Exception as e:  # noqa: BLE001
        print(f"  weasyprint unavailable: {e}")
    print("  PDF generation skipped (no Chrome/weasyprint). HTML report is self-contained.")
    return None
