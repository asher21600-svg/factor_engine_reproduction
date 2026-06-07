#!/usr/bin/env python
"""Phase 5: build the self-contained HTML report and the PDF."""
import _bootstrap  # noqa: F401

from fe import config
from fe.report.build_report import build_html
from fe.report.build_pdf import html_to_pdf


def main():
    html = build_html(config.OUTPUTS)
    print(f"-> HTML report: {html}")
    pdf = html_to_pdf(html, config.OUTPUTS / "reproduction_report.pdf")
    if pdf:
        print(f"-> PDF report:  {pdf}")


if __name__ == "__main__":
    main()
