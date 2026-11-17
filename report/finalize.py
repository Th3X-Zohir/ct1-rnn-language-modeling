"""Two-pass report build: DOCX -> PDF, locate every heading in the PDF, write real page
numbers into the static table of contents, rebuild, and render page previews.

Usage (from report/):  python finalize.py
Env: SOFFICE = path to LibreOffice soffice binary.
"""
import json
import os
import subprocess
from pathlib import Path

import pymupdf

HERE = Path(__file__).resolve().parent
NAME = "CT1_Task3_Report_Recurrent_Minds"
SOFFICE = os.environ.get("SOFFICE", "soffice")


def build():
    subprocess.run(["node", "build_report.js"], cwd=HERE, check=True)
    subprocess.run([SOFFICE, "--headless", "--norestore",
                    "-env:UserInstallation=file:///root/msc_work/tools/lo_profile",
                    "--convert-to", "pdf", f"{NAME}.docx"], cwd=HERE, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return pymupdf.open(HERE / f"{NAME}.pdf")


def heading_pages(doc, headings):
    """Find each heading as a large bold span (>= 12 pt) — TOC lines are smaller."""
    found, start = {}, 0
    big = []
    for i, page in enumerate(doc):
        for b in page.get_text("dict")["blocks"]:
            for l in b.get("lines", []):
                t = "".join(s["text"] for s in l["spans"]).strip()
                if t and max(s["size"] for s in l["spans"]) >= 12:
                    big.append((i, t))
    for h in headings:
        for k in range(start, len(big)):
            i, t = big[k]
            if t == h or (h.startswith(t) and len(t) > 12):
                found[h] = i          # cover = index 0 has no number, so index == printed page number
                start = k + 1
                break
    return found


if __name__ == "__main__":
    (HERE / "toc_pages.json").unlink(missing_ok=True)
    doc = build()
    heads = json.loads((HERE / "toc_headings.json").read_text())
    for _ in range(2):  # second pass in case the TOC length shifted pages
        pages = heading_pages(doc, heads)
        (HERE / "toc_pages.json").write_text(json.dumps(pages, indent=1))
        doc.close()
        doc = build()
    final = heading_pages(doc, heads)
    missing = [h for h in heads if h not in final]
    print("pages:", len(doc), "| headings located:", len(final), "/", len(heads), "| missing:", missing)
    print("toc consistent:", final == json.loads((HERE / "toc_pages.json").read_text()))
    prev = HERE / "preview"; prev.mkdir(exist_ok=True)
    for i, p in enumerate(doc):
        p.get_pixmap(dpi=70).save(prev / f"p{i + 1:02d}.png")
