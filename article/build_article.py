#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_article.py — article/makale.md -> article/makale.docx (python-docx)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.shared import Pt, RGBColor

KOK = Path(__file__).resolve().parent
MD = KOK / "makale.md"
DOCX = KOK / "makale.docx"

BASLIK_RENK = RGBColor(0x1F, 0x4E, 0x79)


def _satir_meta(satir: str):
    if satir.startswith("# "):
        return ("baslik", satir[2:].strip())
    if satir.startswith("## "):
        return ("bolum", satir[3:].strip())
    if satir.startswith("*") and satir.endswith("*") and not satir.startswith("**"):
        return ("altbaslik", satir.strip("*").strip())
    if satir.startswith("**") and satir.endswith("**"):
        return ("lede", satir.strip("*").strip())
    return ("paragraf", satir)


def _sade(metin: str) -> str:
    """Markdown vurgularını düz metne çevir (tablo hücreleri/bağlantılar için)."""
    metin = re.sub(r"\*\*(.+?)\*\*", r"\1", metin)
    metin = re.sub(r"\*(.+?)\*", r"\1", metin)
    metin = re.sub(r"\[(.+?)\]\((.+?)\)", r"\1", metin)
    metin = metin.replace("`", "")
    return metin.strip()


def _zengin(paragraf, metin: str, temel_boyut: int = 11):
    """**kalın** ve *italik* destekli paragraf doldur."""
    for i, parca in enumerate(re.split(r"(\*\*.+?\*\*|\*.+?\*)", metin)):
        if not parca:
            continue
        if parca.startswith("**") and parca.endswith("**"):
            run = paragraf.add_run(parca[2:-2]); run.bold = True
        elif parca.startswith("*") and parca.endswith("*") and len(parca) > 2:
            run = paragraf.add_run(parca[1:-1]); run.italic = True
        else:
            run = paragraf.add_run(parca)
        run.font.size = Pt(temel_boyut)


def insa(md: Path, docx: Path) -> None:
    doc = Document()
    satirlar = md.read_text(encoding="utf-8").splitlines()

    i = 0
    while i < len(satirlar):
        satir = satirlar[i].rstrip()
        if not satir.strip():
            i += 1
            continue

        # tablo
        if satir.startswith("|") and i + 1 < len(satirlar) and "---" in satirlar[i + 1]:
            basliklar = [c.strip() for c in satir.strip("|").split("|")]
            tablo = doc.add_table(rows=1, cols=len(basliklar))
            tablo.style = "Light List Accent 1"
            for j, h in enumerate(basliklar):
                tablo.rows[0].cells[j].text = _sade(h)
            i += 2
            while i < len(satirlar) and satirlar[i].strip().startswith("|"):
                hucreler = [c.strip() for c in satirlar[i].strip().strip("|").split("|")]
                satir_cells = tablo.add_row().cells
                for j, c in enumerate(hucreler):
                    satir_cells[j].text = _sade(c)
                i += 1
            doc.add_paragraph()
            continue

        tur, icerik = _satir_meta(satir)
        if tur == "baslik":
            p = doc.add_paragraph(); run = p.add_run(icerik)
            run.bold = True; run.font.size = Pt(16); run.font.color.rgb = BASLIK_RENK
        elif tur == "bolum":
            p = doc.add_paragraph(); run = p.add_run(icerik)
            run.bold = True; run.font.size = Pt(13); run.font.color.rgb = BASLIK_RENK
        elif tur == "altbaslik":
            p = doc.add_paragraph(); run = p.add_run(icerik)
            run.italic = True; run.font.size = Pt(11)
        elif tur == "lede":
            p = doc.add_paragraph(); _zengin(p, icerik, 11)
            for r in p.runs:
                r.bold = True
        elif satir.strip().startswith("- "):
            p = doc.add_paragraph(style="List Bullet")
            _zengin(p, satir.strip()[2:], 10)
        else:
            p = doc.add_paragraph(); _zengin(p, icerik, 11)
        i += 1

    docx.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(docx))
    print(f"YAZILDI: {docx}")


if __name__ == "__main__":
    insa(MD, DOCX)
