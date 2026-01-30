# -*- coding: utf-8 -*-
import re
import io
import os
import unicodedata
import zipfile
from difflib import SequenceMatcher

import pandas as pd
import streamlit as st


# =============================
# Normalize & similarity
# =============================
def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


# =============================
# ✅ 修正後 APA 分段（關鍵）
# =============================
def split_apa_entries(raw: str) -> list[str]:
    raw = (raw or "").strip()
    if not raw:
        return []

    lines = [ln.rstrip() for ln in raw.splitlines() if ln.strip()]
    entries = []
    buf = []

    start_pat = re.compile(r"^[A-Z][A-Za-z'’\-]+,\s*[A-Z]")
    year_pat = re.compile(r"\(\d{4}[a-z]?\)\.")

    def flush():
        nonlocal buf
        if buf:
            entries.append(" ".join(buf).strip())
            buf = []

    for ln in lines:
        is_new = bool(start_pat.search(ln))
        buf_has_year = bool(year_pat.search(" ".join(buf))) if buf else False

        if is_new and buf and buf_has_year:
            flush()
        buf.append(ln)

    flush()
    return entries


# =============================
# ✅ 修正後 APA 解析（關鍵）
# =============================
def parse_apa_entry(entry: str) -> dict:
    original = entry.strip()
    authors_raw = ""
    year = None
    title = ""

    m_year = re.search(r"\((\d{4})(?:[a-z])?\)\.", original)
    if m_year:
        year = int(m_year.group(1))
        authors_raw = original[:m_year.start()].strip()
        rest = original[m_year.end():].strip()
        title = rest.split(".")[0].strip()

    title = re.sub(r"https?://\S+$", "", title).strip()

    return {
        "raw": original,
        "authors": authors_raw,
        "year": year,
        "title": title,
        "authors_norm": normalize_text(authors_raw),
        "title_norm": normalize_text(title),
    }


def extract_first_author_surname(authors_raw: str) -> str:
    if not authors_raw:
        return ""
    return authors_raw.split(",")[0].strip()


# =============================
# Matching
# =============================
def score_match(ref: dict, filename: str) -> dict:
    fname_norm = normalize_text(os.path.splitext(filename)[0])
    title_sim = similarity(ref["title_norm"], fname_norm)
    year_hit = ref["year"] and str(ref["year"]) in filename
    author_hit = normalize_text(extract_first_author_surname(ref["authors"])) in fname_norm

    score = 0.78 * title_sim + 0.12 * year_hit + 0.10 * author_hit
    return {
        "filename": filename,
        "score": score,
        "title_sim": title_sim,
        "year_hit": year_hit,
        "author_hit": author_hit,
    }


def best_match(ref: dict, uploaded_files):
    best = None
    for uf in uploaded_files:
        r = score_match(ref, uf.name)
        if not best or r["score"] > best["score"]:
            best = r
    return best


def build_zip(files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.writestr(f.name, f.getvalue())
    return buf.getvalue()


# =============================
# Streamlit UI
# =============================
st.set_page_config(page_title="APA Reference Matcher", layout="wide")
st.title("APA 參考文獻 → 檔案比對（修正版）")

apa_text = st.text_area("貼上多筆 APA（允許跨行）", height=200)

if st.button("解析 APA → 表格"):
    entries = split_apa_entries(apa_text)
    refs = [parse_apa_entry(e) for e in entries]
    st.session_state.refs = refs

refs = st.session_state.get("refs", [])

if refs:
    df = pd.DataFrame([
        {"Ref#": i + 1, "Year": r["year"], "Title": r["title"], "Authors": r["authors"]}
        for i, r in enumerate(refs)
    ])
    st.dataframe(df, use_container_width=True)

uploaded = st.file_uploader(
    "上傳參考文獻檔案（PDF/Word）",
    accept_multiple_files=True,
    type=["pdf", "doc", "docx"]
)

if refs and uploaded:
    matches = []
    for i, r in enumerate(refs):
        m = best_match(r, uploaded)
        matches.append({
            "Ref#": i + 1,
            "Pick": m["score"] >= 0.72 if m else False,
            "Score": round(m["score"], 3) if m else 0,
            "File": m["filename"] if m else "",
        })

    dfm = pd.DataFrame(matches)
    edited = st.data_editor(dfm, use_container_width=True)

    picked = [uf for uf in uploaded if uf.name in edited[edited["Pick"]]["File"].tolist()]
    if picked:
        st.download_button(
            "下載勾選檔案 ZIP",
            data=build_zip(picked),
            file_name="selected_refs.zip"
        )
