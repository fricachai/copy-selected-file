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

    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


# =============================
# APA parsing
# =============================
def split_apa_entries(raw: str) -> list[str]:
    raw = (raw or "").strip()
    if not raw:
        return []

    # Prefer blank line separators
    chunks = [c.strip() for c in re.split(r"\n\s*\n+", raw) if c.strip()]
    if len(chunks) >= 2:
        return chunks

    # Fallback: try split by patterns like "(2020)."
    raw2 = re.sub(r"\s*(?=(?:^|[\.\)])\s*\(\d{4}[a-z]?\)\.)", "\n\n", raw)
    chunks2 = [c.strip() for c in re.split(r"\n\s*\n+", raw2) if c.strip()]
    if len(chunks2) >= 2:
        return chunks2

    # Final fallback: line by line
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    return lines


def parse_apa_entry(entry: str) -> dict:
    original = (entry or "").strip()
    authors_raw = ""
    year = None
    title = ""

    # common: (YYYY).
    m_year = re.search(r"\((\d{4})(?:[a-z])?\)\.", original)
    if m_year:
        year = int(m_year.group(1))
        authors_raw = original[: m_year.start()].strip()
        rest = original[m_year.end():].strip()

        # title: next sentence-ish
        m_title = re.search(r"(.+?)(?:\.\s+[A-Z]|$)", rest)
        if m_title:
            title = m_title.group(1).strip()
        else:
            title = rest.split(".")[0].strip()
    else:
        # fallback year
        m_year2 = re.search(r"\b(19\d{2}|20\d{2})\b", original)
        if m_year2:
            year = int(m_year2.group(1))
        parts = [p.strip() for p in original.split(".") if p.strip()]
        if parts:
            authors_raw = parts[0]
        if len(parts) >= 2:
            title = parts[1]

    # cleanup: drop trailing URL/doi
    title = re.sub(r"\s*https?://\S+\s*$", "", title).strip()
    title = re.sub(r"\s*doi:\s*\S+\s*$", "", title, flags=re.I).strip()

    return {
        "raw": original,
        "authors": authors_raw,
        "year": year,
        "title": title,
        "authors_norm": normalize_text(authors_raw),
        "title_norm": normalize_text(title),
    }


def extract_first_author_surname(authors_raw: str) -> str:
    a = (authors_raw or "").strip()
    if not a:
        return ""
    a = re.sub(r"^\W+", "", a)
    if "," in a:
        return a.split(",", 1)[0].strip()
    return a.split()[0].strip()


# =============================
# Matching uploaded files
# =============================
def score_match(ref: dict, filename: str) -> dict:
    fname_noext = os.path.splitext(filename)[0]
    fname_norm = normalize_text(fname_noext)

    title_sim = similarity(ref["title_norm"], fname_norm) if ref["title_norm"] else 0.0
    year_hit = False
    author_hit = False

    if ref["year"] is not None and str(ref["year"]) in fname_noext:
        year_hit = True

    surname = extract_first_author_surname(ref["authors"])
    if surname:
        if normalize_text(surname) in fname_norm:
            author_hit = True

    total = (0.78 * title_sim) + (0.12 * (1.0 if year_hit else 0.0)) + (0.10 * (1.0 if author_hit else 0.0))

    return {
        "filename": filename,
        "total": float(total),
        "title_sim": float(title_sim),
        "year_hit": year_hit,
        "author_hit": author_hit,
    }


def best_match(ref: dict, uploaded_files: list) -> dict | None:
    best = None
    best_score = -1.0
    for uf in uploaded_files:
        det = score_match(ref, uf.name)
        if det["total"] > best_score:
            best_score = det["total"]
            best = det
    return best


def build_zip(selected_files: list) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for uf in selected_files:
            z.writestr(uf.name, uf.getvalue())
    return buf.getvalue()


# =============================
# Streamlit UI
# =============================
st.set_page_config(page_title="APA 參考文獻 ↔ 檔案比對（雲端版）", layout="wide")
st.title("APA 參考文獻 ↔ 檔案比對（Streamlit Cloud 可用）")

st.caption("限制提醒：雲端 App 無法直接讀你的『本機資料夾』，必須改成『多檔上傳』，最後下載 ZIP。")

left, right = st.columns([1, 1])

with left:
    st.subheader("① 貼上多筆 APA 參考文獻")
    apa_text = st.text_area(
        "建議每筆之間空一行（最穩定）",
        height=220,
        placeholder="Smith, J. (2020). Title of article. Journal Name, 12(3), 1-10. https://doi.org/xxxx\n\n..."
    )

    parse_btn = st.button("解析 APA → 表格", type="primary")

with right:
    st.subheader("② 上傳你的參考文獻檔案（多檔）")
    uploaded = st.file_uploader(
        "把那個資料夾內的 PDF/Word 檔全選拖進來",
        type=["pdf", "doc", "docx", "txt", "rtf"],
        accept_multiple_files=True
    )
    threshold = st.slider("自動勾選門檻（越高越保守）", min_value=0.50, max_value=0.95, value=0.72, step=0.01)

# Parse
if "refs" not in st.session_state:
    st.session_state.refs = []
if "matches" not in st.session_state:
    st.session_state.matches = []

if parse_btn:
    entries = split_apa_entries(apa_text)
    refs = [parse_apa_entry(e) for e in entries]
    st.session_state.refs = refs
    st.session_state.matches = []
    st.success(f"已解析 {len(refs)} 筆。")

refs = st.session_state.refs

if refs:
    st.subheader("③ 解析結果")
    df_refs = pd.DataFrame([{
        "Ref#": i + 1,
        "Year": r["year"],
        "Title": r["title"],
        "Authors": r["authors"],
    } for i, r in enumerate(refs)])
    st.dataframe(df_refs, use_container_width=True, hide_index=True)

if refs and uploaded:
    st.subheader("④ 自動比對（以檔名）並勾選")
    matches = []
    for i, r in enumerate(refs):
        det = best_match(r, uploaded)
        if det is None:
            matches.append({
                "Ref#": i + 1,
                "Pick": False,
                "Score": 0.0,
                "BestFile": "",
                "TitleSim": 0.0,
                "YearHit": False,
                "AuthorHit": False,
            })
        else:
            pick = det["total"] >= threshold
            matches.append({
                "Ref#": i + 1,
                "Pick": pick,
                "Score": det["total"],
                "BestFile": det["filename"],
                "TitleSim": det["title_sim"],
                "YearHit": det["year_hit"],
                "AuthorHit": det["author_hit"],
            })

    df_m = pd.DataFrame(matches)
    edited = st.data_editor(
        df_m,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Pick": st.column_config.CheckboxColumn("勾選（要打包）"),
            "Score": st.column_config.NumberColumn("分數", format="%.3f"),
            "TitleSim": st.column_config.NumberColumn("篇名相似度", format="%.3f"),
        },
        disabled=["Ref#", "Score", "BestFile", "TitleSim", "YearHit", "AuthorHit"]
    )

    # Determine selected files by BestFile where Pick=True
    pick_files = set(edited.loc[edited["Pick"] == True, "BestFile"].tolist())
    selected_files = [uf for uf in uploaded if uf.name in pick_files]

    c1, c2 = st.columns([1, 1])
    with c1:
        st.write(f"已勾選：{len(selected_files)} 個檔案")
        st.write("勾選清單：")
        st.code("\n".join([uf.name for uf in selected_files]) or "(無)")

    with c2:
        if selected_files:
            zip_bytes = build_zip(selected_files)
            st.download_button(
                "下載已勾選檔案 ZIP",
                data=zip_bytes,
                file_name="selected_references.zip",
                mime="application/zip",
                type="primary"
            )
        else:
            st.info("沒有勾選檔案可打包下載。")
