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
    """Normalize text for matching: lowercase, remove diacritics, unify punctuation, remove most punct."""
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
    """Return similarity in [0,1]."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


# =============================
# APA split & parse (robust for hard line breaks)
# =============================
def split_apa_entries(raw: str) -> list[str]:
    """
    Robust segmentation for pasted APA that may contain hard line breaks.
    Strategy:
      - Treat a new entry start when line looks like 'Surname, X' AND previous buffer already has a (YYYY).
    """
    raw = (raw or "").strip()
    if not raw:
        return []

    lines = [ln.rstrip() for ln in raw.splitlines() if ln.strip()]
    entries: list[str] = []
    buf: list[str] = []

    start_pat = re.compile(r"^[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’\-]+,\s*[A-Z]")  # Surname, Initial
    year_pat = re.compile(r"\(\d{4}[a-z]?\)\.")  # (2009). / (2020a).

    def flush():
        nonlocal buf
        if buf:
            entries.append(" ".join(buf).strip())
            buf = []

    for ln in lines:
        looks_like_start = bool(start_pat.search(ln.strip()))
        buf_has_year = bool(year_pat.search(" ".join(buf))) if buf else False

        if looks_like_start and buf and buf_has_year:
            flush()

        buf.append(ln.strip())

    flush()
    return entries


def parse_apa_entry(entry: str) -> dict:
    """
    Parse:
      - authors: before (YYYY).
      - year: (YYYY).
      - title: first sentence after (YYYY). (up to first period)
    """
    original = (entry or "").strip()
    authors_raw = ""
    year = None
    title = ""

    m_year = re.search(r"\((\d{4})(?:[a-z])?\)\.", original)
    if m_year:
        year = int(m_year.group(1))
        authors_raw = original[: m_year.start()].strip()
        rest = original[m_year.end():].strip()
        title = rest.split(".")[0].strip()
    else:
        # fallback: try to find 4-digit year anywhere
        m_year2 = re.search(r"\b(19\d{2}|20\d{2})\b", original)
        if m_year2:
            year = int(m_year2.group(1))

        # avoid splitting on '.' because initials like T.-H. contain dots
        m_paren = re.search(r"\(\d{4}", original)
        if m_paren:
            authors_raw = original[: m_paren.start()].strip()
        else:
            # fallback: take up to first " . " boundary with uppercase afterwards
            m_boundary = re.search(r"\.\s+[A-Z]", original)
            authors_raw = original[: m_boundary.start()].strip() if m_boundary else original

        title = ""

    # strip trailing URL/doi in title
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
    """Take surname before first comma."""
    a = (authors_raw or "").strip()
    if not a:
        return ""
    if "," in a:
        return a.split(",", 1)[0].strip()
    return a.split()[0].strip()


# =============================
# ZIP ingestion (100+ files)
# =============================
SUPPORTED_EXT = {".pdf", ".doc", ".docx", ".txt", ".rtf"}

def load_files_from_zip(uploaded_zip) -> list[dict]:
    """
    Return a list of dict items: {"name": filename, "data": bytes}
    - Flatten folder paths (use basename for matching/display)
    - If duplicate basenames exist, add suffix to keep unique.
    """
    zbytes = io.BytesIO(uploaded_zip.getvalue())
    out: list[dict] = []
    name_counts: dict[str, int] = {}

    with zipfile.ZipFile(zbytes, "r") as z:
        for member in z.namelist():
            if member.endswith("/"):
                continue
            ext = os.path.splitext(member)[1].lower()
            if ext not in SUPPORTED_EXT:
                continue

            data = z.read(member)
            base = os.path.basename(member)

            # de-dup basename
            if base in name_counts:
                name_counts[base] += 1
                stem, ext2 = os.path.splitext(base)
                base2 = f"{stem}__dup{name_counts[base]}{ext2}"
                base = base2
            else:
                name_counts[base] = 1

            out.append({"name": base, "data": data})

    return out


def get_file_name(uf):
    return uf["name"] if isinstance(uf, dict) else uf.name


def get_file_bytes(uf):
    return uf["data"] if isinstance(uf, dict) else uf.getvalue()


# =============================
# Matching
# =============================
def score_match(ref: dict, filename: str) -> dict:
    fname_noext = os.path.splitext(filename)[0]
    fname_norm = normalize_text(fname_noext)

    title_sim = similarity(ref["title_norm"], fname_norm) if ref["title_norm"] else 0.0

    year_hit = False
    if ref["year"] is not None and str(ref["year"]) in fname_noext:
        year_hit = True

    author_hit = False
    surname = extract_first_author_surname(ref["authors"])
    if surname:
        if normalize_text(surname) in fname_norm:
            author_hit = True

    # weighted
    total = (0.78 * title_sim) + (0.12 * (1.0 if year_hit else 0.0)) + (0.10 * (1.0 if author_hit else 0.0))

    return {
        "filename": filename,
        "total": float(total),
        "title_sim": float(title_sim),
        "year_hit": bool(year_hit),
        "author_hit": bool(author_hit),
    }


def best_match(ref: dict, uploaded_files: list[dict]) -> dict | None:
    if not uploaded_files:
        return None
    best = None
    best_score = -1.0
    for uf in uploaded_files:
        det = score_match(ref, get_file_name(uf))
        if det["total"] > best_score:
            best_score = det["total"]
            best = det
    return best


def build_zip(selected_files: list) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for uf in selected_files:
            z.writestr(get_file_name(uf), get_file_bytes(uf))
    return buf.getvalue()


# =============================
# Streamlit UI
# =============================
st.set_page_config(page_title="APA 參考文獻 ↔ 檔案比對（ZIP 版）", layout="wide")
st.title("APA 參考文獻 ↔ 檔案比對（ZIP 版：支援 100+ 檔）")
st.caption("雲端限制：無法直接讀本機資料夾，所以改成上傳 ZIP → 解壓 → 比對 → 勾選 → 下載 ZIP。")


# ---- Session state
if "refs" not in st.session_state:
    st.session_state.refs = []
if "files" not in st.session_state:
    st.session_state.files = []
if "matches_df" not in st.session_state:
    st.session_state.matches_df = None


# ---- Layout
left, right = st.columns([1, 1])

with left:
    st.subheader("① 貼上多筆 APA 參考文獻")
    apa_text = st.text_area(
        "建議每筆之間空一行（但即使跨行也能解析）",
        height=260,
        placeholder="Chu, T.-H., & Peng, H.-L. (2009). Investigation ...\n\nDavis, F. D. (1989). Perceived usefulness ..."
    )

    parse_btn = st.button("解析 APA → 表格", type="primary")

with right:
    st.subheader("② 上傳 1 個 ZIP（內含 100+ PDF/Word）")
    uploaded_zip = st.file_uploader(
        "請上傳 ZIP（可含子資料夾；支援 PDF/DOC/DOCX/TXT/RTF）",
        type=["zip"],
        accept_multiple_files=False
    )
    threshold = st.slider("自動勾選門檻（越高越保守）", 0.50, 0.95, 0.72, 0.01)


# ---- Parse
if parse_btn:
    entries = split_apa_entries(apa_text)
    refs = [parse_apa_entry(e) for e in entries]
    st.session_state.refs = refs
    st.session_state.matches_df = None
    st.success(f"已解析 {len(refs)} 筆參考文獻。")

refs = st.session_state.refs


# ---- Load ZIP
if uploaded_zip is not None:
    try:
        files = load_files_from_zip(uploaded_zip)
        st.session_state.files = files
        st.session_state.matches_df = None
        st.success(f"ZIP 解壓完成：載入 {len(files)} 個檔案（可用於比對）")
    except Exception as e:
        st.session_state.files = []
        st.error(f"ZIP 解析失敗：{e}")

files = st.session_state.files


# ---- Display refs
if refs:
    st.subheader("③ 解析結果（Year / Title / Authors）")
    df_refs = pd.DataFrame(
        [
            {
                "Ref#": i + 1,
                "Year": r["year"] if r["year"] is not None else "",
                "Title": r["title"],
                "Authors": r["authors"],
            }
            for i, r in enumerate(refs)
        ]
    )
    st.dataframe(df_refs, use_container_width=True, hide_index=True)


# ---- Match & select
if refs and files:
    st.subheader("④ 自動比對（以檔名）→ 勾選 → 下載勾選 ZIP")

    # Build matches df only when needed
    if st.session_state.matches_df is None:
        rows = []
        for i, r in enumerate(refs):
            det = best_match(r, files)
            if det is None:
                rows.append({
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
                rows.append({
                    "Ref#": i + 1,
                    "Pick": bool(pick),
                    "Score": float(det["total"]),
                    "BestFile": det["filename"],
                    "TitleSim": float(det["title_sim"]),
                    "YearHit": bool(det["year_hit"]),
                    "AuthorHit": bool(det["author_hit"]),
                })
        st.session_state.matches_df = pd.DataFrame(rows)

    # Show editor
    df_m = st.session_state.matches_df.copy()

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

    # Update stored df (keep manual picks)
    st.session_state.matches_df = edited

    pick_files = set(edited.loc[edited["Pick"] == True, "BestFile"].tolist())
    selected_files = [uf for uf in files if get_file_name(uf) in pick_files]

    c1, c2 = st.columns([1, 1])
    with c1:
        st.write(f"已勾選檔案數：{len(selected_files)}")
        st.text_area(
            "勾選檔名清單（可複製）",
            value="\n".join([get_file_name(uf) for uf in selected_files]) if selected_files else "",
            height=180
        )

    with c2:
        if selected_files:
            zip_bytes = build_zip(selected_files)
            st.download_button(
                "下載已勾選檔案 ZIP",
                data=zip_bytes,
                file_name="selected_references.zip",
                mime="application/zip",
                type="primary",
            )
        else:
            st.info("尚未勾選任何檔案。你可以調整門檻或手動勾選。")


# ---- Footer tips
with st.expander("使用建議（提高命中率）"):
    st.markdown(
        """
- 若你的檔名很亂，建議先用規則改名（例如：`作者_年份_短篇名.pdf`），比對會更準。
- 目前比對依據：**篇名相似度為主**，年份/第一作者作為加權輔助。
- 若你想提升準確率到「讀 PDF metadata 或第一頁標題」也能做，但雲端會更耗資源。
        """
    )
