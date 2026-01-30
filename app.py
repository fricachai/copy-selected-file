# -*- coding: utf-8 -*-
"""
APA Reference -> Parse (Year/Title/Authors) -> Match files in a folder -> Check & Copy selected
Requires: Python 3.9+
Optional: requests (not required in this version)
"""

import os
import re
import shutil
import unicodedata
from difflib import SequenceMatcher
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


# -----------------------------
# Text normalization utilities
# -----------------------------
def normalize_text(s: str) -> str:
    """Normalize text for matching: lowercase, strip accents, unify quotes/dashes, remove extra spaces/punct."""
    if not s:
        return ""

    s = s.strip().lower()

    # Unicode normalize & remove diacritics
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))

    # unify punctuation variants
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = s.replace("–", "-").replace("—", "-")

    # collapse whitespace
    s = re.sub(r"\s+", " ", s)

    # remove most punctuation except letters/numbers/space
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    return s


def similarity(a: str, b: str) -> float:
    """Return similarity in [0,1]."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


# -----------------------------
# APA parsing
# -----------------------------
def split_apa_entries(raw: str) -> list[str]:
    """
    Split multiple references.
    Heuristic: split by blank lines; if none, split by newline where pattern '(YYYY).' appears.
    """
    raw = raw.strip()
    if not raw:
        return []

    # Prefer blank-line split
    chunks = [c.strip() for c in re.split(r"\n\s*\n+", raw) if c.strip()]
    if len(chunks) >= 2:
        return chunks

    # If user pasted one-per-line without blank lines, try split by year marker
    # Insert delimiter before occurrences of (YYYY).
    raw2 = re.sub(r"\s*(?=(?:^|[\.\)])\s*\(\d{4}\)\.)", "\n\n", raw)
    chunks2 = [c.strip() for c in re.split(r"\n\s*\n+", raw2) if c.strip()]

    # If still single, fallback to lines
    if len(chunks2) == 1:
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if len(lines) >= 2:
            return lines
        return chunks2

    return chunks2


def parse_apa_entry(entry: str) -> dict:
    """
    Parse an APA reference string into:
      - authors_raw (string)
      - year (int or None)
      - title (string)
    Heuristics:
      Authors: before (YYYY).
      Year: first (YYYY) found
      Title: text after '(YYYY).' up to next period that looks like end of title.
    This will not be perfect for every APA variant, but works well for common journal/book chapter formats.
    """
    original = entry.strip()
    authors_raw = ""
    year = None
    title = ""

    # find year like (2020). or (2020a).
    m_year = re.search(r"\((\d{4})(?:[a-z])?\)\.", original)
    if m_year:
        year = int(m_year.group(1))
        authors_raw = original[: m_year.start()].strip()

        rest = original[m_year.end():].strip()

        # Title is usually the next sentence: up to first period that ends the title sentence.
        # But journal names also have periods, so we use a conservative rule:
        # - take until the first period followed by space and an uppercase letter OR end.
        # - also stop before italic/journal-ish pattern? we don't have italics in plaintext; keep rule simple.
        m_title = re.search(r"(.+?)(?:\.\s+[A-Z]|$)", rest)
        if m_title:
            title = m_title.group(1).strip()
            # If we matched because of uppercase lookahead, we removed the uppercase letter; restore not needed for title.
        else:
            # fallback: first period
            title = rest.split(".")[0].strip()
    else:
        # fallback: try standalone year "2020." without parentheses
        m_year2 = re.search(r"\b(19\d{2}|20\d{2})\b", original)
        if m_year2:
            year = int(m_year2.group(1))

        # crude split by first period for authors, second for title
        parts = [p.strip() for p in original.split(".") if p.strip()]
        if parts:
            authors_raw = parts[0]
        if len(parts) >= 2:
            title = parts[1]

    # cleanup title: remove trailing DOI/URL segment if user pasted in same sentence
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
    """
    APA authors often like: "Smith, J. A., & Brown, K."
    We'll take first token before comma as surname.
    If format is different, fallback to first word.
    """
    a = authors_raw.strip()
    if not a:
        return ""
    # remove leading ellipses etc.
    a = re.sub(r"^\W+", "", a)

    if "," in a:
        return a.split(",", 1)[0].strip()
    return a.split()[0].strip()


# -----------------------------
# File matching
# -----------------------------
SUPPORTED_EXT = {".pdf", ".doc", ".docx", ".txt", ".rtf"}  # you can expand

def list_candidate_files(folder: str) -> list[str]:
    files = []
    for root, _, fnames in os.walk(folder):
        for fn in fnames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in SUPPORTED_EXT:
                files.append(os.path.join(root, fn))
    return files


def score_file_match(ref: dict, filepath: str) -> tuple[float, dict]:
    """
    Score reference vs file name using:
      - title similarity (dominant)
      - year presence
      - first author surname presence
    """
    fname = os.path.basename(filepath)
    fname_noext = os.path.splitext(fname)[0]

    fname_norm = normalize_text(fname_noext)

    title_sim = similarity(ref["title_norm"], fname_norm) if ref["title_norm"] else 0.0

    # year signal
    year_score = 0.0
    if ref["year"] is not None and str(ref["year"]) in fname_noext:
        year_score = 1.0

    # author signal
    first_surname = extract_first_author_surname(ref["authors"])
    author_score = 0.0
    if first_surname:
        first_surname_norm = normalize_text(first_surname)
        if first_surname_norm and first_surname_norm in fname_norm:
            author_score = 1.0

    # weighted score
    # Title matters most; year/author are boosters
    total = (0.78 * title_sim) + (0.12 * year_score) + (0.10 * author_score)

    details = {
        "file": filepath,
        "fname": fname,
        "title_sim": title_sim,
        "year_hit": bool(year_score),
        "author_hit": bool(author_score),
        "total": total,
    }
    return total, details


def best_match_for_ref(ref: dict, files: list[str]) -> dict | None:
    if not files:
        return None
    best = None
    best_score = -1.0
    best_details = None
    for f in files:
        sc, det = score_file_match(ref, f)
        if sc > best_score:
            best_score = sc
            best_details = det
            best = f
    if best_details is None:
        return None
    return best_details


# -----------------------------
# GUI App
# -----------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("APA 參考文獻 → 檔案比對與勾選複製工具")
        self.geometry("1200x720")

        self.refs = []              # list of parsed refs (dict)
        self.source_folder = ""
        self.files = []             # list of file paths
        self.matches = []           # per ref match details

        self._build_ui()

    def _build_ui(self):
        # Top frame
        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=8)

        ttk.Label(top, text="① 貼上多筆 APA 參考文獻（建議每筆用空白行分隔）").pack(anchor="w")

        self.txt = tk.Text(self, height=10, wrap="word")
        self.txt.pack(fill="x", padx=10)

        btns = ttk.Frame(self)
        btns.pack(fill="x", padx=10, pady=8)

        ttk.Button(btns, text="解析 APA → 表格", command=self.on_parse).pack(side="left")
        ttk.Button(btns, text="清空", command=self.on_clear).pack(side="left", padx=6)

        ttk.Separator(self).pack(fill="x", padx=10, pady=6)

        # Middle: parsed refs table
        mid = ttk.Frame(self)
        mid.pack(fill="both", expand=True, padx=10, pady=6)

        left = ttk.Frame(mid)
        left.pack(side="left", fill="both", expand=True)

        ttk.Label(left, text="② 解析結果（可先檢查年份/篇名/作者群）").pack(anchor="w")

        cols = ("idx", "year", "title", "authors")
        self.tree_refs = ttk.Treeview(left, columns=cols, show="headings", height=10)
        self.tree_refs.heading("idx", text="#")
        self.tree_refs.heading("year", text="年份")
        self.tree_refs.heading("title", text="篇名")
        self.tree_refs.heading("authors", text="作者群")
        self.tree_refs.column("idx", width=50, anchor="center")
        self.tree_refs.column("year", width=80, anchor="center")
        self.tree_refs.column("title", width=420)
        self.tree_refs.column("authors", width=320)
        self.tree_refs.pack(fill="both", expand=True)

        # Right: matching & selection
        right = ttk.Frame(mid)
        right.pack(side="left", fill="both", expand=True, padx=(12, 0))

        ttk.Label(right, text="③ 選資料夾 → 自動比對檔案（會自動勾選高分匹配）").pack(anchor="w")

        folder_bar = ttk.Frame(right)
        folder_bar.pack(fill="x", pady=(4, 6))

        self.lbl_folder = ttk.Label(folder_bar, text="來源資料夾：尚未選取", foreground="#555")
        self.lbl_folder.pack(side="left", fill="x", expand=True)

        ttk.Button(folder_bar, text="選取來源資料夾", command=self.on_choose_source_folder).pack(side="right")

        match_btns = ttk.Frame(right)
        match_btns.pack(fill="x", pady=(0, 6))

        ttk.Button(match_btns, text="掃描來源資料夾檔案", command=self.on_scan_folder).pack(side="left")
        ttk.Button(match_btns, text="自動比對並勾選", command=self.on_match).pack(side="left", padx=6)

        # Matches list with checkboxes (Treeview)
        ttk.Label(right, text="④ 比對結果（勾選=將被複製）").pack(anchor="w")

        mcols = ("pick", "ref_idx", "score", "file")
        self.tree_matches = ttk.Treeview(right, columns=mcols, show="headings", height=12)
        for c, t, w in [
            ("pick", "勾選", 60),
            ("ref_idx", "Ref#", 60),
            ("score", "分數", 80),
            ("file", "匹配檔案（路徑）", 520),
        ]:
            self.tree_matches.heading(c, text=t)
            self.tree_matches.column(c, width=w, anchor="center" if c in ("pick", "ref_idx", "score") else "w")
        self.tree_matches.pack(fill="both", expand=True)

        self.tree_matches.bind("<Double-1>", self.on_toggle_pick)

        copy_bar = ttk.Frame(right)
        copy_bar.pack(fill="x", pady=8)

        ttk.Button(copy_bar, text="全選/全不選", command=self.on_toggle_all).pack(side="left")
        ttk.Button(copy_bar, text="複製已勾選 → 選取目的資料夾", command=self.on_copy_selected).pack(side="right")

        tip = (
            "操作提示：\n"
            "- 先按「解析 APA → 表格」\n"
            "- 再選來源資料夾 → 掃描 → 自動比對\n"
            "- 需要時可雙擊「勾選」欄切換勾選狀態\n"
        )
        ttk.Label(self, text=tip, foreground="#444").pack(anchor="w", padx=10, pady=(0, 10))

    # -----------------------------
    # Handlers
    # -----------------------------
    def on_clear(self):
        self.txt.delete("1.0", "end")
        self.refs = []
        self.files = []
        self.matches = []
        self._refresh_refs_table()
        self._refresh_matches_table()

    def on_parse(self):
        raw = self.txt.get("1.0", "end").strip()
        entries = split_apa_entries(raw)
        if not entries:
            messagebox.showwarning("提醒", "沒有可解析的內容。請先貼上 APA 參考文獻。")
            return

        refs = []
        for e in entries:
            ref = parse_apa_entry(e)
            refs.append(ref)

        self.refs = refs
        self._refresh_refs_table()
        messagebox.showinfo("完成", f"已解析 {len(self.refs)} 筆參考文獻。")

    def _refresh_refs_table(self):
        for i in self.tree_refs.get_children():
            self.tree_refs.delete(i)
        for idx, r in enumerate(self.refs, start=1):
            self.tree_refs.insert(
                "", "end",
                values=(idx, r["year"] if r["year"] is not None else "", r["title"], r["authors"])
            )

    def on_choose_source_folder(self):
        folder = filedialog.askdirectory(title="選取來源資料夾（包含你的參考文獻檔案）")
        if not folder:
            return
        self.source_folder = folder
        self.lbl_folder.config(text=f"來源資料夾：{folder}")

    def on_scan_folder(self):
        if not self.source_folder:
            messagebox.showwarning("提醒", "請先選取來源資料夾。")
            return
        self.files = list_candidate_files(self.source_folder)
        messagebox.showinfo("完成", f"已掃描到 {len(self.files)} 個檔案（副檔名：{', '.join(sorted(SUPPORTED_EXT))}）。")

    def on_match(self):
        if not self.refs:
            messagebox.showwarning("提醒", "請先解析 APA。")
            return
        if not self.files:
            messagebox.showwarning("提醒", "請先掃描來源資料夾檔案。")
            return

        self.matches = []
        for idx, ref in enumerate(self.refs, start=1):
            det = best_match_for_ref(ref, self.files)
            if det is None:
                self.matches.append({"ref_idx": idx, "picked": False, "score": 0.0, "file": ""})
                continue

            # auto-pick threshold
            score = float(det["total"])
            picked = score >= 0.72  # 你可調：越高越保守
            self.matches.append({
                "ref_idx": idx,
                "picked": picked,
                "score": score,
                "file": det["file"],
                "fname": det["fname"],
                "title_sim": det["title_sim"],
                "year_hit": det["year_hit"],
                "author_hit": det["author_hit"],
            })

        self._refresh_matches_table()
        picked_n = sum(1 for m in self.matches if m.get("picked"))
        messagebox.showinfo("完成", f"比對完成：共 {len(self.matches)} 筆，已自動勾選 {picked_n} 筆（分數 ≥ 0.72）。")

    def _refresh_matches_table(self):
        for i in self.tree_matches.get_children():
            self.tree_matches.delete(i)

        for m in self.matches:
            pick_txt = "✅" if m.get("picked") else ""
            score_txt = f"{m.get('score', 0.0):.3f}" if m.get("score") is not None else ""
            file_txt = m.get("file", "")
            self.tree_matches.insert("", "end", values=(pick_txt, m.get("ref_idx", ""), score_txt, file_txt))

    def on_toggle_pick(self, event):
        item = self.tree_matches.identify_row(event.y)
        col = self.tree_matches.identify_column(event.x)
        if not item:
            return
        # only toggle when double-click anywhere (simple)
        values = list(self.tree_matches.item(item, "values"))
        # values: pick, ref_idx, score, file
        ref_idx = int(values[1])
        # toggle in self.matches
        for m in self.matches:
            if m.get("ref_idx") == ref_idx:
                m["picked"] = not m.get("picked", False)
                break
        self._refresh_matches_table()

    def on_toggle_all(self):
        if not self.matches:
            return
        any_unpicked = any(not m.get("picked", False) for m in self.matches)
        new_state = True if any_unpicked else False
        for m in self.matches:
            if m.get("file"):
                m["picked"] = new_state
        self._refresh_matches_table()

    def on_copy_selected(self):
        if not self.matches:
            messagebox.showwarning("提醒", "尚未有比對結果。請先自動比對。")
            return

        picked = [m for m in self.matches if m.get("picked") and m.get("file") and os.path.isfile(m["file"])]
        if not picked:
            messagebox.showwarning("提醒", "沒有任何已勾選且存在的檔案可複製。")
            return

        dest = filedialog.askdirectory(title="選取目的資料夾（要複製到哪裡）")
        if not dest:
            return

        copied = 0
        errors = []
        for m in picked:
            src = m["file"]
            try:
                shutil.copy2(src, dest)
                copied += 1
            except Exception as e:
                errors.append(f"{os.path.basename(src)} -> {e}")

        if errors:
            messagebox.showwarning("部分失敗", f"已複製 {copied} 個檔案。\n\n失敗清單：\n" + "\n".join(errors[:20]))
        else:
            messagebox.showinfo("完成", f"已複製 {copied} 個檔案到：\n{dest}")


if __name__ == "__main__":
    app = App()
    app.mainloop()