# =========================================================
# Ophthalmology SRMA Prototype
# - PICO-aware AI screening (PICO 可留空)
# - 自動把 PICO 轉成 keyword[tiab] OR "keyword"[MeSH Terms]
# - NOT exclude 關鍵字
# - PubMed: PMID + DOI + First author + Year + Title + Abstract
# - Title/abstract AI 初篩 + 人工覆寫
# - Full-text 決策 & 理由回填，匯出到 Excel
# =========================================================

import streamlit as st
import requests
import pandas as pd
from xml.etree import ElementTree as ET
from typing import Dict

NCBI_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
NCBI_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

st.set_page_config(page_title="Ophthalmology SRMA Prototype", layout="wide")


# ---------------------------------------------------------
# 小工具：把一個 term 變成 (term[tiab] OR "term"[MeSH Terms])
# 若使用者自己打了 [MeSH] 或 [tiab] 就不再加工，直接使用原字串
# ---------------------------------------------------------
def build_mesh_block(term: str) -> str:
    term = term.strip()
    if not term:
        return ""
    lowered = term.lower()
    # 若已經含有欄位語法，尊重使用者
    if "[" in lowered and "]" in lowered:
        return term
    # 否則幫他做 TIAB+MeSH 同步搜尋
    return f'({term}[tiab] OR "{term}"[MeSH Terms])'


# ---------------------------------------------------------
# Task 1. PICO 輸入 ＋ Query 建構（PICO 可留空，含 NOT 排除 + MeSH 同步）
# ---------------------------------------------------------
def task_1_pico_input():
    st.header("Step 1. 定義 PICO ＋ 搜尋式（欄位可留空）")

    col1, col2 = st.columns(2)
    with col1:
        P_raw = st.text_input("P (Population)", "")
        I_raw = st.text_input("I (Intervention)", "")
    with col2:
        C_raw = st.text_input("C (Comparison)", "")
        O_raw = st.text_input("O (Outcome)", "")

    # 排除條件（NOT）—這裡先用原字串，不幫你做 MeSH 展開
    exclude_terms = st.text_input(
        "排除關鍵字（NOT，例：pediatric OR animal OR case report）",
        "",
        help="這裡輸入的關鍵字會以 NOT (...) 的形式加入 PubMed 搜尋式，用來排除特定族群或研究類型。可留空。"
    )

    extra_raw = st.text_input("額外關鍵字 / 限制（例：specific device name，可留空）")
    add_rct = st.checkbox("自動加入 RCT 關鍵字", value=True)
    retmax = st.slider("抓取文獻數量 (records identified)", 10, 200, 50)

    # ====== 把有填的 PICO 都轉成「TIAB + MeSH」block ======
    include_parts = []
    if P_raw:
        include_parts.append(build_mesh_block(P_raw))
    if I_raw:
        include_parts.append(build_mesh_block(I_raw))
    if C_raw:
        include_parts.append(build_mesh_block(C_raw))
    if O_raw:
        include_parts.append(build_mesh_block(O_raw))
    if extra_raw:
        include_parts.append(build_mesh_block(extra_raw))
    if add_rct:
        include_parts.append(
            '(randomized controlled trial[tiab] OR "Randomized Controlled Trial"[Publication Type])'
        )

    base_query = " AND ".join(include_parts) if include_parts else ""

    # 再用 NOT 把排除條件接進去
    if exclude_terms.strip():
        if base_query:
            query = f"({base_query}) NOT ({exclude_terms})"
        else:
            query = f"NOT ({exclude_terms})"
    else:
        query = base_query

    st.subheader("自動產生的 PubMed Query（已含 MeSH 同步，可手動微調）")
    query = st.text_area("PubMed Query", query, height=120)

    # 把原始文字版 PICO 存進 pico dict（給 AI 和匯出用）
    pico = {"P": P_raw, "I": I_raw, "C": C_raw, "O": O_raw, "X": exclude_terms}
    return pico, query, retmax


# ---------------------------------------------------------
# Task 2. PubMed 抓取與整理（含 First author + DOI）
# ---------------------------------------------------------
def task_2_pubmed_fetch(query: str, retmax: int) -> pd.DataFrame:
    if not query.strip():
        return pd.DataFrame()

    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": retmax,
    }
    r = requests.get(NCBI_ESEARCH, params=params, timeout=30)
    r.raise_for_status()
    idlist = r.json().get("esearchresult", {}).get("idlist", [])
    if not idlist:
        return pd.DataFrame()

    fetch_params = {
        "db": "pubmed",
        "id": ",".join(idlist),
        "retmode": "xml",
    }
    r = requests.get(NCBI_EFETCH, params=fetch_params, timeout=60)
    r.raise_for_status()

    root = ET.fromstring(r.text)
    records = []

    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID", "")
        title = article.findtext(".//ArticleTitle", "")

        # 多段 AbstractText 接在一起
        ab_parts = []
        for ab in article.findall(".//AbstractText"):
            if ab.text:
                ab_parts.append(ab.text)
        abstract = " ".join(ab_parts)

        year = article.findtext(".//PubDate/Year", "")

        # First author
        first_author = ""
        author = article.find(".//AuthorList/Author[1]")
        if author is not None:
            last = author.findtext("LastName", "")
            initials = author.findtext("Initials", "")
            if last and initials:
                first_author = f"{last} {initials}"
            else:
                first_author = last or initials

        # DOI
        doi = ""
        for aid in article.findall(".//ArticleIdList/ArticleId"):
            if aid.get("IdType") == "doi" and aid.text:
                doi = aid.text.strip()
                break

        records.append(
            {
                "pmid": pmid,
                "doi": doi,
                "title": title,
                "abstract": abstract,
                "year": year,
                "first_author": first_author,
            }
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------
# Task 4. AI 初篩（依 PICO 推論 Include/Exclude/Unsure，PICO 可部分留空）
# ---------------------------------------------------------
def count_match(text_low: str, term: str) -> int:
    if not term:
        return 0
    words = [w.strip().lower() for w in term.split() if w.strip()]
    return sum(1 for w in words if w in text_low)


def ai_screen_single(row: pd.Series, pico: Dict) -> Dict:
    """
    很粗略的 rule-based 初篩：
    - 只對「實際有填的 P/I/C/O」做比對
    - 若全部 P/I/C/O 都空白，就不下判斷，標成 Unsure
    """
    title = row.get("title", "") or ""
    abstract = row.get("abstract", "") or ""
    text = (title + " " + abstract).lower()

    P = pico.get("P", "") or ""
    I = pico.get("I", "") or ""
    C = pico.get("C", "") or ""
    O = pico.get("O", "") or ""
    X = pico.get("X", "") or ""

    # 計算每一項是否命中
    p_hit = count_match(text, P)
    i_hit = count_match(text, I)
    c_hit = count_match(text, C)
    o_hit = count_match(text, O)
    x_hit = count_match(text, X) if X else 0

    # 只把「有填」的 P/I/C/O 納入分母
    non_empty_keys = [k for k, v in [("P", P), ("I", I), ("C", C), ("O", O)] if v]
    denom = len(non_empty_keys)

    hits = {
        "P": bool(p_hit),
        "I": bool(i_hit),
        "C": bool(c_hit),
        "O": bool(o_hit),
    }
    total_score = sum(1 for k in non_empty_keys if hits[k])

    # 判斷 study 是否像 trial（非常粗略）
    is_trial = any(k in text for k in ["randomized", "randomised", "trial", "prospective"])

    # 若完全沒有提供 PICO，也沒有排除條件，就不要太有意見，直接 Unsure
    if denom == 0 and not X:
        label = "Unsure"
    else:
        ratio = total_score / denom if denom > 0 else 0.0
        # Label 邏輯：若明顯 hit 到排除條件，可偏向 Exclude
        if X and x_hit:
            label = "Exclude"
        else:
            if ratio >= 0.75 and is_trial:
                label = "Include"
            elif ratio <= 0.25 and not is_trial:
                label = "Exclude"
            else:
                label = "Unsure"

    # 組中文理由
    reason_parts = []
    if P:
        reason_parts.append(
            f"P（{P}）" + ("有在標題/摘要中出現" if p_hit else "在標題/摘要中較少被明確提到")
        )
    if I:
        reason_parts.append(
            f"I（{I}）" + ("有明顯相關描述" if i_hit else "未明確符合介入描述")
        )
    if C:
        reason_parts.append(
            f"C（{C}）" + ("似乎有提到比較對象" if c_hit else "比較對象不清楚或未提及")
        )
    if O:
        reason_parts.append(
            f"O（{O}）" + ("有提到相關 outcome" if o_hit else "未明確描述該 outcome")
        )
    if X:
        reason_parts.append(
            f"排除關鍵字（{X}）" + ("似乎有在摘要中出現，偏向排除" if x_hit else "未明顯觸及排除條件")
        )

    if not non_empty_keys and not X:
        reason_parts.append("未提供明確 PICO，AI 只做非常粗略的判斷，建議人工檢視。")
    if is_trial:
        reason_parts.append("摘要中有 randomized / trial 等字樣，較像介入性研究")

    reason_text = "；".join(reason_parts) if reason_parts else "依 PICO 無法判斷，建議人工檢視全文。"

    # confidence 用「命中比例」，若 denom=0 則設為 0.0
    confidence = round((total_score / denom) if denom > 0 else 0.0, 2)

    alignment = {
        "P": hits["P"],
        "I": hits["I"],
        "C": hits["C"],
        "O": hits["O"],
    }

    return {
        "label": label,
        "reason": reason_text,
        "alignment": alignment,
        "confidence": confidence,
    }


def run_ai_for_all(df: pd.DataFrame, pico: Dict):
    if "ai_results" not in st.session_state:
        st.session_state.ai_results = {}
    if "decisions" not in st.session_state:
        st.session_state.decisions = {}
    if "fulltext_decisions" not in st.session_state:
        st.session_state.fulltext_decisions = {}
    if "fulltext_reasons" not in st.session_state:
        st.session_state.fulltext_reasons = {}

    for _, row in df.iterrows():
        pmid = row["pmid"]
        res = ai_screen_single(row, pico)
        st.session_state.ai_results[pmid] = res
        # title/abstract 預設人工決策 = AI 建議（可在 UI 再改）
        st.session_state.decisions.setdefault(pmid, res["label"])
        # full-text 預設 = Not reviewed yet
        st.session_state.fulltext_decisions.setdefault(pmid, "Not reviewed yet")
        st.session_state.fulltext_reasons.setdefault(pmid, "")


# ---------------------------------------------------------
# Task 3. 篩選 UI：顯示 AI 建議 + 理由 + 人工覆寫 + full-text 回報
# ---------------------------------------------------------
def task_3_screening_ui(df: pd.DataFrame):
    st.header("Step 3. 逐篇 screening：AI 建議 + 人工覆寫 + Full-text 回報")

    if "decisions" not in st.session_state:
        st.session_state.decisions = {}
    if "ai_results" not in st.session_state:
        st.session_state.ai_results = {}
    if "fulltext_decisions" not in st.session_state:
        st.session_state.fulltext_decisions = {}
    if "fulltext_reasons" not in st.session_state:
        st.session_state.fulltext_reasons = {}

    # 總覽表
    st.subheader("總覽表（含 AI 建議）")
    overview_rows = []
    for _, row in df.iterrows():
        pmid = row["pmid"]
        ai_res = st.session_state.ai_results.get(pmid, {})
        ai_label = ai_res.get("label", "未預測")
        ai_conf = ai_res.get("confidence", None)
        overview_rows.append(
            {
                "pmid": pmid,
                "doi": row.get("doi", ""),
                "first_author": row.get("first_author", ""),
                "year": row["year"],
                "title": row["title"][:80] + ("..." if len(row["title"]) > 80 else ""),
                "AI_label": ai_label,
                "AI_confidence": ai_conf,
                "TA_decision": st.session_state.decisions.get(pmid, ai_label),
                "FT_decision": st.session_state.fulltext_decisions.get(pmid, "Not reviewed yet"),
            }
        )
    st.dataframe(pd.DataFrame(overview_rows), use_container_width=True)

    st.markdown("---")

    # 逐篇卡片
    for _, row in df.iterrows():
        pmid = row["pmid"]
        title = row["title"]
        abstract = row["abstract"]
        year = row["year"]
        first_author = row.get("first_author", "")
        doi = row.get("doi", "")

        ai_res = st.session_state.ai_results.get(pmid, None)
        ai_label = ai_res["label"] if ai_res else "未預測"
        ai_reason = ai_res["reason"] if ai_res else "（尚未產生 AI 理由）"
        ai_conf = ai_res["confidence"] if ai_res else None

        with st.expander(f"{title}", expanded=False):
            st.markdown(
                f"**PMID:** {pmid}　　**DOI:** {doi}　　**First author:** {first_author}　　**Year:** {year}"
            )
            st.markdown(
                f"**AI 初篩建議：{ai_label}**"
                + (f"（信心度 {ai_conf}）" if ai_conf is not None else "")
            )
            st.caption(f"理由：{ai_reason}")

            st.markdown("**Abstract**")
            st.write(abstract if abstract else "_No abstract available._")

            # ---- Title / Abstract 決策 ----
            current = st.session_state.decisions.get(pmid, ai_label)
            options = ["Include", "Exclude", "Unsure"]
            if current not in options:
                current = "Unsure"
            decision = st.radio(
                "你的 title/abstract 判斷（可覆寫 AI 建議）",
                options,
                index=options.index(current),
                key=f"human_decision_{pmid}",
            )
            st.session_state.decisions[pmid] = decision

            st.markdown("---")

            # ---- Full-text 決策 & 原因／摘要（由你看完全文後回填）----
            ft_opts = [
                "Not reviewed yet",
                "Include for meta-analysis",
                "Exclude after full-text",
            ]
            current_ft = st.session_state.fulltext_decisions.get(pmid, "Not reviewed yet")
            if current_ft not in ft_opts:
                current_ft = "Not reviewed yet"

            ft_dec = st.radio(
                "Full-text decision（看完全文後再回填，現在可先跳過）",
                ft_opts,
                index=ft_opts.index(current_ft),
                key=f"ft_decision_{pmid}",
            )
            st.session_state.fulltext_decisions[pmid] = ft_dec

            ft_reason = st.text_area(
                "Full-text reason / notes（例如：樣本數太小、錯誤族群、不同介入等）",
                value=st.session_state.fulltext_reasons.get(pmid, ""),
                key=f"ft_reason_{pmid}",
            )
            st.session_state.fulltext_reasons[pmid] = ft_reason


# ---------------------------------------------------------
# Task 6. PRISMA 數字（簡化版）
# ---------------------------------------------------------
def task_6_prisma_summary(df: pd.DataFrame):
    st.header("Step 4. 簡化版 PRISMA 數字")

    identified = len(df)
    after_dedup = len(df)  # 未實作去重，先當作相同

    include_ta = sum(1 for v in st.session_state.decisions.values() if v == "Include")
    exclude_ta = sum(1 for v in st.session_state.decisions.values() if v == "Exclude")
    unsure_ta = sum(1 for v in st.session_state.decisions.values() if v == "Unsure")

    include_ft = sum(
        1
        for v in st.session_state.fulltext_decisions.values()
        if v == "Include for meta-analysis"
    )
    exclude_ft = sum(
        1
        for v in st.session_state.fulltext_decisions.values()
        if v == "Exclude after full-text"
    )

    st.write(f"Records identified（esearch 回傳）：**{identified}**")
    st.write(f"Records after duplicates removed：**{after_dedup}**")
    st.write(f"Included for full-text review（依 title/abstract）：**{include_ta}**")
    st.write(f"Unsure（待確認 full-text）：**{unsure_ta}**")
    st.write(f"Excluded at title/abstract screening：**{exclude_ta}**")
    st.write(f"Included in meta-analysis（full-text 決策）：**{include_ft}**")
    st.write(f"Excluded after full-text：**{exclude_ft}**")


# ---------------------------------------------------------
# Task 5. 匯出清單
#   A. screening summary（含 PICO/AI/TA/FT + PMID/DOI）
#   B. excluded_title_abstract（給 PRISMA 排除清單）
#   C. extraction-style table（欄位近似你貼的 Excel，含 FT 決策＆理由）
# ---------------------------------------------------------
def task_5_export_tables(df: pd.DataFrame, pico: Dict):
    st.header("Step 5. 匯出清單（含 PICO & 資料萃取表）")

    if "ai_results" not in st.session_state or not st.session_state.ai_results:
        st.info("請先在 Step 2 執行 AI 初篩後再匯出。")
        return

    # ---------- 5A. Screening summary（全部文章） ----------
    screen_rows = []
    for _, row in df.iterrows():
        pmid = row["pmid"]
        ai_res = st.session_state.ai_results.get(pmid, {})
        align = ai_res.get("alignment", {}) or {}
        ta_human = st.session_state.decisions.get(pmid, "")
        ft_dec = st.session_state.fulltext_decisions.get(pmid, "Not reviewed yet")
        ft_reason = st.session_state.fulltext_reasons.get(pmid, "")

        screen_rows.append(
            {
                "pmid": pmid,
                "doi": row.get("doi", ""),
                "first_author": row.get("first_author", ""),
                "year": row["year"],
                "title": row["title"],
                "P_query": pico.get("P", ""),
                "I_query": pico.get("I", ""),
                "C_query": pico.get("C", ""),
                "O_query": pico.get("O", ""),
                "X_query": pico.get("X", ""),
                "P_match": align.get("P", False),
                "I_match": align.get("I", False),
                "C_match": align.get("C", False),
                "O_match": align.get("O", False),
                "AI_label": ai_res.get("label", ""),
                "AI_confidence": ai_res.get("confidence", None),
                "AI_reason": ai_res.get("reason", ""),
                "TA_decision": ta_human,
                "FT_decision": ft_dec,
                "FT_reason": ft_reason,
            }
        )
    screening_df = pd.DataFrame(screen_rows)

    st.subheader("5A. screening summary 預覽（全部文章）")
    st.dataframe(screening_df, use_container_width=True)

    csv_screen = screening_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "下載 screening summary CSV（含 PICO / AI / TA / full-text 決策）",
        data=csv_screen,
        file_name="srma_screening_summary.csv",
        mime="text/csv",
    )

    # ---------- 5A-1. 只看「title/abstract 被排除」的文章清單 ----------
    excluded_df = screening_df[screening_df["TA_decision"] == "Exclude"].copy()

    if not excluded_df.empty:
        st.subheader("5A-1. Title/abstract 被排除文章清單")
        show_cols = [
            "pmid",
            "doi",
            "first_author",
            "year",
            "title",
            "AI_label",
            "AI_reason",
            "TA_decision",
        ]
        st.dataframe(excluded_df[show_cols], use_container_width=True)

        csv_excluded = excluded_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "下載『被排除文章＋AI 理由』CSV（給 PRISMA 附錄用）",
            data=csv_excluded,
            file_name="srma_excluded_title_abstract.csv",
            mime="text/csv",
        )
    else:
        st.info("目前沒有被標記為 Exclude 的文章。")

    st.markdown("---")

    # ---------- 5B. Data extraction 模板 ----------
    extract_rows = []
    for _, row in df.iterrows():
        pmid = row["pmid"]
        ta_human = st.session_state.decisions.get(pmid, "")
        ft_dec = st.session_state.fulltext_decisions.get(pmid, "Not reviewed yet")
        ft_reason = st.session_state.fulltext_reasons.get(pmid, "")
        ai_res = st.session_state.ai_results.get(pmid, {})

        extract_rows.append(
            {
                "PMID": pmid,
                "DOI": row.get("doi", ""),
                "First author": row.get("first_author", ""),
                "Year": row["year"],
                "Country": "",
                "Experimental": "",
                "Sample size (Exp)": "",
                "Controlled": "",
                "Sample size (Ctrl)": "",
                "Visual acuity testing eye condition": "",
                "Follow-up": "",
                "Outcomes": "",
                # 輔助欄位
                "Title": row["title"],
                "P_query": pico.get("P", ""),
                "I_query": pico.get("I", ""),
                "C_query": pico.get("C", ""),
                "O_query": pico.get("O", ""),
                "X_query": pico.get("X", ""),
                "AI_label": ai_res.get("label", ""),
                "TA_decision": ta_human,
                "FT_decision": ft_dec,
                "FT_reason": ft_reason,
            }
        )
    extraction_df = pd.DataFrame(extract_rows)

    st.subheader("5B. Data extraction 表格預覽（可在 Excel 補完 Experimental / Outcomes 等欄位）")
    st.dataframe(extraction_df, use_container_width=True)

    csv_extract = extraction_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "下載 data extraction CSV（欄位如 SRMA 萃取表＋full-text 決策）",
        data=csv_extract,
        file_name="srma_data_extraction_template.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------
# 主流程
# ---------------------------------------------------------
st.title("眼科 SRMA 自動化 Prototype（PICO 可留空，含 MeSH 同步＋NOT＋AI 初篩＋full-text 回報＋萃取表匯出）")

pico, query, retmax = task_1_pico_input()

if st.button("Step 2. 抓 PubMed 文獻並執行 AI 初篩"):
    with st.spinner("向 PubMed 抓取資料並進行 AI 初篩…"):
        try:
            df = task_2_pubmed_fetch(query, retmax)
            if df.empty:
                st.warning("沒有抓到任何文獻，可能是搜尋式太嚴格。")
                st.session_state.df = None
            else:
                st.session_state.df = df
                run_ai_for_all(df, pico)
                st.success(f"抓到 {len(df)} 篇文獻，已完成 AI 初步建議與理由。")
        except Exception as e:
            st.session_state.df = None
            st.error(f"抓取 PubMed 或解析時出錯：{e}")

# 有資料時顯示後續步驟
if "df" in st.session_state and isinstance(st.session_state.df, pd.DataFrame) and not st.session_state.df.empty:
    df = st.session_state.df
    task_3_screening_ui(df)
    task_6_prisma_summary(df)
    task_5_export_tables(df, pico)
else:
    st.info("請先完成 Step 2 抓取文獻。")
