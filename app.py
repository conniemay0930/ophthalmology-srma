# =========================================================
# Ophthalmology SRMA Prototype (Streamlit app.py)
# - PICO 可留空，支援 NOT 排除關鍵字
# - 多資料庫：PubMed / CrossRef / ClinicalTrials.gov
# - AI rule-based 初篩 + Covidence 風格介面
# - PRISMA 簡化數字 + 匯出
# - Step 6：互動式 Data extraction（Effect measure 下拉＋手動填 effect / CI）＋森林圖（fixed effect）
# - 入口驗證：Email + 通行碼
# =========================================================

import streamlit as st
import requests
import pandas as pd
from xml.etree import ElementTree as ET
from typing import Dict, List
import re
import html
import math

# Altair 用來畫森林圖（如果沒有安裝，也不會讓整個 app 掛掉）
try:
    import altair as alt
    HAS_ALTAIR = True
except ImportError:
    HAS_ALTAIR = False

# --------------------- Streamlit 設定 ---------------------
st.set_page_config(page_title="Ophthalmology SRMA Prototype", layout="wide")


# --------------------- CSS ---------------------
CARD_CSS = """
<style>
.card {
    border: 1px solid #dde2eb;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.8rem;
    background-color: #ffffff;
}
.card-title {
    font-weight: 700;
    font-size: 1.0rem;
    margin-bottom: 0.2rem;
}
.card-meta {
    font-size: 0.85rem;
    color: #555;
}
.badge {
    display: inline-block;
    padding: 0.1rem 0.45rem;
    border-radius: 999px;
    font-size: 0.75rem;
    margin-right: 0.3rem;
}
.badge-ai-include {
    background-color: #d1fae5;
    color: #047857;
}
.badge-ai-exclude {
    background-color: #fee2e2;
    color: #b91c1c;
}
.badge-ai-unsure {
    background-color: #e0f2fe;
    color: #0369a1;
}
.progress-box {
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    background-color: #f9fafb;
}
.progress-label {
    font-size: 0.8rem;
    color: #666;
}
.progress-value {
    font-size: 1.2rem;
    font-weight: 700;
}
</style>
"""
st.markdown(CARD_CSS, unsafe_allow_html=True)

# --------------------- NCBI API endpoint ---------------------
NCBI_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
NCBI_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


# =========================================================
# 工具：把 term 變成 (term[tiab] OR "term"[MeSH Terms])
# 若使用者自己打了 [MeSH] / [tiab] 就直接使用
# =========================================================
def build_mesh_block(term: str) -> str:
    term = term.strip()
    if not term:
        return ""
    lowered = term.lower()
    if "[" in lowered and "]" in lowered:
        return term
    return f'({term}[tiab] OR "{term}"[MeSH Terms])'


# =========================================================
# Step 1. PICO + Query 輸入 + 資料庫選擇
# =========================================================
def task_1_pico_input():
    st.header("Step 1. 定義 PICO ＋ 搜尋式（各欄位可留空）")

    col1, col2 = st.columns(2)
    with col1:
        P_raw = st.text_input("P (Population)", "")
        I_raw = st.text_input("I (Intervention)", "")
    with col2:
        C_raw = st.text_input("C (Comparison)", "")
        O_raw = st.text_input("O (Outcome)", "")

    exclude_terms = st.text_input(
        "排除關鍵字（NOT，例：pediatric OR animal OR case report）",
        "",
        help="這裡輸入的關鍵字會以 NOT (...) 的形式加入 PubMed 搜尋式，用來排除特定族群或研究類型。可留空。",
    )

    extra_raw = st.text_input("額外關鍵字 / 限制（例：specific device name，可留空）")
    add_rct = st.checkbox("自動加入 RCT 關鍵字", value=True)

    retmax = st.number_input(
        "每個資料庫抓取文獻數量上限（請小心，設太大會很慢）",
        min_value=1,
        max_value=10000,
        value=500,
        step=50,
        help="PubMed / CrossRef / ClinicalTrials.gov 各自的抓取上限。建議超過 2000 就先 refine 搜尋式或分批處理。",
    )

    st.subheader("自動產生的 PubMed Query（已含 MeSH 同步，可手動微調）")

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

    if exclude_terms.strip():
        if base_query:
            query = f"({base_query}) NOT ({exclude_terms})"
        else:
            query = f"NOT ({exclude_terms})"
    else:
        query = base_query

    query = st.text_area("PubMed Query", query, height=120)

    st.subheader("要搜尋哪些資料庫？（免費 API 範圍內）")
    c1, c2, c3 = st.columns(3)
    with c1:
        use_pubmed = st.checkbox("PubMed", value=True)
    with c2:
        use_crossref = st.checkbox("CrossRef（期刊文獻 / DOI）", value=False)
    with c3:
        use_ctgov = st.checkbox("ClinicalTrials.gov（臨床試驗註冊）", value=False)

    sources = {
        "pubmed": use_pubmed,
        "crossref": use_crossref,
        "ctgov": use_ctgov,
    }

    pico = {"P": P_raw, "I": I_raw, "C": C_raw, "O": O_raw, "X": exclude_terms}
    return pico, query, int(retmax), sources


# =========================================================
# Step 2-1. PubMed 抓取
# =========================================================
def fetch_pubmed(query: str, retmax: int) -> pd.DataFrame:
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

        ab_parts = []
        for ab in article.findall(".//AbstractText"):
            if ab.text:
                ab_parts.append(ab.text)
        abstract = " ".join(ab_parts)

        year = article.findtext(".//PubDate/Year", "")

        first_author = ""
        author = article.find(".//AuthorList/Author[1]")
        if author is not None:
            last = author.findtext("LastName", "")
            initials = author.findtext("Initials", "")
            if last and initials:
                first_author = f"{last} {initials}"
            else:
                first_author = last or initials

        doi = ""
        for aid in article.findall(".//ArticleIdList/ArticleId"):
            if aid.get("IdType") == "doi" and aid.text:
                doi = aid.text.strip()
                break

        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""

        records.append(
            {
                "pmid": pmid,
                "doi": doi,
                "title": title,
                "abstract": abstract,
                "year": year,
                "first_author": first_author,
                "source": "PubMed",
                "url": url,
            }
        )

    return pd.DataFrame(records)


# =========================================================
# Step 2-2. CrossRef 抓取
# =========================================================
def clean_crossref_abstract(text: str) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fetch_crossref(query: str, retmax: int) -> pd.DataFrame:
    if not query.strip():
        return pd.DataFrame()

    params = {
        "query": query,
        "rows": retmax,
    }
    r = requests.get("https://api.crossref.org/works", params=params, timeout=60)
    r.raise_for_status()
    items = r.json().get("message", {}).get("items", [])
    records: List[dict] = []

    for idx, it in enumerate(items):
        doi = it.get("DOI", "") or ""
        titles = it.get("title", [])
        title = " ".join(titles) if titles else ""

        abstract = clean_crossref_abstract(it.get("abstract", ""))

        year = ""
        for key in ["published-print", "published-online", "issued"]:
            block = it.get(key, {})
            parts = block.get("date-parts")
            if parts and isinstance(parts, list) and parts[0]:
                year = str(parts[0][0])
                break

        fa = ""
        authors = it.get("author", [])
        if authors:
            a0 = authors[0]
            family = a0.get("family", "")
            given = a0.get("given", "")
            if family and given:
                fa = f"{family} {given[0]}."
            else:
                fa = family or given

        if doi:
            pmid = f"CR:{doi}"
            url = f"https://doi.org/{doi}"
        else:
            pmid = f"CR:{idx}"
            url = it.get("URL", "")

        records.append(
            {
                "pmid": pmid,
                "doi": doi,
                "title": title,
                "abstract": abstract,
                "year": year,
                "first_author": fa,
                "source": "CrossRef",
                "url": url,
            }
        )

    return pd.DataFrame(records)


# =========================================================
# Step 2-3. ClinicalTrials.gov 抓取
# =========================================================
def fetch_ctgov(query: str, retmax: int) -> pd.DataFrame:
    if not query.strip():
        return pd.DataFrame()

    fields = [
        "NCTId",
        "BriefTitle",
        "BriefSummary",
        "StartDate",
        "Condition",
        "StudyType",
    ]
    params = {
        "expr": query,
        "fields": ",".join(fields),
        "min_rnk": 1,
        "max_rnk": retmax,
        "fmt": "json",
    }
    r = requests.get(
        "https://clinicaltrials.gov/api/query/study_fields",
        params=params,
        timeout=60,
    )
    r.raise_for_status()
    studies = r.json().get("StudyFieldsResponse", {}).get("StudyFields", [])

    records: List[dict] = []
    for stf in studies:
        def first_or_empty(key):
            v = stf.get(key, [])
            return v[0] if v else ""

        nct = first_or_empty("NCTId")
        title = first_or_empty("BriefTitle")
        abstract = first_or_empty("BriefSummary")
        start = first_or_empty("StartDate")
        year = start.split()[-1] if start else ""

        pmid = f"NCT:{nct}" if nct else f"NCT:{len(records)}"
        url = f"https://clinicaltrials.gov/study/{nct}" if nct else ""

        records.append(
            {
                "pmid": pmid,
                "doi": "",
                "title": title,
                "abstract": abstract,
                "year": year,
                "first_author": "",
                "source": "ClinicalTrials.gov",
                "url": url,
            }
        )

    return pd.DataFrame(records)


# =========================================================
# AI rule-based 初篩
# =========================================================
def count_match(text_low: str, term: str) -> int:
    if not term:
        return 0
    words = [w.strip().lower() for w in term.split() if w.strip()]
    return sum(1 for w in words if w in text_low)


def ai_screen_single(row: pd.Series, pico: Dict) -> Dict:
    title = row.get("title", "") or ""
    abstract = row.get("abstract", "") or ""
    text = (title + " " + abstract).lower()

    P = pico.get("P", "") or ""
    I = pico.get("I", "") or ""
    C = pico.get("C", "") or ""
    O = pico.get("O", "") or ""
    X = pico.get("X", "") or ""

    p_hit = count_match(text, P)
    i_hit = count_match(text, I)
    c_hit = count_match(text, C)
    o_hit = count_match(text, O)
    x_hit = count_match(text, X) if X else 0

    non_empty_keys = [k for k, v in [("P", P), ("I", I), ("C", C), ("O", O)] if v]
    denom = len(non_empty_keys)

    hits = {
        "P": bool(p_hit),
        "I": bool(i_hit),
        "C": bool(c_hit),
        "O": bool(o_hit),
    }
    total_score = sum(1 for k in non_empty_keys if hits[k])

    is_trial = any(k in text for k in ["randomized", "randomised", "trial", "prospective"])

    if denom == 0 and not X:
        label = "Unsure"
    else:
        ratio = total_score / denom if denom > 0 else 0.0
        if X and x_hit:
            label = "Exclude"
        else:
            if ratio >= 0.75 and is_trial:
                label = "Include"
            elif ratio <= 0.25 and not is_trial:
                label = "Exclude"
            else:
                label = "Unsure"

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
    if "fulltext_content" not in st.session_state:
        st.session_state.fulltext_content = {}

    for _, row in df.iterrows():
        pmid = row["pmid"]
        res = ai_screen_single(row, pico)
        st.session_state.ai_results[pmid] = res
        st.session_state.decisions.setdefault(pmid, res["label"])
        st.session_state.fulltext_decisions.setdefault(pmid, "Not reviewed yet")
        st.session_state.fulltext_reasons.setdefault(pmid, "")
        st.session_state.fulltext_content.setdefault(pmid, "")


# =========================================================
# Step 3. Covidence 風格 screening（expander + radio）
# =========================================================
def task_3_screening_ui(df: pd.DataFrame):
    st.header("Step 3. Title / Abstract screening")

    if "decisions" not in st.session_state:
        st.session_state.decisions = {}
    if "ai_results" not in st.session_state:
        st.session_state.ai_results = {}
    if "fulltext_decisions" not in st.session_state:
        st.session_state.fulltext_decisions = {}
    if "fulltext_reasons" not in st.session_state:
        st.session_state.fulltext_reasons = {}
    if "fulltext_content" not in st.session_state:
        st.session_state.fulltext_content = {}
    if "notes" not in st.session_state:
        st.session_state.notes = {}

    current_pmids = set(df["pmid"])

    # ---------- Team progress ----------
    decisions = st.session_state.decisions
    include_n = sum(1 for p, v in decisions.items() if p in current_pmids and v == "Include")
    exclude_n = sum(1 for p, v in decisions.items() if p in current_pmids and v == "Exclude")
    unsure_n = sum(1 for p, v in decisions.items() if p in current_pmids and v == "Unsure")
    total_n = len(df)

    st.subheader("Team progress（這一次搜尋）")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            '<div class="progress-box"><div class="progress-label">Done (Include + Exclude)</div>'
            f'<div class="progress-value">{include_n + exclude_n}</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="progress-box"><div class="progress-label">Include</div>'
            f'<div class="progress-value">{include_n}</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="progress-box"><div class="progress-label">Exclude</div>'
            f'<div class="progress-value">{exclude_n}</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            '<div class="progress-box"><div class="progress-label">Unsure / 未決定</div>'
            f'<div class="progress-value">{unsure_n + (total_n - include_n - exclude_n - unsure_n)}</div></div>',
            unsafe_allow_html=True,
        )

    # ---------- AI 建議分布 ----------
    ai_labels = [res["label"] for pid, res in st.session_state.ai_results.items() if pid in current_pmids]
    ai_inc = sum(1 for x in ai_labels if x == "Include")
    ai_exc = sum(1 for x in ai_labels if x == "Exclude")
    ai_uns = sum(1 for x in ai_labels if x == "Unsure")

    st.caption("AI 初篩建議分布：")
    a1, a2, a3 = st.columns(3)
    with a1:
        st.markdown(
            '<div class="progress-box"><div class="progress-label">AI 建議 Include</div>'
            f'<div class="progress-value">{ai_inc}</div></div>',
            unsafe_allow_html=True,
        )
    with a2:
        st.markdown(
            '<div class="progress-box"><div class="progress-label">AI 建議 Exclude</div>'
            f'<div class="progress-value">{ai_exc}</div></div>',
            unsafe_allow_html=True,
        )
    with a3:
        st.markdown(
            '<div class="progress-box"><div class="progress-label">AI 建議 Unsure</div>'
            f'<div class="progress-value">{ai_uns}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # 篩選模式
    filter_mode = st.radio(
        "要顯示哪些文獻？",
        ["全部", "尚未決定 (No votes / Unsure)", "目前標記為 Include 的"],
        horizontal=True,
    )

    # ---------- 逐篇 expander ----------
    for _, row in df.iterrows():
        pmid = row["pmid"]
        title = row["title"]
        abstract = row["abstract"]
        year = row["year"]
        first_author = row.get("first_author", "")
        doi = row.get("doi", "")
        url = row.get("url", "")
        source = row.get("source", "")

        ai_res = st.session_state.ai_results.get(pmid, None)
        ai_label = ai_res["label"] if ai_res else "未預測"
        ai_conf = ai_res["confidence"] if ai_res else None

        human_decision = st.session_state.decisions.get(pmid, ai_label)

        if filter_mode == "尚未決定 (No votes / Unsure)" and human_decision in ["Include", "Exclude"]:
            continue
        if filter_mode == "目前標記為 Include 的" and human_decision != "Include":
            continue

        with st.expander(title):
            st.markdown('<div class="card">', unsafe_allow_html=True)

            meta_line = (
                f"PMID/ID: {pmid} &nbsp;&nbsp;&nbsp; Year: {year} &nbsp;&nbsp;&nbsp; "
                f"First author: {first_author} &nbsp;&nbsp;&nbsp; Source: {source}"
            )
            if doi:
                meta_line += f" &nbsp;&nbsp;&nbsp; DOI: {doi}"
            st.markdown(meta_line, unsafe_allow_html=True)

            if url:
                st.markdown(f"[Open link]({url})", unsafe_allow_html=False)

            if ai_label == "Include":
                badge_class = "badge badge-ai-include"
            elif ai_label == "Exclude":
                badge_class = "badge badge-ai-exclude"
            else:
                badge_class = "badge badge-ai-unsure"

            ai_text = f"AI 初篩建議：{ai_label}"
            if ai_conf is not None:
                ai_text += f"（信心度 {ai_conf}）"
            st.markdown(f'<span class="{badge_class}">{ai_text}</span>', unsafe_allow_html=True)

            if ai_res:
                st.markdown(f"理由：{ai_res['reason']}")

            st.markdown("### Abstract")
            st.write(abstract if abstract else "_No abstract available._")

            st.markdown("你的 title/abstract 判斷（可覆寫 AI 建議）")
            options = ["Include", "Exclude", "Unsure"]
            if human_decision not in options:
                human_decision = "Unsure"
            idx = options.index(human_decision)
            ta_decision = st.radio(
                "",
                options,
                index=idx,
                key=f"ta_decision_{pmid}",
            )
            st.session_state.decisions[pmid] = ta_decision

            st.markdown("---")

            st.markdown("Full-text decision（看完全文後再回填，現在可先跳過）")
            ft_opts = [
                "Not reviewed yet",
                "Include for meta-analysis",
                "Exclude after full-text",
            ]
            current_ft = st.session_state.fulltext_decisions.get(pmid, "Not reviewed yet")
            if current_ft not in ft_opts:
                current_ft = "Not reviewed yet"
            ft_dec = st.radio(
                "",
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

            ft_text = st.text_area(
                "Full text / 關鍵段落（可貼上全文或主要結果，之後可給 LLM 做 full-text review）",
                value=st.session_state.fulltext_content.get(pmid, ""),
                key=f"ft_text_{pmid}",
                height=150,
            )
            st.session_state.fulltext_content[pmid] = ft_text

            st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# Step 4. PRISMA（只算本次搜尋）＋ Unsure 清單
# =========================================================
def task_4_prisma_summary(df: pd.DataFrame):
    st.header("Step 4. 簡化版 PRISMA 數字")

    current_pmids = set(df["pmid"])

    decisions = st.session_state.decisions if "decisions" in st.session_state else {}
    ft_decisions = (
        st.session_state.fulltext_decisions
        if "fulltext_decisions" in st.session_state
        else {}
    )

    include_ta_pmids = [p for p, v in decisions.items() if p in current_pmids and v == "Include"]
    exclude_ta_pmids = [p for p, v in decisions.items() if p in current_pmids and v == "Exclude"]
    unsure_ta_pmids = [p for p, v in decisions.items() if p in current_pmids and v == "Unsure"]

    include_ft_pmids = [
        p
        for p, v in ft_decisions.items()
        if p in current_pmids and v == "Include for meta-analysis"
    ]
    exclude_ft_pmids = [
        p
        for p, v in ft_decisions.items()
        if p in current_pmids and v == "Exclude after full-text"
    ]

    identified = len(df)
    after_dedup = len(df)

    st.write(f"Records identified（所有資料庫合併）：**{identified}**")
    st.write(f"Records after duplicates removed：**{after_dedup}**")
    st.write(f"Included for full-text review（依 title/abstract）：**{len(include_ta_pmids)}**")
    st.write(f"Unsure（待確認 full-text）：**{len(unsure_ta_pmids)}**")
    st.write(f"Excluded at title/abstract screening：**{len(exclude_ta_pmids)}**")
    st.write(f"Included in meta-analysis（full-text 決策）：**{len(include_ft_pmids)}**")
    st.write(f"Excluded after full-text：**{len(exclude_ft_pmids)}**")

    if "source" in df.columns:
        st.caption("依來源分布：")
        st.write(df["source"].value_counts())

    if unsure_ta_pmids:
        st.subheader("目前 Unsure（title/abstract）清單")
        unsure_df = df[df["pmid"].isin(unsure_ta_pmids)].copy()
        cols = ["pmid"]
        if "year" in df.columns:
            cols.append("year")
        if "first_author" in df.columns:
            cols.append("first_author")
        if "title" in df.columns:
            cols.append("title")
        if "source" in df.columns:
            cols.append("source")
        if "url" in df.columns:
            cols.append("url")
        unsure_df = unsure_df[cols]
        st.dataframe(unsure_df, use_container_width=True)
    else:
        st.info("目前沒有被標記為 Unsure 的文章。")


# =========================================================
# 共用：建 screening summary DataFrame（給 Step 5 / Step 6 用）
# =========================================================
def build_screening_df(df: pd.DataFrame, pico: Dict) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        pmid = row["pmid"]
        ai_res = st.session_state.ai_results.get(pmid, {})
        align = ai_res.get("alignment", {}) or {}
        ta_human = st.session_state.decisions.get(pmid, "")
        ft_dec = st.session_state.fulltext_decisions.get(pmid, "Not reviewed yet")
        ft_reason = st.session_state.fulltext_reasons.get(pmid, "")
        ft_text = st.session_state.fulltext_content.get(pmid, "")

        rows.append(
            {
                "pmid": pmid,
                "source": row.get("source", ""),
                "url": row.get("url", ""),
                "doi": row.get("doi", ""),
                "first_author": row.get("first_author", ""),
                "year": row.get("year", ""),
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
                "FT_fulltext_text": ft_text,
            }
        )
    return pd.DataFrame(rows)


# =========================================================
# Step 5. 匯出：screening summary / excluded / unsure / data extraction
# =========================================================
def task_5_export_tables(df: pd.DataFrame, pico: Dict):
    st.header("Step 5. 匯出清單（含 PICO & Data extraction template）")

    if "ai_results" not in st.session_state or not st.session_state.ai_results:
        st.info("請先在 Step 2 執行 AI 初篩後再匯出。")
        return

    screening_df = build_screening_df(df, pico)

    st.subheader("5A. screening summary 預覽（全部文章）")
    st.dataframe(screening_df, use_container_width=True)

    csv_screen = screening_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "下載 screening summary CSV",
        data=csv_screen,
        file_name="srma_screening_summary.csv",
        mime="text/csv",
    )

    # 5A-1. title/abstract 被排除清單
    excluded_df = screening_df[screening_df["TA_decision"] == "Exclude"].copy()
    if not excluded_df.empty:
        st.subheader("5A-1. Title/abstract 被排除文章清單")
        show_cols = [
            "pmid",
            "source",
            "url",
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
            "下載『被排除文章＋AI 理由』CSV",
            data=csv_excluded,
            file_name="srma_excluded_title_abstract.csv",
            mime="text/csv",
        )
    else:
        st.info("目前沒有被標記為 Exclude 的文章。")

    # 5A-2. title/abstract Unsure 清單
    unsure_df = screening_df[screening_df["TA_decision"] == "Unsure"].copy()
    if not unsure_df.empty:
        st.subheader("5A-2. Title/abstract Unsure 清單")
        show_cols_u = [
            "pmid",
            "source",
            "url",
            "doi",
            "first_author",
            "year",
            "title",
            "AI_label",
            "AI_reason",
            "TA_decision",
        ]
        st.dataframe(unsure_df[show_cols_u], use_container_width=True)

        csv_unsure = unsure_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "下載『Unsure 文章清單＋AI 理由』CSV",
            data=csv_unsure,
            file_name="srma_unsure_title_abstract.csv",
            mime="text/csv",
        )
    else:
        st.info("目前沒有被標記為 Unsure 的文章。")

    st.markdown("---")

    # 5B. Data extraction template（先匯出一份，Step 6 會在頁面上讓你填 effect / CI）
    extract_rows = []
    for _, row in screening_df.iterrows():
        extract_rows.append(
            {
                "PMID_or_ID": row["pmid"],
                "Source": row["source"],
                "URL": row["url"],
                "DOI": row["doi"],
                "First author": row["first_author"],
                "Year": row["year"],
                "Country": "",
                "Experimental": "",
                "Sample size (Exp)": "",
                "Controlled": "",
                "Sample size (Ctrl)": "",
                "Visual acuity testing eye condition": "",
                "Follow-up": "",
                "Outcomes": "",
                "Title": row["title"],
                "P_query": row["P_query"],
                "I_query": row["I_query"],
                "C_query": row["C_query"],
                "O_query": row["O_query"],
                "X_query": row["X_query"],
                "AI_label": row["AI_label"],
                "TA_decision": row["TA_decision"],
                "FT_decision": row["FT_decision"],
                "FT_reason": row["FT_reason"],
                "FT_fulltext_snippet": row["FT_fulltext_text"],
                "Effect_measure": "",
                "Effect": "",
                "Lower_CI": "",
                "Upper_CI": "",
            }
        )

    extraction_df = pd.DataFrame(extract_rows)

    st.subheader("5B. Data extraction template 預覽")
    st.dataframe(extraction_df, use_container_width=True)

    csv_extract = extraction_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "下載 data extraction CSV（含 effect 欄位）",
        data=csv_extract,
        file_name="srma_data_extraction_template.csv",
        mime="text/csv",
    )


# =========================================================
# Step 6. 互動式 Data extraction ＋ 森林圖（fixed effect）
# =========================================================
def task_6_extraction_and_forest(df: pd.DataFrame, pico: Dict):
    st.header("Step 6. Data extraction + 森林圖（固定效果）")

    if "ai_results" not in st.session_state or not st.session_state.ai_results:
        st.info("請先完成 Step 2 / Step 3 的初篩。")
        return

    if not HAS_ALTAIR:
        st.warning("目前環境沒有安裝 Altair，無法畫森林圖，但你仍可匯出 data extraction CSV。")
        return

    screening_df = build_screening_df(df, pico)

    # 候選研究：Include for meta-analysis 或 TA 已 Include
    candidates = screening_df[
        (screening_df["FT_decision"] == "Include for meta-analysis")
        | (
            (screening_df["FT_decision"] == "Not reviewed yet")
            & (screening_df["TA_decision"] == "Include")
        )
    ].copy()

    if candidates.empty:
        st.info("目前沒有任何研究被標記為『Include for meta-analysis』或 Title/Abstract 為 Include。")
        return

    st.markdown(
        """
1. 對下列表格中的研究，選擇 effect measure（RR / OR / HR / MD / SMD / Risk difference / Other）。  
2. 填入 Effect、Lower 95% CI、Upper 95% CI。  
3. 選擇要合併的 effect measure，系統會用 fixed-effect 模型計算 pooled effect 並畫森林圖。  
"""
    )

    base_cols = [
        "pmid",
        "source",
        "url",
        "doi",
        "first_author",
        "year",
        "title",
        "TA_decision",
        "FT_decision",
    ]
    base = candidates[base_cols].rename(
        columns={
            "pmid": "PMID_or_ID",
            "title": "Title",
        }
    )

    # 若之前已經在 session 裡編輯過，沿用舊值
    prev = st.session_state.get("extract_editor_df")
    if prev is not None:
        base = base.merge(
            prev[["PMID_or_ID", "Effect_measure", "Effect", "Lower_CI", "Upper_CI"]],
            on="PMID_or_ID",
            how="left",
        )

    if "Effect_measure" not in base.columns:
        base["Effect_measure"] = ""
    if "Effect" not in base.columns:
        base["Effect"] = ""
    if "Lower_CI" not in base.columns:
        base["Lower_CI"] = ""
    if "Upper_CI" not in base.columns:
        base["Upper_CI"] = ""

    st.subheader("6A. 手動填寫 effect / CI（可在頁面上直接編輯）")

    edited = st.data_editor(
        base,
        key="extract_editor",
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "PMID_or_ID": st.column_config.TextColumn("PMID/ID", disabled=True),
            "Title": st.column_config.TextColumn("Title", disabled=True, width="large"),
            "Effect_measure": st.column_config.SelectboxColumn(
                "Effect measure",
                options=["", "RR", "OR", "HR", "MD", "SMD", "Risk difference", "Other"],
                required=False,
            ),
            "Effect": st.column_config.NumberColumn("Effect", format="%.3f", required=False),
            "Lower_CI": st.column_config.NumberColumn("Lower 95% CI", format="%.3f", required=False),
            "Upper_CI": st.column_config.NumberColumn("Upper 95% CI", format="%.3f", required=False),
        },
        hide_index=True,
    )
    st.session_state.extract_editor_df = edited

    csv_cur = edited.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "下載目前的 data extraction CSV（含 effect 資料）",
        data=csv_cur,
        file_name="srma_data_extraction_with_effect.csv",
        mime="text/csv",
    )

    st.markdown("---")

    st.subheader("6B. 選擇 effect measure 並產生森林圖（fixed effect）")

    selected_measure = st.selectbox(
        "要合併哪一種 effect measure？",
        ["RR", "OR", "HR", "MD", "SMD", "Risk difference", "Other"],
    )

    sub = edited[(edited["Effect_measure"] == selected_measure)].copy()
    sub = sub.dropna(subset=["Effect", "Lower_CI", "Upper_CI"])

    if sub.empty:
        st.info("目前在此 effect measure 底下沒有填好 Effect + CI 的研究。")
        return

    # 轉成數值
    for col in ["Effect", "Lower_CI", "Upper_CI"]:
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
    sub = sub.dropna(subset=["Effect", "Lower_CI", "Upper_CI"])

    if sub.empty:
        st.error("Effect / Lower_CI / Upper_CI 中有非數值內容，清理後已無有效資料。")
        return

    eff = sub["Effect"].tolist()
    lcl = sub["Lower_CI"].tolist()
    ucl = sub["Upper_CI"].tolist()

    ratio_measures = {"RR", "OR", "HR"}

    if selected_measure in ratio_measures:
        # 在 log scale 合併，再轉回原尺度
        if any(x <= 0 for x in eff + lcl + ucl):
            st.error("RR/OR/HR 需要都是正值才能取 log，請確認資料。")
            return

        log_eff = [math.log(x) for x in eff]
        log_lcl = [math.log(x) for x in lcl]
        log_ucl = [math.log(x) for x in ucl]

        se = [(hi - lo) / (2 * 1.96) for lo, hi in zip(log_lcl, log_ucl)]
        weights = [1.0 / (s ** 2) if s > 0 else 0.0 for s in se]
        sum_w = sum(weights)

        if sum_w == 0:
            st.error("權重總和為 0，無法計算 pooled effect。")
            return

        pooled_log = sum(w * x for w, x in zip(weights, log_eff)) / sum_w
        se_pool = math.sqrt(1.0 / sum_w)
        pooled_log_lcl = pooled_log - 1.96 * se_pool
        pooled_log_ucl = pooled_log + 1.96 * se_pool

        pooled = math.exp(pooled_log)
        pooled_lcl = math.exp(pooled_log_lcl)
        pooled_ucl = math.exp(pooled_log_ucl)
    else:
        # 直接以原尺度合併
        se = [(hi - lo) / (2 * 1.96) for lo, hi in zip(lcl, ucl)]
        weights = [1.0 / (s ** 2) if s > 0 else 0.0 for s in se]
        sum_w = sum(weights)

        if sum_w == 0:
            st.error("權重總和為 0，無法計算 pooled effect。")
            return

        pooled = sum(w * x for w, x in zip(weights, eff)) / sum_w
        se_pool = math.sqrt(1.0 / sum_w)
        pooled_lcl = pooled - 1.96 * se_pool
        pooled_ucl = pooled + 1.96 * se_pool

    st.write(
        f"Fixed-effect pooled {selected_measure} = **{pooled:.3f}** "
        f"(95% CI {pooled_lcl:.3f} – {pooled_ucl:.3f})"
    )

    # 準備畫森林圖的資料
    plot_df = pd.DataFrame(
        {
            "Study": [
                f"{str(r['PMID_or_ID'])} | "
                f"{(str(r['Title'])[:37] + '...') if len(str(r['Title'])) > 40 else str(r['Title'])}"
                for _, r in sub.iterrows()
            ],
            "Effect": eff,
            "Lower_CI": lcl,
            "Upper_CI": ucl,
        }
    )

    # Altair 森林圖：水平方向 CI + 點 + pooled 垂直線
    ci_layer = (
        alt.Chart(plot_df)
        .mark_rule()
        .encode(
            y=alt.Y("Study:N", sort="-x"),
            x=alt.X("Lower_CI:Q"),
            x2="Upper_CI:Q",
        )
    )

    point_layer = (
        alt.Chart(plot_df)
        .mark_point(size=60)
        .encode(
            y=alt.Y("Study:N", sort="-x"),
            x=alt.X("Effect:Q"),
        )
    )

    pooled_df = pd.DataFrame({"x": [pooled]})
    pooled_line = (
        alt.Chart(pooled_df)
        .mark_rule(strokeDash=[4, 4], color="red")
        .encode(x="x:Q")
    )

    chart = (ci_layer + point_layer + pooled_line).properties(
        width=600,
        height=max(200, 25 * len(plot_df)),
        title=f"Forest plot (fixed-effect, {selected_measure})",
    )

    st.altair_chart(chart, use_container_width=True)

    st.caption(
        "提醒：這裡是簡化版森林圖與固定效果合併，假設各研究 effect 已經是可直接合併的尺度。"
    )


# =========================================================
# 主程式流程
# =========================================================
st.title("眼科 SRMA 自動化 Prototype")
st.caption("作者：Ya Hsin Yao（TSGH Ophthalmology Lab）")

pico, query, retmax, sources = task_1_pico_input()

if st.button("Step 2. 抓文獻並執行 AI 初篩（從勾選的資料庫）"):
    with st.spinner("向各資料庫抓取資料並進行 AI 初篩…"):
        dfs = []
        try:
            if sources.get("pubmed"):
                pub_df = fetch_pubmed(query, retmax)
                if not pub_df.empty:
                    dfs.append(pub_df)
                    st.success(f"PubMed：抓到 {len(pub_df)} 篇")
                else:
                    st.warning("PubMed 沒有抓到文獻（可能是搜尋式太嚴格）")

            if sources.get("crossref"):
                cr_df = fetch_crossref(query, retmax)
                if not cr_df.empty:
                    dfs.append(cr_df)
                    st.success(f"CrossRef：抓到 {len(cr_df)} 筆 DOI 紀錄")
                else:
                    st.warning("CrossRef 沒有抓到資料")

            if sources.get("ctgov"):
                ct_df = fetch_ctgov(query, retmax)
                if not ct_df.empty:
                    dfs.append(ct_df)
                    st.success(f"ClinicalTrials.gov：抓到 {len(ct_df)} 筆試驗")
                else:
                    st.warning("ClinicalTrials.gov 沒有抓到資料")

            if not dfs:
                st.session_state.df = None
                st.error("所有勾選的資料庫都沒有回傳文獻。")
            else:
                df_all = pd.concat(dfs, ignore_index=True)

                if "doi" in df_all.columns:
                    df_all = df_all.sort_values(["source"]).drop_duplicates(
                        subset=["doi", "title"], keep="first"
                    )
                else:
                    df_all = df_all.drop_duplicates(subset=["title"], keep="first")

                st.session_state.df = df_all.reset_index(drop=True)
                run_ai_for_all(st.session_state.df, pico)
                st.success(
                    f"合併後共有 {len(st.session_state.df)} 筆獨立文獻，已完成 AI 初步建議與理由。"
                )

        except Exception as e:
            st.session_state.df = None
            st.error(f"抓取或解析時出錯：{e}")

if (
    "df" in st.session_state
    and isinstance(st.session_state.df, pd.DataFrame)
    and not st.session_state.df.empty
):
    df = st.session_state.df
    task_3_screening_ui(df)
    task_4_prisma_summary(df)
    task_5_export_tables(df, pico)
    task_6_extraction_and_forest(df, pico)
else:
    st.info("請先完成 Step 2 抓取文獻。")
