# =========================================================
# Ophthalmology SRMA Prototype (Streamlit app.py)
# - PICO 可留空，支援 NOT 排除關鍵字
# - 多資料庫：PubMed / CrossRef / ClinicalTrials.gov
# - 自動把 P/I/C/O/extra 轉成 (term[tiab] OR "term"[MeSH Terms])
# - 統一欄位：pmid(主鍵) / doi / title / abstract / year / first_author / source / url
# - AI rule-based 初篩 (Include / Exclude / Unsure + 理由 + 信心度)
# - Covidence 風格 Title/Abstract screening：expander 卡片 + radio 按鈕
# - Full-text decision + reason + 可貼上全文
# - PRISMA 簡化數字（只看當次搜尋）＋ Unsure 清單
# - 匯出：screening summary / Excluded / Unsure / data extraction template
# =========================================================

import streamlit as st
import requests
import pandas as pd
from xml.etree import ElementTree as ET
from typing import Dict, List
import re
import html

# --------------------- Streamlit 設定與 CSS ---------------------
st.set_page_config(page_title="Ophthalmology SRMA Prototype", layout="wide")

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
        help="PubMed / CrossRef / ClinicalTrials.gov 各自的抓取上限。API 與記憶體實務上仍會有限制，建議超過 2000 就先 refine 搜尋式或分批處理。",
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
    st.header("Step 3. Title / Abstract screening（Covidence 風格 + AI 初篩）")

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

        exp_label = f"{title}"
        with st.expander(exp_label):
            st.markdown('<div class="card">', unsafe_allow_html=True)

            meta_line = f"PMID/ID: {pmid} &nbsp;&nbsp;&nbsp; Year: {year} &nbsp;&nbsp;&nbsp; First author: {first_author} &nbsp;&nbsp;&nbsp; Source: {source}"
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
def task_6_prisma_summary(df: pd.DataFrame):
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

    # 避免沒有 source 欄位造成 KeyError（舊 session 或自訂 df）
    if "source" in df.columns:
        st.caption("依來源分布：")
        st.write(df["source"].value_counts())
    else:
        st.caption("目前的資料表沒有 'source' 欄位（可能是舊版 session），略過來源分布。")

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
# Step 5. 匯出：screening summary / excluded / unsure / data extraction
# =========================================================
def task_5_export_tables(df: pd.DataFrame, pico: Dict):
    st.header("Step 5. 匯出清單（含 PICO & 資料萃取表）")

    if "ai_results" not in st.session_state or not st.session_state.ai_results:
        st.info("請先在 Step 2 執行 AI 初篩後再匯出。")
        return

    # 5A. screening summary
    screen_rows = []
    for _, row in df.iterrows():
        pmid = row["pmid"]
        ai_res = st.session_state.ai_results.get(pmid, {})
        align = ai_res.get("alignment", {}) or {}
        ta_human = st.session_state.decisions.get(pmid, "")
        ft_dec = st.session_state.fulltext_decisions.get(pmid, "Not reviewed yet")
        ft_reason = st.session_state.fulltext_reasons.get(pmid, "")
        ft_text = st.session_state.fulltext_content.get(pmid, "")

        screen_rows.append(
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
    screening_df = pd.DataFrame(screen_rows)

    st.subheader("5A. screening summary 預覽（全部文章）")
    st.dataframe(screening_df, use_container_width=True)

    csv_screen = screening_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "下載 screening summary CSV（含 PICO / AI / TA / full-text 決策＋全文貼上）",
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
            "下載『被排除文章＋AI 理由』CSV（給 PRISMA 附錄用）",
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

    # 5B. Data extraction 表
    extract_rows = []
    for _, row in df.iterrows():
        pmid = row["pmid"]
        ta_human = st.session_state.decisions.get(pmid, "")
        ft_dec = st.session_state.fulltext_decisions.get(pmid, "Not reviewed yet")
        ft_reason = st.session_state.fulltext_reasons.get(pmid, "")
        ft_text = st.session_state.fulltext_content.get(pmid, "")
        ai_res = st.session_state.ai_results.get(pmid, {})

        extract_rows.append(
            {
                "PMID_or_ID": pmid,
                "Source": row.get("source", ""),
                "URL": row.get("url", ""),
                "DOI": row.get("doi", ""),
                "First author": row.get("first_author", ""),
                "Year": row.get("year", ""),
                "Country": "",
                "Experimental": "",
                "Sample size (Exp)": "",
                "Controlled": "",
                "Sample size (Ctrl)": "",
                "Visual acuity testing eye condition": "",
                "Follow-up": "",
                "Outcomes": "",
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
                "FT_fulltext_snippet": ft_text,
            }
        )
    extraction_df = pd.DataFrame(extract_rows)

    st.subheader("5B. Data extraction 表格預覽")
    st.dataframe(extraction_df, use_container_width=True)

    csv_extract = extraction_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "下載 data extraction CSV（欄位如 SRMA 萃取表＋來源 URL＋full-text 決策）",
        data=csv_extract,
        file_name="srma_data_extraction_template.csv",
        mime="text/csv",
    )


# =========================================================
# 主程式流程
# =========================================================
st.title("眼科 SRMA 自動化 Prototype（多資料庫＋AI 初篩＋Covidence 風格介面）")

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
    task_6_prisma_summary(df)
    task_5_export_tables(df, pico)
else:
    st.info("請先完成 Step 2 抓取文獻。")
