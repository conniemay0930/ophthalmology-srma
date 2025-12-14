# =========================================================
# SRMA Web App (Streamlit) - v3 (Scope/Feasibility + Query Expander + Dynamic Extraction)
# =========================================================
# Design targets (per your lab requirements):
# 1) Minimal user input: by default only P + I are needed.
# 2) PubMed query auto-generation supports robust free-text expansion (synonyms, phrases, abbreviations)
#    WITHOUT hardcoding ophthalmology-specific examples.
# 3) Before any downstream steps: run "Existing SR/MA scan" + feasibility report (optionally LLM) to
#    refine PICO, propose inclusion/exclusion, and propose extraction schema.
# 4) Data extraction prompt explicitly instructs OCR and figure/table reading; leave blanks if not found.
# 5) Extraction table is NOT hard-coded: user can edit schema; or let AI propose schema at PICO level.
# =========================================================

from __future__ import annotations

import os
import io
import re
import math
import json
import time
import html
from typing import Dict, List, Tuple, Optional

import requests
import pandas as pd
import streamlit as st

# ---------------- Optional dependencies ----------------
try:
    from PyPDF2 import PdfReader
    HAS_PYPDF2 = True
except Exception:
    HAS_PYPDF2 = False

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

try:
    from docx import Document
    from docx.shared import Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    HAS_DOCX = True
except Exception:
    HAS_DOCX = False

# ---------------- Page ----------------
st.set_page_config(page_title="SRMA", layout="wide")

CSS = """
<style>
.card { border:1px solid #dde2eb; border-radius:10px; padding:0.9rem 1rem; margin-bottom:0.9rem; background:#fff; }
.meta { font-size:0.85rem; color:#444; }
.badge { display:inline-block; padding:0.15rem 0.55rem; border-radius:999px; font-size:0.78rem; margin-right:0.35rem; border:1px solid rgba(0,0,0,0.06); }
.badge-include { background:#d1fae5; color:#065f46; }
.badge-exclude { background:#fee2e2; color:#991b1b; }
.badge-unsure  { background:#e0f2fe; color:#075985; }
.small { font-size:0.85rem; color:#666; }
.kpi { border:1px solid #e5e7eb; border-radius:10px; padding:0.75rem 0.9rem; background:#f9fafb; }
.kpi .label { font-size:0.8rem; color:#6b7280; }
.kpi .value { font-size:1.2rem; font-weight:700; color:#111827; }
hr.soft { border:none; border-top:1px solid #eef2f7; margin:0.8rem 0; }
.codebox { background:#0b1020; color:#e6edf3; border-radius:10px; padding:0.75rem 0.85rem; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono"; font-size:0.86rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.title("SRMA")

# =========================================================
# Access gate (optional)
# =========================================================
APP_PASSWORD = os.getenv("APP_PASSWORD", "").strip()
if APP_PASSWORD:
    with st.sidebar:
        st.subheader("Access")
        pw = st.text_input("App password", type="password")
    if pw != APP_PASSWORD:
        st.warning("Password required.")
        st.stop()

# =========================================================
# Helpers
# =========================================================
def ensure_columns(df: pd.DataFrame, cols: List[str], default="") -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = default
    return df

def safe_int(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        return default

def norm_text(x: str) -> str:
    if not x:
        return ""
    x = html.unescape(str(x))
    x = re.sub(r"\s+", " ", x).strip()
    return x

def short(s: str, n=120) -> str:
    s = s or ""
    return (s[:n] + "…") if len(s) > n else s

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")

def badge_html(label: str) -> str:
    label = label or "Unsure"
    if label == "Include":
        cls = "badge badge-include"
    elif label == "Exclude":
        cls = "badge badge-exclude"
    else:
        cls = "badge badge-unsure"
    return f'<span class="{cls}">{label}</span>'

def json_from_text(s: str) -> Optional[dict]:
    if not s:
        return None
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None

def has_advanced_syntax(term: str) -> bool:
    low = (term or "").lower()
    return any(x in low for x in [" or ", " and ", " not ", "[tiab]", "[mesh", "[mh]", "(", ")", "\"", ":", "adj", "next/"])

def quote_if_needed(x: str) -> str:
    x = (x or "").strip()
    if not x:
        return ""
    if '"' in x:
        return x  # assume user knows what they're doing
    if re.search(r"\s|-|/|:", x):
        return f'"{x}"'
    return x

def split_synonyms(s: str) -> List[str]:
    """
    Accept newline-separated or comma/semicolon-separated synonyms.
    """
    if not s:
        return []
    parts = []
    for line in s.splitlines():
        line = line.strip()
        if not line:
            continue
        # allow inline separators
        for p in re.split(r"[;,]", line):
            p = p.strip()
            if p:
                parts.append(p)
    out, seen = [], set()
    for p in parts:
        p2 = p.strip()
        if p2 and p2.lower() not in seen:
            out.append(p2)
            seen.add(p2.lower())
    return out

# =========================================================
# Institutional links (OpenURL/EZproxy) - no credential storage
# =========================================================
def build_openurl(resolver_base: str, doi: str = "", pmid: str = "", title: str = "") -> str:
    resolver_base = (resolver_base or "").strip()
    if not resolver_base:
        return ""
    params = ["url_ver=Z39.88-2004", "ctx_ver=Z39.88-2004"]
    if doi:
        params.append("rft_id=" + requests.utils.quote(f"info:doi/{doi}"))
    elif pmid:
        params.append("rft_id=" + requests.utils.quote(f"info:pmid/{pmid}"))
    if title:
        params.append("rft.title=" + requests.utils.quote(title[:200]))
    joiner = "&" if "?" in resolver_base else "?"
    return resolver_base + joiner + "&".join(params)

def apply_ezproxy(ezproxy_prefix: str, url: str) -> str:
    ezproxy_prefix = (ezproxy_prefix or "").strip()
    url = (url or "").strip()
    if not ezproxy_prefix or not url:
        return url
    if "url=" in ezproxy_prefix:
        return ezproxy_prefix + requests.utils.quote(url, safe="")
    if ezproxy_prefix.endswith("/"):
        ezproxy_prefix = ezproxy_prefix[:-1]
    return ezproxy_prefix + "/login?url=" + requests.utils.quote(url, safe="")

def doi_to_url(doi: str) -> str:
    doi = (doi or "").strip()
    return f"https://doi.org/{doi}" if doi else ""

def pubmed_url(pmid: str) -> str:
    pmid = (pmid or "").strip().replace("PMID:", "").strip()
    return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""

def pmc_url(pmcid: str) -> str:
    pmcid = (pmcid or "").strip()
    if not pmcid:
        return ""
    if not pmcid.upper().startswith("PMC"):
        pmcid = "PMC" + pmcid
    return f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"

# =========================================================
# MeSH suggestion (optional)
# =========================================================
@st.cache_data(show_spinner=False, ttl=60*60)
def mesh_suggest(term: str, limit: int = 8) -> List[str]:
    term = (term or "").strip()
    if not term:
        return []
    url = "https://id.nlm.nih.gov/mesh/lookup/descriptor"
    params = {"label": term, "match": "contains", "limit": str(limit)}
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        labels = []
        for item in data:
            lab = item.get("label")
            if lab:
                labels.append(lab)
        out, seen = [], set()
        for x in labels:
            if x.lower() not in seen:
                out.append(x); seen.add(x.lower())
        return out[:limit]
    except Exception:
        return []

# =========================================================
# Query builder (key fix)
# - default: P + I (+ Extra + NOT)
# - no hardcoded domain examples
# - robust free-text expansion via synonyms boxes
# - RCT filter is OPTIONAL and off by default (you can turn it on)
# =========================================================
def build_concept_group(
    term: str,
    mesh_label: str,
    synonyms_text: str,
    field_tag: str = "tiab",
    allow_mesh: bool = True
) -> str:
    """
    Build: (term[tiab] OR "syn 1"[tiab] OR ... OR "MeSH label"[MeSH Terms])
    If user used advanced syntax, return as-is.
    """
    term = (term or "").strip()
    mesh_label = (mesh_label or "").strip()
    syns = split_synonyms(synonyms_text or "")

    if not term and not syns and not mesh_label:
        return ""

    # Advanced syntax => pass through (do not rewrite).
    if term and has_advanced_syntax(term):
        return f"({term})"

    free_terms = []
    if term:
        free_terms.append(term)
    free_terms.extend([s for s in syns if s])

    # Build free-text ORs
    free_blocks = []
    for t in free_terms:
        if not t:
            continue
        if has_advanced_syntax(t):
            free_blocks.append(t)
        else:
            free_blocks.append(f"{quote_if_needed(t)}[{field_tag}]")

    # MeSH
    mesh_block = ""
    if allow_mesh and mesh_label:
        mesh_block = f'{quote_if_needed(mesh_label)}[MeSH Terms]'

    blocks = []
    if free_blocks:
        blocks.append(" OR ".join(free_blocks))
    if mesh_block:
        blocks.append(mesh_block)

    if not blocks:
        return ""
    if len(blocks) == 1:
        return f"({blocks[0]})"
    return f"(({blocks[0]}) OR ({blocks[1]}))"

def build_pubmed_query(pico: Dict[str, str], mesh: Dict[str, str], syns: Dict[str, str],
                       include_CO: bool, include_rct_filter: bool) -> str:
    """
    Default minimal: P + I only
    """
    parts = []
    if pico.get("P","").strip() or syns.get("P","").strip() or mesh.get("P","").strip():
        parts.append(build_concept_group(pico.get("P",""), mesh.get("P",""), syns.get("P","")))
    if pico.get("I","").strip() or syns.get("I","").strip() or mesh.get("I","").strip():
        parts.append(build_concept_group(pico.get("I",""), mesh.get("I",""), syns.get("I","")))
    if include_CO:
        if pico.get("C","").strip() or syns.get("C","").strip() or mesh.get("C","").strip():
            parts.append(build_concept_group(pico.get("C",""), mesh.get("C",""), syns.get("C","")))
        if pico.get("O","").strip() or syns.get("O","").strip() or mesh.get("O","").strip():
            parts.append(build_concept_group(pico.get("O",""), mesh.get("O",""), syns.get("O","")))
    if pico.get("EXTRA","").strip() or syns.get("EXTRA","").strip() or mesh.get("EXTRA","").strip():
        # Extra is treated as free text only by default; mesh optionally
        parts.append(build_concept_group(pico.get("EXTRA",""), mesh.get("EXTRA",""), syns.get("EXTRA",""), allow_mesh=True))

    base = " AND ".join([p for p in parts if p]).strip()
    if not base:
        return ""

    if include_rct_filter:
        rct = '(randomized controlled trial[pt] OR randomized[tiab] OR randomised[tiab] OR trial[tiab] OR placebo[tiab])'
        base = f"({base}) AND ({rct})"
    else:
        base = f"({base})"

    not_block = (pico.get("X","") or "").strip()
    if not_block:
        return f"{base} NOT ({not_block})"
    return base

# =========================================================
# PubMed fetchers
# =========================================================
NCBI_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
NCBI_EFETCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

def pubmed_esearch_ids(query: str, retstart: int, retmax: int) -> Tuple[List[str], int]:
    params = {"db": "pubmed", "term": query, "retmode": "json", "retstart": retstart, "retmax": retmax}
    r = requests.get(NCBI_ESEARCH, params=params, timeout=30)
    r.raise_for_status()
    js = r.json().get("esearchresult", {})
    ids = js.get("idlist", []) or []
    count = safe_int(js.get("count", 0), 0)
    return ids, count

def pubmed_efetch_xml(pmids: List[str]) -> str:
    params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
    r = requests.get(NCBI_EFETCH, params=params, timeout=90)
    r.raise_for_status()
    return r.text

def parse_pubmed_xml(xml_text: str) -> pd.DataFrame:
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml_text)
    rows = []
    for art in root.findall(".//PubmedArticle"):
        pmid = (art.findtext(".//PMID") or "").strip()
        title = norm_text(art.findtext(".//ArticleTitle") or "")
        ab_parts = []
        for ab in art.findall(".//AbstractText"):
            if ab.text:
                ab_parts.append(norm_text(ab.text))
        abstract = " ".join([x for x in ab_parts if x]).strip()
        year = (art.findtext(".//PubDate/Year") or "").strip()
        first_author = ""
        a0 = art.find(".//AuthorList/Author[1]")
        if a0 is not None:
            last = (a0.findtext("LastName") or "").strip()
            ini  = (a0.findtext("Initials") or "").strip()
            first_author = f"{last} {ini}".strip() if (last or ini) else ""
        journal = norm_text(art.findtext(".//Journal/Title") or "")
        doi = ""; pmcid = ""
        for aid in art.findall(".//ArticleIdList/ArticleId"):
            idt = (aid.get("IdType") or "").lower()
            val = (aid.text or "").strip()
            if idt == "doi" and val and not doi:
                doi = val
            if idt == "pmc" and val and not pmcid:
                pmcid = val
        rows.append({
            "record_id": f"PMID:{pmid}" if pmid else "",
            "pmid": pmid,
            "pmcid": pmcid,
            "doi": doi,
            "title": title,
            "abstract": abstract,
            "year": year,
            "first_author": first_author,
            "journal": journal,
            "source": "PubMed",
            "url": pubmed_url(pmid),
            "doi_url": doi_to_url(doi),
            "pmc_url": pmc_url(pmcid) if pmcid else "",
        })
    df = pd.DataFrame(rows)
    return ensure_columns(df, ["record_id","pmid","pmcid","doi","title","abstract","year","first_author","journal","source","url","doi_url","pmc_url"], default="")

def fetch_pubmed(query: str, max_records: int = 0, batch_size: int = 200, polite_delay: float = 0.0) -> Tuple[pd.DataFrame,int]:
    query = (query or "").strip()
    if not query:
        return pd.DataFrame(), 0

    ids, count = pubmed_esearch_ids(query, retstart=0, retmax=min(batch_size, 500))
    all_ids = list(ids)

    target = min(count, max_records) if (max_records and max_records > 0) else count
    while len(all_ids) < target:
        retstart = len(all_ids)
        need = min(batch_size, target - len(all_ids))
        ids, _ = pubmed_esearch_ids(query, retstart=retstart, retmax=need)
        if not ids:
            break
        all_ids.extend(ids)
        if polite_delay > 0:
            time.sleep(polite_delay)

    rows = []
    for i in range(0, len(all_ids), batch_size):
        chunk = all_ids[i:i+batch_size]
        xml = pubmed_efetch_xml(chunk)
        df = parse_pubmed_xml(xml)
        if not df.empty:
            rows.append(df)
        if polite_delay > 0:
            time.sleep(polite_delay)

    if not rows:
        return pd.DataFrame(), count
    out = pd.concat(rows, ignore_index=True)
    out = ensure_columns(out, ["record_id","pmid","pmcid","doi","title","abstract","year","first_author","journal","source","url","doi_url","pmc_url"], default="")
    return out, count

# =========================================================
# Crossref fetch (optional database, free API)
# =========================================================
@st.cache_data(show_spinner=False, ttl=60*30)
def crossref_search(query: str, rows: int = 50) -> pd.DataFrame:
    query = (query or "").strip()
    if not query:
        return pd.DataFrame()
    url = "https://api.crossref.org/works"
    params = {"query.bibliographic": query, "rows": int(rows)}
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        items = r.json().get("message", {}).get("items", []) or []
        out = []
        for it in items:
            doi = (it.get("DOI") or "").strip()
            title = ""
            tlist = it.get("title") or []
            if tlist:
                title = norm_text(tlist[0])
            year = ""
            issued = it.get("issued", {}).get("date-parts", [])
            if issued and issued[0]:
                year = str(issued[0][0])
            authors = it.get("author") or []
            first_author = ""
            if authors:
                a0 = authors[0]
                first_author = (a0.get("family","") + " " + (a0.get("given","")[:1] if a0.get("given") else "")).strip()
            abstract = norm_text(it.get("abstract","") or "")
            out.append({
                "record_id": f"DOI:{doi}" if doi else f"Crossref:{len(out)}",
                "pmid": "",
                "pmcid": "",
                "doi": doi,
                "title": title,
                "abstract": abstract,
                "year": year,
                "first_author": first_author,
                "journal": norm_text((it.get("container-title") or [""])[0]),
                "source": "Crossref",
                "url": doi_to_url(doi),
                "doi_url": doi_to_url(doi),
                "pmc_url": "",
            })
        df = pd.DataFrame(out)
        return ensure_columns(df, ["record_id","pmid","pmcid","doi","title","abstract","year","first_author","journal","source","url","doi_url","pmc_url"], default="")
    except Exception:
        return pd.DataFrame()

def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["title_norm"] = df["title"].fillna("").str.lower().str.replace(r"\s+"," ", regex=True).str.strip()
    df["doi_norm"] = df["doi"].fillna("").str.lower().str.strip()
    if df["doi_norm"].astype(bool).any():
        df = df.sort_values(["doi_norm","source"]).drop_duplicates(subset=["doi_norm"], keep="first")
    df = df.sort_values(["title_norm","year","source"]).drop_duplicates(subset=["title_norm","year"], keep="first")
    df = df.drop(columns=["title_norm","doi_norm"], errors="ignore")
    return df.reset_index(drop=True)

# =========================================================
# LLM (OpenAI-compatible) - optional
# =========================================================
def llm_available() -> bool:
    api_key = (st.session_state.get("LLM_API_KEY") or "").strip()
    base = (st.session_state.get("LLM_BASE_URL") or "").strip()
    model = (st.session_state.get("LLM_MODEL") or "").strip()
    return bool(api_key and base and model)

def llm_chat(messages: List[dict], temperature: float = 0.2, timeout: int = 90) -> Optional[str]:
    base = (st.session_state.get("LLM_BASE_URL") or "").strip().rstrip("/")
    api_key = (st.session_state.get("LLM_API_KEY") or "").strip()
    model = (st.session_state.get("LLM_MODEL") or "").strip()
    if not (base and api_key and model):
        return None
    url = base + "/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": float(temperature)}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        r.raise_for_status()
        js = r.json()
        return js["choices"][0]["message"]["content"]
    except Exception as e:
        st.warning(f"LLM 呼叫失敗（將改用規則法/或跳過）：{e}")
        return None

# =========================================================
# Rule-based TA screening (high recall)
# =========================================================
def _count_hits(text_low: str, term: str) -> int:
    term = (term or "").strip()
    if not term:
        return 0
    toks = [t.strip().lower() for t in re.split(r"[,\s]+", term) if t.strip()]
    return sum(1 for t in toks if t and t in text_low)

def ai_screen_rule_based(row: pd.Series, pico: Dict[str,str]) -> Dict:
    title = row.get("title","") or ""
    abstract = row.get("abstract","") or ""
    text = (title + " " + abstract).lower()

    P = pico.get("P","") or ""
    I = pico.get("I","") or ""
    X = pico.get("X","") or ""

    p_hit = _count_hits(text, P)
    i_hit = _count_hits(text, I)
    x_hit = _count_hits(text, X) if X else 0

    is_trial = any(w in text for w in ["randomized", "randomised", "trial", "controlled", "prospective", "double-blind", "single-blind"])
    is_basic = any(w in text for w in ["in vitro", "cell line", "mouse", "mice", "rat", "animal model"])
    is_case_report = any(w in text for w in ["case report", "case series"])

    if X.strip() and x_hit:
        return {"label": "Exclude", "reason": "NOT keyword hit", "confidence": 0.9}

    if is_basic and not is_trial:
        return {"label": "Exclude", "reason": "Basic research likely (and not trial-like)", "confidence": 0.8}

    if is_case_report and not is_trial:
        return {"label": "Exclude", "reason": "Case report/series likely (and not trial-like)", "confidence": 0.8}

    if (P.strip() and I.strip() and p_hit and i_hit) and (is_trial or len(text) > 0):
        return {"label": "Include", "reason": "P+I hit; candidate for full-text", "confidence": 0.75}

    if is_trial and ((P.strip() and p_hit) or (I.strip() and i_hit)):
        return {"label": "Include", "reason": "Trial-like + P or I hit; keep for FT", "confidence": 0.7}

    return {"label": "Unsure", "reason": "Insufficient TA evidence; keep as unsure", "confidence": 0.4}

# =========================================================
# Feasibility: Existing SR/MA scan + LLM feasibility report
# =========================================================
def srma_scan_pubmed(pubmed_query: str, top_n: int = 25) -> Dict:
    q = (pubmed_query or "").strip()
    if not q:
        return {"summary":"(未提供 query)", "hits": pd.DataFrame()}
    sr_filter = '(systematic review[pt] OR meta-analysis[pt] OR "systematic review"[tiab] OR "meta analysis"[tiab] OR "network meta-analysis"[tiab])'
    q_sr = f"({q}) AND ({sr_filter})"
    try:
        ids, count = pubmed_esearch_ids(q_sr, 0, min(top_n, 50))
        df_hits = pd.DataFrame()
        if ids:
            xml = pubmed_efetch_xml(ids)
            df_hits = parse_pubmed_xml(xml)[["pmid","doi","pmcid","title","year","first_author","journal","url","doi_url","pmc_url"]]
        summary = f"PubMed 既有 SR/MA 掃描：count≈{count}；列出前 {min(len(ids), top_n)} 篇。"
        return {"summary": summary, "hits": df_hits}
    except Exception as e:
        return {"summary": f"SR/MA 掃描失敗：{e}", "hits": pd.DataFrame()}

def feasibility_llm(pico: Dict[str,str], goal_mode: str, clinical_context: str, srma_hits: pd.DataFrame, pubmed_query: str) -> Dict:
    sr_list = []
    if isinstance(srma_hits, pd.DataFrame) and not srma_hits.empty:
        for _, r in srma_hits.head(20).iterrows():
            sr_list.append({"pmid": r.get("pmid",""), "year": r.get("year",""), "title": r.get("title","")})

    sys = "You are a senior SR/MA methodologist. Output ONLY valid JSON."
    user = {
        "task": "Feasibility report BEFORE screening",
        "goal_mode": goal_mode,  # Gap-fill fast vs rigorous scope
        "clinical_context": clinical_context,
        "input_pico": pico,
        "current_pubmed_query": pubmed_query,
        "existing_srma_hits": sr_list,
        "required_outputs": {
            "refined_topic_options": "3 options; rank by feasibility and novelty",
            "recommended_pico": "one best PICO with justification",
            "inclusion_criteria": "explicit rule list; separate mandatory vs optional",
            "exclusion_criteria": "explicit rule list",
            "recommended_search_expansion": "synonyms/phrases for each concept; keep general",
            "recommended_extraction_schema": "base fields + outcomes; ensure typical primary/secondary RCT outcomes considered",
            "notes_on_human_judgement": "where minimal human judgement is still needed"
        },
        "constraints": [
            "Do not assume ophthalmology; keep suggestions topic-agnostic unless user context implies otherwise.",
            "If goal_mode indicates 'Fast/feasible', prioritize narrower topic and higher publishability.",
            "If goal_mode indicates 'Rigorous', prioritize comprehensive scope; be transparent about workload."
        ]
    }
    messages = [{"role":"system","content":sys},{"role":"user","content":json.dumps(user, ensure_ascii=False)}]
    txt = llm_chat(messages, temperature=0.2, timeout=140)
    js = json_from_text(txt or "")
    if not js:
        return {"error":"LLM 回傳無法解析 JSON", "raw": txt or ""}
    return js

# =========================================================
# Extraction prompt (OCR + table/figure)
# =========================================================
def build_extraction_prompt(schema: dict, pico: dict, criteria_text: str, goal_mode: str) -> str:
    base_cols = schema.get("base_cols", [])
    outcomes = schema.get("outcomes", [])
    prompt = f"""
You are performing systematic review full-text data extraction.

Goal mode: {goal_mode}

OCR / Figure / Table instructions (critical):
- If text looks missing/garbled OR PDF is scanned: explicitly output "OCR REQUIRED" and state what to OCR.
- When extracting outcomes, prioritize: tables, figure legends, supplementary appendix.
- If a value is taken from a figure, state "from Figure X (approx.)" and capture axis units/timepoint.
- Never fabricate numbers. If not found, leave the field empty and list where it might be (e.g., Table 2, Figure 3, appendix).

Inclusion/Exclusion criteria (traceable; follow as executable rules):
{criteria_text if criteria_text else "(No formal criteria provided. Use P+I as primary guidance; be conservative and mark Unsure if unclear.)"}

Topic PICO:
P={pico.get("P","")}
I={pico.get("I","")}
C={pico.get("C","")}
O={pico.get("O","")}
NOT={pico.get("X","")}

Extraction schema (do not change keys; leave blank if not reported):
Base fields:
{json.dumps(base_cols, ensure_ascii=False)}

Outcome fields:
{json.dumps(outcomes, ensure_ascii=False)}

Effect estimates (if available):
- effect_measure: OR/RR/HR/MD/SMD/RD/Other
- effect, lower_CI, upper_CI, timepoint, unit
If multiple outcomes/timepoints exist, prioritize the primary outcome and main follow-up timepoint; note others in missing_notes.

Output JSON ONLY with keys:
- fulltext_decision: "Include for meta-analysis" / "Exclude after full-text" / "Not reviewed yet"
- fulltext_reason: short reason
- extracted_fields: object mapping field -> value (string)
- meta: object {effect_measure,effect,lower_CI,upper_CI,timepoint,unit}
- missing_notes: list of strings describing missing items + where to look (table/figure/page/section)
"""
    return prompt.strip()

# =========================================================
# Session state
# =========================================================
def init_state():
    ss = st.session_state
    ss.setdefault("GOAL_MODE", "Rigorous / comprehensive")
    ss.setdefault("CTX", "")
    ss.setdefault("P", "")
    ss.setdefault("I", "")
    ss.setdefault("C", "")
    ss.setdefault("O", "")
    ss.setdefault("X", "")
    ss.setdefault("EXTRA", "")
    ss.setdefault("STRICT_CO", False)
    ss.setdefault("INCLUDE_RCT_FILTER", False)
    ss.setdefault("AUTO_FOLLOW_QUERY", True)

    ss.setdefault("P_MESH","")
    ss.setdefault("I_MESH","")
    ss.setdefault("C_MESH","")
    ss.setdefault("O_MESH","")
    ss.setdefault("EXTRA_MESH","")

    ss.setdefault("P_SYNS","")
    ss.setdefault("I_SYNS","")
    ss.setdefault("C_SYNS","")
    ss.setdefault("O_SYNS","")
    ss.setdefault("EXTRA_SYNS","")

    ss.setdefault("PUBMED_QUERY","")
    ss.setdefault("criteria_text","")

    ss.setdefault("feasibility", {"srma_summary":"", "srma_hits": pd.DataFrame(), "llm_json": None})

    ss.setdefault("records_df", pd.DataFrame())
    ss.setdefault("pubmed_total_count", 0)
    ss.setdefault("ai_ta_df", pd.DataFrame())
    ss.setdefault("ta_final", {})
    ss.setdefault("ft_decisions", {})
    ss.setdefault("ft_reasons", {})
    ss.setdefault("ft_text", {})
    ss.setdefault("ft_note", {})
    ss.setdefault("rob2", {})
    ss.setdefault("extract_wide_df", pd.DataFrame())
init_state()

# =========================================================
# Sidebar settings
# =========================================================
with st.sidebar:
    st.subheader("設定")

    resolver_base = st.text_input(
        "OpenURL / Link resolver base（例：https://resolver.xxx.edu/openurl?）",
        value=st.session_state.get("RESOLVER_BASE",""),
        help="不存帳密；點全文連結後由使用者自行登入下載。"
    )
    ezproxy_prefix = st.text_input(
        "EZproxy prefix（可選；例：https://ezproxy.xxx.edu/login?url=）",
        value=st.session_state.get("EZPROXY",""),
        help="若有 EZproxy，可將外部連結轉為 EZproxy 版本。"
    )
    st.session_state["RESOLVER_BASE"] = resolver_base
    st.session_state["EZPROXY"] = ezproxy_prefix

    st.markdown("---")
    st.markdown("**LLM（可選；OpenAI-compatible）**")
    st.session_state["LLM_BASE_URL"] = st.text_input("Base URL", value=st.session_state.get("LLM_BASE_URL",""))
    st.session_state["LLM_API_KEY"]  = st.text_input("API Key", value=st.session_state.get("LLM_API_KEY",""), type="password")
    st.session_state["LLM_MODEL"]    = st.text_input("Model", value=st.session_state.get("LLM_MODEL",""))

    if llm_available():
        st.success("LLM 已設定：可用於可行性報告/criteria/schema/更準確抽取")
    else:
        st.info("未設定 LLM：仍可做檢索與規則法粗篩；可行性/抽取會較弱")

    st.markdown("---")
    if not HAS_DOCX:
        st.warning("未安裝 python-docx：Word 匯出停用。")
    if not HAS_MPL:
        st.warning("未安裝 matplotlib：森林圖停用。")
    if not HAS_PYPDF2:
        st.warning("未安裝 PyPDF2：PDF 抽字停用（可改貼全文）。")

# =========================================================
# Step 1: PICO + Search Strategy (minimal input)
# =========================================================
st.header("Step 1. 定義 PICO + 搜尋式（預設只需 P + I）")

col1, col2 = st.columns([1,1])
with col1:
    goal_mode = st.selectbox("Goal mode", options=["Fast / feasible (gap-fill)", "Rigorous / comprehensive"], index=1)
with col2:
    clinical_context = st.text_area("Clinical context（可留空；用於可行性判斷，不會自動進搜尋式）", value=st.session_state.get("CTX",""), height=70)

st.session_state["GOAL_MODE"] = goal_mode
st.session_state["CTX"] = clinical_context

colP, colC = st.columns([1,1])
with colP:
    P = st.text_input("P (Population / Topic) 〔建議填〕", value=st.session_state.get("P",""))
with colC:
    C = st.text_input("C (Comparison) 〔可留白〕", value=st.session_state.get("C",""))

colI, colO = st.columns([1,1])
with colI:
    I = st.text_input("I (Intervention / Exposure) 〔建議填〕", value=st.session_state.get("I",""))
with colO:
    O = st.text_input("O (Outcome) 〔可留白〕", value=st.session_state.get("O",""))

exclude_not = st.text_input("排除關鍵字（NOT；例：pediatric OR animal OR case report）", value=st.session_state.get("X",""))
extra_kw = st.text_input("額外關鍵字/限制（可留白；例：device name / setting）", value=st.session_state.get("EXTRA",""))

st.session_state.update({"P":P,"I":I,"C":C,"O":O,"X":exclude_not,"EXTRA":extra_kw})

with st.expander("（關鍵）Free-text 擴充（synonyms / phrases / abbreviations；每行或逗號分隔）", expanded=True):
    st.caption("這裡用來補足『同一個 P/I 但你手動查得到、網站查不到』的主要差異來源。不要寫死範例，請依你的題目貼同義詞。")
    sP = st.text_area("P 的 synonyms（可留白）", value=st.session_state.get("P_SYNS",""), height=90)
    sI = st.text_area("I 的 synonyms（可留白）", value=st.session_state.get("I_SYNS",""), height=90)
    sC = st.text_area("C 的 synonyms（可留白）", value=st.session_state.get("C_SYNS",""), height=90)
    sO = st.text_area("O 的 synonyms（可留白）", value=st.session_state.get("O_SYNS",""), height=90)
    sX = st.text_area("Extra 的 synonyms（可留白）", value=st.session_state.get("EXTRA_SYNS",""), height=70)
    st.session_state.update({"P_SYNS":sP,"I_SYNS":sI,"C_SYNS":sC,"O_SYNS":sO,"EXTRA_SYNS":sX})

with st.expander("MeSH term 同步（可選，查不到很正常）", expanded=False):
    def mesh_picker(label: str, term: str, key_prefix: str) -> str:
        term = (term or "").strip()
        if not term:
            st.caption(f"{label}: (空白)")
            st.session_state[f"{key_prefix}_MESH"] = ""
            return ""
        sug = mesh_suggest(term)
        default = st.session_state.get(f"{key_prefix}_MESH","")
        choice = st.selectbox(
            f"{label} 的 MeSH 建議（可留空不用）",
            options=[""] + sug,
            index=([""]+sug).index(default) if default in ([""]+sug) else 0,
            key=f"{key_prefix}_mesh_select"
        )
        st.session_state[f"{key_prefix}_MESH"] = choice
        if sug:
            st.caption("建議：" + " / ".join(sug[:8]))
        else:
            st.caption("查不到建議（或 API 暫時不可用）。")
        return choice

    mesh_picker("P", P, "P")
    mesh_picker("I", I, "I")
    mesh_picker("C", C, "C")
    mesh_picker("O", O, "O")
    mesh_picker("Extra", extra_kw, "EXTRA")

strict_include_CO = st.checkbox("嚴格把 C / O 納入檢索（會降低召回率；預設關閉）", value=st.session_state.get("STRICT_CO", False))
include_rct_filter = st.checkbox("在檢索式階段就套 RCT filter（可能大幅漏抓；預設關閉，建議後面再篩）", value=st.session_state.get("INCLUDE_RCT_FILTER", False))
st.session_state["STRICT_CO"] = strict_include_CO
st.session_state["INCLUDE_RCT_FILTER"] = include_rct_filter

pico_now = {"P":P,"I":I,"C":C,"O":O,"X":exclude_not,"EXTRA":extra_kw}
mesh_now = {"P":st.session_state.get("P_MESH",""),"I":st.session_state.get("I_MESH",""),
            "C":st.session_state.get("C_MESH",""),"O":st.session_state.get("O_MESH",""),
            "EXTRA":st.session_state.get("EXTRA_MESH","")}
syns_now = {"P":st.session_state.get("P_SYNS",""),"I":st.session_state.get("I_SYNS",""),
            "C":st.session_state.get("C_SYNS",""),"O":st.session_state.get("O_SYNS",""),
            "EXTRA":st.session_state.get("EXTRA_SYNS","")}

auto_q = build_pubmed_query(pico_now, mesh_now, syns_now, include_CO=strict_include_CO, include_rct_filter=include_rct_filter)

auto_follow = st.checkbox("PubMed 搜尋式自動跟隨（會覆蓋手動修改）", value=st.session_state.get("AUTO_FOLLOW_QUERY", True))
st.session_state["AUTO_FOLLOW_QUERY"] = auto_follow

if auto_follow and auto_q:
    st.session_state["PUBMED_QUERY"] = auto_q

colQ1, colQ2 = st.columns([1,3])
with colQ1:
    if st.button("套用：重建自動搜尋式"):
        if auto_q:
            st.session_state["PUBMED_QUERY"] = auto_q
        else:
            st.warning("無法產生搜尋式：請至少填 P 或 I（或 synonyms）。")
with colQ2:
    pubmed_query = st.text_area("自動產生的 PubMed Query（已含 MeSH/同義詞，可手動微調）", value=st.session_state.get("PUBMED_QUERY",""), height=120)
    st.session_state["PUBMED_QUERY"] = pubmed_query

# =========================================================
# Step 1A: BEFORE ALL - Feasibility report + Existing SR/MA scan (restored)
# =========================================================
st.subheader("Step 1A.（開始所有步驟前）搜尋既有主題 + 可行性報告 + PICO/criteria/schema 建議")

colF1, colF2, colF3 = st.columns([1,1,1])
with colF1:
    if st.button("1A-1 既有 SR/MA 掃描（PubMed）"):
        if not st.session_state.get("PUBMED_QUERY","").strip():
            st.error("請先產生/填入 PubMed query。")
        else:
            with st.spinner("掃描中…"):
                fb = srma_scan_pubmed(st.session_state["PUBMED_QUERY"], top_n=25)
                st.session_state["feasibility"]["srma_summary"] = fb["summary"]
                st.session_state["feasibility"]["srma_hits"] = fb["hits"]
with colF2:
    if st.button("1A-2 產出可行性報告（LLM）"):
        if not llm_available():
            st.error("未設定 LLM，無法產出完整可行性報告。")
        elif not st.session_state.get("PUBMED_QUERY","").strip():
            st.error("請先產生/填入 PubMed query。")
        else:
            with st.spinner("LLM 分析中…"):
                hits = st.session_state["feasibility"].get("srma_hits", pd.DataFrame())
                js = feasibility_llm(pico_now, goal_mode, clinical_context, hits, st.session_state["PUBMED_QUERY"])
                st.session_state["feasibility"]["llm_json"] = js
with colF3:
    st.caption("建議流程：先掃描既有 SR/MA → 再出可行性報告（可能需要少量人工判斷，決定是否縮小/轉向題目）。")

fb = st.session_state.get("feasibility", {})
if fb.get("srma_summary"):
    st.info(fb["srma_summary"])
    hits = fb.get("srma_hits", pd.DataFrame())
    if isinstance(hits, pd.DataFrame) and not hits.empty:
        st.dataframe(hits, use_container_width=True)
        st.download_button("下載 SR/MA 命中清單（CSV）", data=to_csv_bytes(hits), file_name="srma_hits.csv", mime="text/csv")

js = fb.get("llm_json")
if js:
    if js.get("error"):
        st.error(js.get("error"))
        st.code(js.get("raw",""), language="text")
    else:
        st.markdown("**LLM 可行性報告（JSON）**")
        st.code(json.dumps(js, ensure_ascii=False, indent=2), language="json")

        st.markdown("**在此層級就定義 inclusion/exclusion（可套用）**")
        if st.button("套用：把 LLM criteria 寫入 criteria_text"):
            inc = js.get("inclusion_criteria") or []
            exc = js.get("exclusion_criteria") or []
            lines = ["Inclusion criteria:"]
            if isinstance(inc, list):
                for r in inc:
                    if isinstance(r, dict):
                        lines.append(f"- [{r.get('id','')}] {r.get('rule','')}")
                    else:
                        lines.append(f"- {r}")
            else:
                lines.append(str(inc))
            lines.append("")
            lines.append("Exclusion criteria:")
            if isinstance(exc, list):
                for r in exc:
                    if isinstance(r, dict):
                        lines.append(f"- [{r.get('id','')}] {r.get('rule','')}")
                    else:
                        lines.append(f"- {r}")
            else:
                lines.append(str(exc))
            st.session_state["criteria_text"] = "\n".join(lines).strip()
            st.success("已套用 criteria。")

        st.markdown("**在此層級就規劃 extraction sheet（可套用）**")
        if st.button("套用：把 LLM schema 寫入 extraction schema"):
            es = js.get("recommended_extraction_schema") or {}
            base_cols2 = es.get("base_cols") or []
            outcomes2 = es.get("outcomes") or []
            if base_cols2:
                st.session_state["BASECOLS"] = "\n".join(base_cols2)
            if outcomes2:
                st.session_state["OUTCOME_LINES"] = "\n".join(outcomes2)
            st.success("已套用 schema（請往下檢查/微調）。")

st.markdown("---")

# =========================================================
# Step 1B: extraction schema (NOT hard-coded)
# =========================================================
st.subheader("Step 1B. extraction schema（欄位不寫死；可自訂；可由 AI 建議）")

default_outcomes = st.session_state.get("OUTCOME_LINES", "Primary outcome\nSecondary outcome 1\nSecondary outcome 2")
outcome_lines = st.text_area("Outcome / 欄位名稱（每行一個，可自訂）", value=default_outcomes, height=120)
st.session_state["OUTCOME_LINES"] = outcome_lines

default_base_cols = st.session_state.get(
    "BASECOLS",
    "\n".join([
        "First author","Year","Country",
        "Intervention","Sample size (Intervention)",
        "Comparator","Sample size (Comparator)",
        "Follow-up","Key outcomes",
        "Notes (table/figure/page)","Fulltext availability / note",
    ])
)
base_cols_text = st.text_area("基本欄位（每行一個）", value=default_base_cols, height=160)
st.session_state["BASECOLS"] = base_cols_text

schema = {
    "base_cols": [x.strip() for x in (base_cols_text or "").splitlines() if x.strip()],
    "outcomes": [x.strip() for x in (outcome_lines or "").splitlines() if x.strip()],
}

st.markdown("#### criteria（可先手寫；或用 Step 1A LLM 套用）")
criteria_text = st.text_area("Inclusion/Exclusion criteria", value=st.session_state.get("criteria_text",""), height=220)
st.session_state["criteria_text"] = criteria_text

st.markdown("---")

# =========================================================
# Step 2: Fetch records + run AI TA screening
# =========================================================
st.header("Step 2. 抓文獻並執行 AI 初篩（從勾選的資料庫）")

use_pubmed = st.checkbox("PubMed", value=True)
use_crossref = st.checkbox("CrossRef（期刊文獻/DOI；可補漏）", value=False)

max_records = st.number_input("每個資料庫抓取上限（0=全部；太大會慢）", min_value=0, max_value=2000000, value=1000, step=200)
polite_delay = st.slider("（可選）API 友善延遲（秒）", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

ta_engine = st.selectbox("Title/Abstract 初篩引擎", options=["Auto (LLM if available else rule)", "Rule-based", "LLM"], index=0)

def run_ta_rule(df_chunk: pd.DataFrame, pico_: dict) -> pd.DataFrame:
    out = []
    for _, r in df_chunk.iterrows():
        rb = ai_screen_rule_based(r, pico_)
        out.append({"record_id": r["record_id"], "AI_label": rb["label"], "AI_confidence": rb["confidence"], "AI_reason": rb["reason"]})
    return pd.DataFrame(out)

def run_ta_llm(df_chunk: pd.DataFrame, pico_: dict, criteria_text_: str) -> pd.DataFrame:
    out = []
    sys = "You are an SR/MA screening assistant. Output ONLY valid JSON."
    for _, r in df_chunk.iterrows():
        user = {
            "task":"Title/Abstract screening",
            "pico": pico_,
            "criteria_text": criteria_text_,
            "record": {"title": r.get("title",""), "abstract": r.get("abstract",""), "year": r.get("year",""), "source": r.get("source","")},
            "output_schema": {"label":"Include/Exclude/Unsure", "confidence":"0..1", "reason":"short"}
        }
        txt = llm_chat([{"role":"system","content":sys},{"role":"user","content":json.dumps(user, ensure_ascii=False)}], temperature=0.1, timeout=90)
        js = json_from_text(txt or "")
        if not js:
            rb = ai_screen_rule_based(r, pico_)
            js = {"label": rb["label"], "confidence": rb["confidence"], "reason": rb["reason"]}
        out.append({"record_id": r["record_id"], "AI_label": js.get("label","Unsure"), "AI_confidence": js.get("confidence",0.0), "AI_reason": js.get("reason","")})
    return pd.DataFrame(out)

def fetch_all_selected() -> Tuple[pd.DataFrame, Dict[str,int]]:
    q = (st.session_state.get("PUBMED_QUERY","") or "").strip()
    if not q:
        st.error("PubMed query 為空，請先在 Step 1 產生/填入。")
        return pd.DataFrame(), {}

    dfs = []
    counts = {}

    if use_pubmed:
        dfp, cnt = fetch_pubmed(q, max_records=int(max_records), batch_size=200, polite_delay=float(polite_delay))
        counts["PubMed"] = len(dfp)
        st.session_state["pubmed_total_count"] = cnt
        if not dfp.empty:
            dfs.append(dfp)

    if use_crossref:
        dfc = crossref_search(q, rows=int(min(max_records, 200)))
        counts["Crossref"] = len(dfc)
        if not dfc.empty:
            dfs.append(dfc)

    if not dfs:
        return pd.DataFrame(), counts

    df_all = pd.concat(dfs, ignore_index=True)
    df_all = ensure_columns(df_all, ["record_id","pmid","pmcid","doi","title","abstract","year",
                                     "first_author","journal","source","url","doi_url","pmc_url"], default="")
    df_dedup = deduplicate(df_all)
    return df_dedup, counts

def init_record_states(df: pd.DataFrame):
    for rid in df["record_id"].tolist():
        st.session_state["ta_final"].setdefault(rid, "Unsure")
        st.session_state["ft_decisions"].setdefault(rid, "Not reviewed yet")
        st.session_state["ft_reasons"].setdefault(rid, "")
        st.session_state["ft_text"].setdefault(rid, "")
        st.session_state["ft_note"].setdefault(rid, "")
        st.session_state["rob2"].setdefault(rid, {})

if st.button("Step 2. 抓文獻並執行 AI 初篩"):
    with st.spinner("抓取中…"):
        df_dedup, counts = fetch_all_selected()
    if df_dedup.empty:
        st.error("沒有抓到資料。")
    else:
        st.session_state["records_df"] = df_dedup
        init_record_states(df_dedup)
        st.success(f"合併去重後共有 {len(df_dedup)} 篇。各庫：{counts}")

        # Run TA screening immediately (as your original UX)
        pico_basic = {"P":P,"I":I,"C":C,"O":O,"X":exclude_not}
        engine_use = ta_engine
        use_llm = False
        if engine_use.startswith("Auto"):
            use_llm = llm_available()
        elif engine_use == "LLM":
            use_llm = True
        else:
            use_llm = False

        with st.spinner("AI Title/Abstract 初篩中…"):
            if use_llm and llm_available():
                df_ai = run_ta_llm(df_dedup, pico_basic, st.session_state.get("criteria_text",""))
            else:
                df_ai = run_ta_rule(df_dedup, pico_basic)

        st.session_state["ai_ta_df"] = df_ai
        for _, rr in df_ai.iterrows():
            st.session_state["ta_final"][rr["record_id"]] = rr["AI_label"]
        st.success("已完成 AI 初篩。")

df_records = st.session_state.get("records_df", pd.DataFrame())
ai_ta_df = st.session_state.get("ai_ta_df", pd.DataFrame())

if df_records.empty:
    st.stop()

# =========================================================
# Step 3: TA screening results + FT links
# =========================================================
st.header("Step 3. Title/Abstract screening（含理由；不需像 Covidence 人工逐篇勾）")

view_df = df_records.merge(ai_ta_df, on="record_id", how="left")
view_df = ensure_columns(view_df, ["AI_label","AI_reason","AI_confidence"], default="")

ta_vals = [st.session_state["ta_final"].get(rid, "Unsure") for rid in df_records["record_id"].tolist()]
k_include = sum(1 for x in ta_vals if x == "Include")
k_exclude = sum(1 for x in ta_vals if x == "Exclude")
k_unsure  = sum(1 for x in ta_vals if x == "Unsure")

c1,c2,c3,c4 = st.columns(4)
with c1: st.markdown(f'<div class="kpi"><div class="label">Total</div><div class="value">{len(df_records)}</div></div>', unsafe_allow_html=True)
with c2: st.markdown(f'<div class="kpi"><div class="label">Include</div><div class="value">{k_include}</div></div>', unsafe_allow_html=True)
with c3: st.markdown(f'<div class="kpi"><div class="label">Exclude</div><div class="value">{k_exclude}</div></div>', unsafe_allow_html=True)
with c4: st.markdown(f'<div class="kpi"><div class="label">Unsure</div><div class="value">{k_unsure}</div></div>', unsafe_allow_html=True)

filter_mode = st.radio("檢視清單", ["只看 Unsure", "只看 Include", "只看 Exclude", "全部"], horizontal=True, index=0)

def want(dec: str) -> bool:
    if filter_mode == "全部": return True
    if filter_mode == "只看 Unsure": return dec == "Unsure"
    if filter_mode == "只看 Include": return dec == "Include"
    if filter_mode == "只看 Exclude": return dec == "Exclude"
    return True

for _, row in view_df.iterrows():
    rid = row["record_id"]
    ta_dec = st.session_state["ta_final"].get(rid, "Unsure")
    if not want(ta_dec):
        continue

    title = row.get("title","") or rid
    pmid = row.get("pmid","")
    doi  = row.get("doi","")
    pmcid= row.get("pmcid","")
    year = row.get("year","")
    fa   = row.get("first_author","")
    url  = row.get("url","")
    doi_url = row.get("doi_url","")
    pmc_link= row.get("pmc_url","")

    openurl = build_openurl(st.session_state.get("RESOLVER_BASE",""), doi=doi, pmid=pmid, title=title)
    openurl = apply_ezproxy(st.session_state.get("EZPROXY",""), openurl) if openurl else ""
    pub_link = apply_ezproxy(st.session_state.get("EZPROXY",""), url) if url else ""
    doi_link = apply_ezproxy(st.session_state.get("EZPROXY",""), doi_url) if doi_url else ""
    pmc_link2= apply_ezproxy(st.session_state.get("EZPROXY",""), pmc_link) if pmc_link else ""

    with st.expander(title, expanded=False):
        st.markdown('<div class="card">', unsafe_allow_html=True)
        meta = f"<div class='meta'><b>ID</b>: {rid}"
        if pmid: meta += f" &nbsp;&nbsp; <b>PMID</b>: {pmid}"
        if doi:  meta += f" &nbsp;&nbsp; <b>DOI</b>: {doi}"
        if year: meta += f" &nbsp;&nbsp; <b>Year</b>: {year}"
        if fa:   meta += f" &nbsp;&nbsp; <b>First author</b>: {fa}"
        meta += f" &nbsp;&nbsp; <b>Source</b>: {row.get('source','')}"
        meta += "</div>"
        st.markdown(meta, unsafe_allow_html=True)

        links = []
        if pub_link: links.append(f"[PubMed/Link]({pub_link})")
        if doi_link: links.append(f"[DOI]({doi_link})")
        if pmc_link2: links.append(f"[PMC OA]({pmc_link2})")
        if openurl: links.append(f"[全文(OpenURL)]({openurl})")
        if links:
            st.markdown(" | ".join(links))

        st.markdown(badge_html(ta_dec) + "<span class='small'> AI Title/Abstract 建議</span>", unsafe_allow_html=True)
        st.write(f"理由：{row.get('AI_reason','')}")
        st.caption(f"信心度：{row.get('AI_confidence','')}")

        st.markdown("### Abstract")
        st.write(row.get("abstract","") or "_No abstract available._")

        st.markdown('<hr class="soft">', unsafe_allow_html=True)
        st.markdown("### Full-text decision（看完全文後回填）")
        ft_opts = ["Not reviewed yet", "Include for meta-analysis", "Exclude after full-text"]
        cur_ft = st.session_state["ft_decisions"].get(rid, "Not reviewed yet")
        if cur_ft not in ft_opts:
            cur_ft = "Not reviewed yet"
        new_ft = st.radio("", ft_opts, index=ft_opts.index(cur_ft), key=f"ft_{rid}")
        st.session_state["ft_decisions"][rid] = new_ft

        ft_reason = st.text_area("Full-text reason / notes", value=st.session_state["ft_reasons"].get(rid,""), key=f"ft_reason_{rid}", height=80)
        st.session_state["ft_reasons"][rid] = ft_reason

        ft_note = st.text_input("若查不到全文：填原因/狀態（付費牆、館際、等待作者回信…）", value=st.session_state["ft_note"].get(rid,""), key=f"ft_note_{rid}")
        st.session_state["ft_note"][rid] = ft_note

        st.markdown("#### 上傳 PDF（可選）")
        uploaded_pdf = st.file_uploader("PDF 上傳（每篇文章各自上傳）", type=["pdf"], key=f"pdf_{rid}")
        extracted = ""
        if uploaded_pdf is not None and HAS_PYPDF2:
            try:
                reader = PdfReader(uploaded_pdf)
                texts = []
                for page in reader.pages[:80]:
                    t = page.extract_text() or ""
                    if t.strip():
                        texts.append(t)
                extracted = "\n".join(texts).strip()
                if not extracted:
                    st.warning("PDF 可能是掃描圖檔或無文字層。建議 OCR 後再上傳，或貼 figure/table 段落。")
                else:
                    st.success(f"已抽取文字（前 80 頁），長度={len(extracted)}。")
            except Exception as e:
                st.error(f"PDF 讀取失敗：{e}")
        elif uploaded_pdf is not None and not HAS_PYPDF2:
            st.warning("環境無 PyPDF2，無法從 PDF 抽字。請改用貼全文。")

        st.markdown("#### Full text / 關鍵段落（可貼全文、或貼 figure/table 相關段落）")
        default_text = st.session_state["ft_text"].get(rid,"")
        if extracted and len(extracted) > len(default_text):
            default_text = extracted
        ft_text = st.text_area("", value=default_text, key=f"ft_text_{rid}", height=180)
        st.session_state["ft_text"][rid] = ft_text

        st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# Step 4: Export screening summary (CSV) + PRISMA counts (basic)
# =========================================================
st.header("Step 4. 匯出（screening summary / fulltext 任務）")

def build_summary_df() -> pd.DataFrame:
    rows = []
    for _, r in df_records.iterrows():
        rid = r["record_id"]
        a = ai_ta_df[ai_ta_df["record_id"]==rid].head(1)
        a = a.iloc[0].to_dict() if not a.empty else {}
        openurl = build_openurl(st.session_state.get("RESOLVER_BASE",""), doi=r.get("doi",""), pmid=r.get("pmid",""), title=r.get("title",""))
        openurl = apply_ezproxy(st.session_state.get("EZPROXY",""), openurl) if openurl else ""
        rows.append({
            "record_id": rid,
            "pmid": r.get("pmid",""),
            "doi": r.get("doi",""),
            "year": r.get("year",""),
            "first_author": r.get("first_author",""),
            "title": r.get("title",""),
            "source": r.get("source",""),
            "institution_openurl": openurl,
            "AI_label": a.get("AI_label",""),
            "AI_confidence": a.get("AI_confidence",""),
            "AI_reason": a.get("AI_reason",""),
            "TA_final": st.session_state["ta_final"].get(rid, "Unsure"),
            "FT_decision": st.session_state["ft_decisions"].get(rid, "Not reviewed yet"),
            "FT_reason": st.session_state["ft_reasons"].get(rid, ""),
            "FT_note": st.session_state["ft_note"].get(rid, ""),
        })
    return pd.DataFrame(rows)

summary_df = build_summary_df()
st.download_button("下載 screening summary（CSV）", data=to_csv_bytes(summary_df), file_name="screening_summary.csv", mime="text/csv")

ft_queue = summary_df[summary_df["TA_final"].isin(["Include","Unsure"])].copy()
st.caption("Full-text 任務隊列（建議：Include + Unsure 都進去拿全文）")
st.dataframe(ft_queue[["record_id","title","TA_final","institution_openurl","FT_note"]], use_container_width=True)
st.download_button("下載 fulltext 任務隊列（CSV）", data=to_csv_bytes(ft_queue), file_name="fulltext_queue.csv", mime="text/csv")

# =========================================================
# Step 5: Extraction wide table + optional AI extraction (LLM)
# =========================================================
st.header("Step 5. Data extraction（寬表；欄位不寫死）")

base_frame = summary_df[["record_id","pmid","doi","title","institution_openurl","FT_note"]].copy()
for c in schema["base_cols"]:
    if c not in base_frame.columns:
        base_frame[c] = ""
for ocol in schema["outcomes"]:
    if ocol not in base_frame.columns:
        base_frame[ocol] = ""
for c in ["Effect_measure","Effect","Lower_CI","Upper_CI","Timepoint","Unit"]:
    if c not in base_frame.columns:
        base_frame[c] = ""

existing = st.session_state.get("extract_wide_df", pd.DataFrame())
if isinstance(existing, pd.DataFrame) and not existing.empty and "record_id" in existing.columns:
    # merge by record_id, keep old edits
    merged = base_frame.merge(existing, on="record_id", how="left", suffixes=("","_old"))
    for c in base_frame.columns:
        oc = c + "_old"
        if oc in merged.columns:
            merged[c] = merged.apply(lambda r: r[oc] if (str(r[oc]).strip() not in ["", "nan", "None"]) else r[c], axis=1)
            merged.drop(columns=[oc], inplace=True)
    base_frame = merged

edited = st.data_editor(
    base_frame,
    use_container_width=True,
    num_rows="dynamic",
    hide_index=True,
    column_config={"institution_openurl": st.column_config.LinkColumn("全文(openurl)", display_text="open")}
)
st.session_state["extract_wide_df"] = edited
st.download_button("下載 extraction 寬表（CSV）", data=to_csv_bytes(edited), file_name="extraction_wide.csv", mime="text/csv")

st.subheader("5B.（可選）AI extraction（含 OCR/figure/table 提示；抽不到留空）")
if llm_available():
    n_ai = st.number_input("每次 AI 抽取筆數", min_value=1, max_value=30, value=5, step=1)
    if st.button("執行 AI extraction（對前 N 筆已有全文者）"):
        with st.spinner("AI 抽取中…"):
            text_map = st.session_state["ft_text"]
            targets = []
            for rid in edited["record_id"].tolist():
                if (text_map.get(rid) or "").strip():
                    targets.append(rid)
                if len(targets) >= int(n_ai):
                    break
            if not targets:
                st.warning("沒有找到已貼/已抽取全文的研究。")
            else:
                prompt_template = build_extraction_prompt(schema, pico_now, st.session_state.get("criteria_text",""), goal_mode)
                for rid in targets:
                    fulltext = (text_map.get(rid) or "").strip()
                    messages = [
                        {"role":"system","content":"You are an SR/MA full-text reviewer and extractor. Output ONLY valid JSON."},
                        {"role":"user","content":prompt_template + "\n\n[Full text]\n" + fulltext[:120000]}
                    ]
                    txt = llm_chat(messages, temperature=0.1, timeout=150)
                    js = json_from_text(txt or "")
                    if not js:
                        continue

                    d = js.get("fulltext_decision")
                    if d in ["Include for meta-analysis","Exclude after full-text","Not reviewed yet"]:
                        st.session_state["ft_decisions"][rid] = d
                    if js.get("fulltext_reason"):
                        st.session_state["ft_reasons"][rid] = str(js.get("fulltext_reason"))

                    fields = js.get("extracted_fields") or {}
                    for k,v in fields.items():
                        if k in edited.columns:
                            edited.loc[edited["record_id"]==rid, k] = str(v)

                    meta = js.get("meta") or {}
                    if "effect_measure" in meta and "Effect_measure" in edited.columns:
                        edited.loc[edited["record_id"]==rid, "Effect_measure"] = str(meta.get("effect_measure",""))
                    if "effect" in meta and "Effect" in edited.columns:
                        edited.loc[edited["record_id"]==rid, "Effect"] = meta.get("effect","")
                    if "lower_CI" in meta and "Lower_CI" in edited.columns:
                        edited.loc[edited["record_id"]==rid, "Lower_CI"] = meta.get("lower_CI","")
                    if "upper_CI" in meta and "Upper_CI" in edited.columns:
                        edited.loc[edited["record_id"]==rid, "Upper_CI"] = meta.get("upper_CI","")
                    if "timepoint" in meta and "Timepoint" in edited.columns:
                        edited.loc[edited["record_id"]==rid, "Timepoint"] = str(meta.get("timepoint",""))
                    if "unit" in meta and "Unit" in edited.columns:
                        edited.loc[edited["record_id"]==rid, "Unit"] = str(meta.get("unit",""))

                st.session_state["extract_wide_df"] = edited
                st.success("AI 抽取完成（已回填到寬表）。")
else:
    st.info("未設定 LLM：可先用寬表手動抽取。")

# =========================================================
# Step 6: ROB 2.0 (manual)
# =========================================================
st.header("Step 6. ROB 2.0（手動下拉）")

rob_candidates = summary_df[(summary_df["FT_decision"]=="Include for meta-analysis")].copy()
if rob_candidates.empty:
    st.info("目前沒有 FT=Include for meta-analysis 的研究；ROB 2.0 通常在納入後做。")
else:
    rob_levels = ["", "Low risk", "Some concerns", "High risk"]
    domain_labels = [("D1","Randomization process"),
                     ("D2","Deviations from intended interventions"),
                     ("D3","Missing outcome data"),
                     ("D4","Measurement of the outcome"),
                     ("D5","Selection of the reported result"),
                     ("Overall","Overall Risk of Bias")]
    for _, r in rob_candidates.iterrows():
        rid = r["record_id"]
        st.markdown(f"**{r.get('first_author','')} ({r.get('year','')})** — {short(r.get('title',''), 120)}")
        cols = st.columns(6)
        rb = st.session_state["rob2"].get(rid, {}) or {}
        for i,(k,lab) in enumerate(domain_labels):
            with cols[i]:
                val = st.selectbox(lab, options=rob_levels,
                                   index=rob_levels.index(rb.get(k,"")) if rb.get(k,"") in rob_levels else 0,
                                   key=f"rob_{rid}_{k}")
                rb[k] = val
        st.session_state["rob2"][rid] = rb
        st.markdown("---")

    out_rows = []
    for _, r in rob_candidates.iterrows():
        rid = r["record_id"]
        name = f"{r.get('first_author','')} ({r.get('year','')})".strip() or rid
        rb = st.session_state["rob2"].get(rid, {}) or {}
        out_rows.append({
            "Study Name": name,
            "D1 Randomization": rb.get("D1",""),
            "D2 Deviations": rb.get("D2",""),
            "D3 Missing data": rb.get("D3",""),
            "D4 Measurement": rb.get("D4",""),
            "D5 Reporting": rb.get("D5",""),
            "Overall": rb.get("Overall",""),
        })
    df_rob = pd.DataFrame(out_rows)
    st.download_button("下載 ROB2（CSV）", data=to_csv_bytes(df_rob), file_name="rob2.csv", mime="text/csv")
