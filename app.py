{\rtf1\ansi\ansicpg950\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import streamlit as st\
import requests\
import xml.etree.ElementTree as ET\
import pandas as pd\
import math\
import os\
import json\
\
# =========================\
#  \uc0\u22522 \u26412 \u35498 \u26126 \
# =========================\
st.title("\uc0\u30524 \u31185  SRMA \u33258 \u21205 \u21270  Prototype\u65288 \u21547  GPT / Gemini AI \u21021 \u31721 \u65289 ")\
\
st.write(\
    """\
    \uc0\u21151 \u33021 \u27969 \u31243 \u65306 \
    1. \uc0\u36664 \u20837  PICO \u8594  \u33258 \u21205 \u32068 \u20986  PubMed \u25628 \u23563 \u24335 \u65288 \u21547 \u31777 \u21934  MeSH / \u21516 \u32681 \u35422 \u24314 \u35696 \u65289   \
    2. \uc0\u21628 \u21483  PubMed API \u25235 \u22238 \u25991 \u29563 \u12289 \u21435 \u37325 \u20006 \u35336 \u31639  PRISMA \u25976 \u23383   \
    3. \uc0\u26681 \u25818 \u12300 \u30070 \u21069  PICO\u12301 \u65291 \u65288 \u36984 \u29992 \u65289 GPT / Gemini \u23565 \u27599 \u31687 \u25991 \u29563 \u20570  AI \u21021 \u31721  (Include / Exclude / Unsure) \u20006 \u38468 \u29702 \u30001 \u65292 \u21487 \u36880 \u31687 \u20154 \u24037 \u26657 \u27491   \
    4. \uc0\u19978 \u20659  effect size CSV\u65292 \u35336 \u31639  fixed-effect pooled effect\u65288 \u25991 \u23383 \u32080 \u26524 \u65289 \
    """\
)\
\
# =========================\
#  LLM \uc0\u35373 \u23450 \u65288 \u20596 \u37002 \u27396 \u65306 \u21487 \u20999 \u25563  GPT / Gemini / \u19981 \u29992  LLM\u65289 \
# =========================\
st.sidebar.header("LLM \uc0\u35373 \u23450 ")\
\
llm_provider = st.sidebar.selectbox(\
    "AI \uc0\u27169 \u22411 \u20358 \u28304 \u65288 \u29992 \u20358 \u20570 \u21021 \u31721 \u33287 \u23531 \u29702 \u30001 \u65289 ",\
    [\
        "\uc0\u19981 \u20351 \u29992  LLM\u65288 \u20677 \u35215 \u21063 \u29256 \u65289 ",\
        "OpenAI GPT",\
        "Google Gemini",\
    ],\
)\
\
st.session_state.llm_provider = llm_provider\
\
# \uc0\u27298 \u26597  API key \u26159 \u21542 \u23384 \u22312 \u65288 \u25918 \u22312 \u29872 \u22659 \u35722 \u25976 \u25110  .streamlit/secrets.toml \u30342 \u21487 \u65289 \
openai_key = st.secrets.get("OPENAI_API_KEY", None) or os.environ.get("OPENAI_API_KEY")\
gemini_key = (\
    st.secrets.get("GEMINI_API_KEY", None)\
    or os.environ.get("GEMINI_API_KEY")\
    or os.environ.get("GOOGLE_API_KEY")\
)\
\
if llm_provider == "OpenAI GPT":\
    if not openai_key:\
        st.sidebar.warning("\uc0\u26410 \u20597 \u28204 \u21040  OPENAI_API_KEY\u65292 \u23526 \u38555 \u26371 \u36864 \u22238 \u35215 \u21063 \u29256 \u21028 \u35712 \u12290 ")\
elif llm_provider == "Google Gemini":\
    if not gemini_key:\
        st.sidebar.warning("\uc0\u26410 \u20597 \u28204 \u21040  GEMINI_API_KEY / GOOGLE_API_KEY\u65292 \u23526 \u38555 \u26371 \u36864 \u22238 \u35215 \u21063 \u29256 \u21028 \u35712 \u12290 ")\
\
\
# =========================\
#  PubMed utilities\
# =========================\
def pubmed_esearch(term: str, retmax: int = 50):\
    """\uc0\u29992  ESearch \u21040  PubMed \u21462 \u24471  PMID list\u12290 """\
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"\
    params = \{\
        "db": "pubmed",\
        "term": term,\
        "retmode": "json",\
        "retmax": retmax,\
    \}\
    r = requests.get(url, params=params, timeout=30)\
    r.raise_for_status()\
    data = r.json()\
    return data["esearchresult"]["idlist"]\
\
\
def pubmed_efetch(id_list):\
    """\uc0\u29992  EFetch \u25235 \u22238  PubMed \u35443 \u32048 \u36039 \u26009 \u65292 \u22238 \u20659  dict list\u12290 """\
    if not id_list:\
        return []\
\
    ids = ",".join(id_list)\
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"\
    params = \{\
        "db": "pubmed",\
        "id": ids,\
        "retmode": "xml",\
    \}\
    r = requests.get(url, params=params, timeout=60)\
    r.raise_for_status()\
\
    root = ET.fromstring(r.text)\
    records = []\
\
    for article in root.findall(".//PubmedArticle"):\
        pmid_el = article.find(".//PMID")\
        title_el = article.find(".//ArticleTitle")\
        doi_el = article.find(".//ArticleId[@IdType='doi']")\
\
        pmid = pmid_el.text if pmid_el is not None else ""\
        title = title_el.text if title_el is not None else ""\
        doi = doi_el.text if doi_el is not None else ""\
\
        # \uc0\u25235 \u25152 \u26377  AbstractText\u65292 \u21253 \u21547 \u26377  Label \u30340 \u22810 \u27573 \u25688 \u35201 \
        abstract_parts = []\
        for ab in article.findall(".//AbstractText"):\
            if ab.text:\
                label = ab.get("Label")\
                if label:\
                    abstract_parts.append(f"\{label\}: \{ab.text\}")\
                else:\
                    abstract_parts.append(ab.text)\
        abstract = "\\n".join(abstract_parts) if abstract_parts else ""\
\
        records.append(\
            \{\
                "PMID": pmid,\
                "Title": title,\
                "Abstract": abstract,\
                "DOI": doi,\
            \}\
        )\
\
    return records\
\
\
# =========================\
#  MeSH / \uc0\u21516 \u32681 \u35422 \u24314 \u35696 \u65288 \u31777 \u21270 \u29256 \u65292 \u21487 \u20043 \u24460 \u25563 \u25104  LLM \u25110 \u30495 \u27491  MeSH \u26597 \u35426 \u65289 \
# =========================\
def suggest_mesh_block(term: str, role: str):\
    """\
    \uc0\u26681 \u25818 \u36664 \u20837  term \u29986 \u29983 \u19968 \u27573 \u24314 \u35696 \u25628 \u23563 \u24335  (string)\u12290 \
    role: "P" / "I" / "C" / "O"\uc0\u65292 \u30446 \u21069 \u20677 \u20570 \u23569 \u37327 \u30524 \u31185 \u29305 \u20363 \u12290 \
    """\
    if not term:\
        return ""\
\
    t = term.lower().strip()\
\
    # \uc0\u24120 \u35211 \u30524 \u31185 \u35422 \u29305 \u20363 \
    if "glaucoma" in t:\
        return "(glaucoma[MeSH Terms] OR glaucoma* OR ocular hypertension)"\
    if "cataract" in t:\
        return "(Cataract[MeSH Terms] OR cataract* OR lens opacity)"\
    if "edof" in t and "iol" in t:\
        return "(\\"extended depth of focus\\" OR EDOF OR \\"extended depth-of-focus\\" OR presbyopia-correcting IOL OR Intraocular Lenses[MeSH Terms])"\
    if "iol" in t or "intraocular lens" in t:\
        return "(Intraocular Lenses[MeSH Terms] OR intraocular lens* OR IOL*)"\
\
    # \uc0\u36890 \u29992  fallback\
    return f"(\{term\}[MeSH Terms] OR \{term\})"\
\
\
def build_pubmed_query(p, i, c, o, extra, add_rct_filter):\
    """\uc0\u25226  PICO + \u38989 \u22806 \u26781 \u20214 \u36681 \u25104  PubMed \u25628 \u23563 \u24335 \u65292 \u20839 \u21547 \u31777 \u26131  MeSH block\u12290 """\
    blocks = []\
\
    if p:\
        blocks.append(suggest_mesh_block(p, "P"))\
    if i:\
        blocks.append(suggest_mesh_block(i, "I"))\
    if c:\
        blocks.append(suggest_mesh_block(c, "C"))\
    if o:\
        blocks.append(suggest_mesh_block(o, "O"))\
\
    query = " AND ".join(blocks) if blocks else ""\
\
    if extra.strip():\
        query = f"\{query\} AND (\{extra.strip()\})" if query else extra.strip()\
\
    if add_rct_filter:\
        rct_block = "(randomized controlled trial[pt] OR randomized[tiab] OR randomised[tiab])"\
        query = f"\{query\} AND \{rct_block\}" if query else rct_block\
\
    return query\
\
\
# =========================\
#  LLM helper\uc0\u65306 \u21628 \u21483  GPT / Gemini \u20570 \u21021 \u31721 \u65288 \u33509 \u22833 \u25943 \u23601 \u22238  None\u65289 \
# =========================\
def extract_json_like(text: str):\
    """\uc0\u24478  LLM \u22238 \u35206 \u20013 \u30433 \u37327 \u25235 \u20986 \u31532 \u19968 \u27573  JSON\u65292 \u22833 \u25943 \u23601 \u19999 \u20363 \u22806 \u12290 """\
    first = text.find("\{")\
    last = text.rfind("\}")\
    if first == -1 or last == -1 or last <= first:\
        raise ValueError("No JSON found")\
    snippet = text[first : last + 1]\
    return json.loads(snippet)\
\
\
def llm_screening(title: str, abstract: str, pico: dict, provider: str):\
    """\
    \uc0\u29992  LLM \u20570  include/exclude/unsure \u21028 \u26039 \u12290 \
    \uc0\u22238 \u20659  (label, reason) \u25110  None \u34920 \u31034 \u22833 \u25943 \u65292 \u22806 \u38754 \u20877  fallback \u21040 \u35215 \u21063 \u29256 \u12290 \
    """\
    if provider == "OpenAI GPT" and openai_key:\
        try:\
            from openai import OpenAI  # type: ignore\
\
            client = OpenAI(api_key=openai_key)\
\
            prompt = f"""\
You are assisting in a systematic review in ophthalmology.\
\
The PICO of the current review is:\
P: \{pico.get('P', '')\}\
I: \{pico.get('I', '')\}\
C: \{pico.get('C', '')\}\
O: \{pico.get('O', '')\}\
\
Given the following PubMed record (title and abstract), decide whether it should be:\
- "Include" (matches PICO reasonably well and likely relevant),\
- "Exclude" (clearly not related),\
- or "Unsure" (partially related or insufficient information).\
\
Title:\
\{title\}\
\
Abstract:\
\{abstract\}\
\
Return ONLY a JSON object in the following format:\
\
\{\{\
  "label": "Include or Exclude or Unsure",\
  "reason": "A short explanation in Chinese, explicitly mentioning which parts of PICO match or mismatch."\
\}\}\
"""\
            resp = client.responses.create(\
                model="gpt-5.1-mini",  # \uc0\u21487 \u35222 \u24773 \u27841 \u25913 \u20854 \u20182 \u22411 \u34399 \
                input=prompt,\
            )\
            text = resp.output_text\
            data = extract_json_like(text)\
            label = data.get("label", "").strip()\
            reason = data.get("reason", "").strip()\
            if label not in ["Include", "Exclude", "Unsure"]:\
                raise ValueError("Invalid label from GPT")\
            return label, reason or "\uc0\u65288 GPT \u26410 \u25552 \u20379 \u29702 \u30001 \u65289 "\
        except Exception as e:\
            # \uc0\u22312  UI \u20013 \u39023 \u31034 \u19968 \u27425 \u21363 \u21487 \u65292 \u36991 \u20813 \u27927 \u29256 \
            st.warning(f"GPT \uc0\u21021 \u31721 \u20986 \u37679 \u65292 \u24050 \u25913 \u29992 \u35215 \u21063 \u29256 \u12290 \u35443 \u32048 \u35338 \u24687 \u65306 \{e\}")\
            return None\
\
    if provider == "Google Gemini" and gemini_key:\
        try:\
            import google.generativeai as genai  # type: ignore\
\
            genai.configure(api_key=gemini_key)\
            model = genai.GenerativeModel("gemini-1.5-flash")\
\
            prompt = f"""\
You are assisting in a systematic review in ophthalmology.\
\
The PICO of the current review is:\
P: \{pico.get('P', '')\}\
I: \{pico.get('I', '')\}\
C: \{pico.get('C', '')\}\
O: \{pico.get('O', '')\}\
\
Given the following PubMed record (title and abstract), decide whether it should be:\
- "Include" (matches PICO reasonably well and likely relevant),\
- "Exclude" (clearly not related),\
- or "Unsure" (partially related or insufficient information).\
\
Title:\
\{title\}\
\
Abstract:\
\{abstract\}\
\
Return ONLY a JSON object in the following format:\
\
\{\{\
  "label": "Include or Exclude or Unsure",\
  "reason": "A short explanation in Chinese, explicitly mentioning which parts of PICO match or mismatch."\
\}\}\
"""\
            resp = model.generate_content(prompt)\
            text = resp.text or ""\
            data = extract_json_like(text)\
            label = data.get("label", "").strip()\
            reason = data.get("reason", "").strip()\
            if label not in ["Include", "Exclude", "Unsure"]:\
                raise ValueError("Invalid label from Gemini")\
            return label, reason or "\uc0\u65288 Gemini \u26410 \u25552 \u20379 \u29702 \u30001 \u65289 "\
        except Exception as e:\
            st.warning(f"Gemini \uc0\u21021 \u31721 \u20986 \u37679 \u65292 \u24050 \u25913 \u29992 \u35215 \u21063 \u29256 \u12290 \u35443 \u32048 \u35338 \u24687 \u65306 \{e\}")\
            return None\
\
    # \uc0\u33509  provider \u19981 \u31526 \u25110 \u27794  key\
    return None\
\
\
# =========================\
#  \uc0\u35215 \u21063 \u29256 \u21021 \u31721 \u65288 \u30070  LLM \u38364 \u25481 \u25110 \u22833 \u25943 \u26178 \u20351 \u29992 \u65289 \
# =========================\
def rule_based_screening(title: str, abstract: str, pico: dict):\
    """\
    \uc0\u22238 \u20659  (\u24314 \u35696 \u27161 \u31844 , \u29702 \u30001 \u25991 \u23383 )\
    label: 'Include' / 'Exclude' / 'Unsure'\
    """\
    text = (title or "") + " " + (abstract or "")\
    low = text.lower()\
\
    if not low.strip():\
        return "Unsure", "Title/Abstract \uc0\u24190 \u20046 \u27794 \u26377 \u20839 \u23481 \u65292 \u28961 \u27861 \u26681 \u25818  PICO \u21028 \u35712 \u65292 \u24314 \u35696 \u20154 \u24037 \u27298 \u35222 \u20840 \u25991 \u12290 "\
\
    def count_match(term):\
        if not term:\
            return 0\
        words = [w.strip().lower() for w in term.split() if w.strip()]\
        return sum(1 for w in words if w in low)\
\
    p_match = count_match(pico.get("P", ""))\
    i_match = count_match(pico.get("I", ""))\
    c_match = count_match(pico.get("C", ""))\
    o_match = count_match(pico.get("O", ""))\
\
    total_match = p_match + i_match + c_match + o_match\
    is_trial = any(k in low for k in ["randomized", "randomised", "prospective", "trial"])\
\
    if total_match == 0:\
        label = "Exclude"\
        reason = "Title/Abstract \uc0\u24190 \u20046 \u27794 \u26377 \u25552 \u21040 \u30070 \u21069  PICO \u30340  Population / Intervention / Comparison / Outcome\u65292 \u25512 \u28204 \u30740 \u31350 \u20027 \u38988 \u33287 \u26412 \u27425 \u21839 \u38988 \u38364 \u32879 \u24230 \u20302 \u12290 "\
        return label, reason\
\
    if (p_match > 0 or i_match > 0) and is_trial:\
        label = "Include"\
        reason_parts = []\
        if p_match > 0:\
            reason_parts.append("Population \uc0\u33287  P \u23383 \u20018 \u26377 \u23565 \u25033 \u38364 \u37749 \u35422 ")\
        if i_match > 0:\
            reason_parts.append("Intervention \uc0\u33287  I \u23383 \u20018 \u26377 \u23565 \u25033 \u38364 \u37749 \u35422 ")\
        if c_match > 0:\
            reason_parts.append("Comparison \uc0\u33287  C \u23383 \u20018 \u21487 \u33021 \u26377 \u37096 \u20998 \u23565 \u25033 ")\
        if o_match > 0:\
            reason_parts.append("Outcome \uc0\u33287  O \u23383 \u20018 \u26377 \u30456 \u38364 \u25551 \u36848 ")\
        reason = "\uc0\u65292 \u65307 ".join(reason_parts)\
        reason += "\uc0\u12290 \u19988 \u25688 \u35201 \u20013 \u25552 \u21040  randomized/prospective/trial\u65292 \u25512 \u28204 \u28858 \u20171 \u20837 \u24615 \u35430 \u39511 \u65292 \u26283 \u21015 \u28858  Include\u65292 \u24453 \u20154 \u24037 \u35079 \u26680 \u12290 "\
        return label, reason\
\
    label = "Unsure"\
    reason = (\
        f"Title/Abstract \uc0\u20013 \u23565 \u26044  P/I/C/O \u26377  \{total_match\} \u20491 \u38364 \u37749 \u35422 \u21629 \u20013 "\
        "\uc0\u65288 \u33267 \u23569 \u37096 \u20998 \u20839 \u23481 \u33287 \u30070 \u21069  PICO \u30456 \u31526 \u65289 \u65292 \u20294 \u26410 \u26126 \u30906 \u39023 \u31034 \u28858 \u38568 \u27231 \u23565 \u29031 \u35430 \u39511 \u25110 \u30740 \u31350 \u35373 \u35336 \u19981 \u28165 \u65292 \u26283 \u21015  Unsure\u65292 \u24314 \u35696 \u20154 \u24037 \u21028 \u35712 \u33287 \u30475 \u20840 \u25991 \u12290 "\
    )\
    return label, reason\
\
\
def ai_screening_suggestion(title: str, abstract: str, pico: dict, provider: str):\
    """\
    \uc0\u20808 \u22039 \u35430 \u29992  LLM\u65307 \u33509  provider \u35373 \u28858 \u19981 \u29992 \u25110  LLM \u22833 \u25943 \u65292 \u23601  fallback \u21040 \u35215 \u21063 \u29256 \u12290 \
    """\
    if provider != "\uc0\u19981 \u20351 \u29992  LLM\u65288 \u20677 \u35215 \u21063 \u29256 \u65289 ":\
        llm_result = llm_screening(title, abstract, pico, provider)\
        if llm_result is not None:\
            return llm_result\
\
    # fallback\
    return rule_based_screening(title, abstract, pico)\
\
\
# =========================\
#  \uc0\u21021 \u22987 \u21270  session_state\
# =========================\
if "decisions" not in st.session_state:\
    st.session_state.decisions = \{\}          # PMID -> final decision\
if "ai_suggestions" not in st.session_state:\
    st.session_state.ai_suggestions = \{\}     # PMID -> (label, reason)\
if "results_df" not in st.session_state:\
    st.session_state.results_df = None\
if "last_query" not in st.session_state:\
    st.session_state.last_query = ""\
if "prisma" not in st.session_state:\
    st.session_state.prisma = \{\}\
if "pico" not in st.session_state:\
    st.session_state.pico = \{\}\
\
\
# =========================\
#  Step 1: PICO \uc0\u36664 \u20837 \
# =========================\
st.header("Step 1\uc0\u65306 \u36664 \u20837  PICO")\
\
col1, col2 = st.columns(2)\
with col1:\
    p = st.text_input("P (Population)", "glaucoma patients")\
    i = st.text_input("I (Intervention)", "trabeculectomy")\
with col2:\
    c = st.text_input("C (Comparison)", "tube shunt")\
    o = st.text_input("O (Outcome)", "intraocular pressure reduction")\
\
extra = st.text_input(\
    "\uc0\u38989 \u22806 \u38364 \u37749 \u23383 \u65295 \u38480 \u21046 \u65288 \u21487 \u30041 \u30333 \u65292 \u20363 \u22914 : selective laser trabeculoplasty\u65289 ", ""\
)\
\
add_rct_filter = st.checkbox("\uc0\u33258 \u21205 \u21152 \u19978  RCT \u38364 \u37749 \u23383 ", value=True)\
\
retmax = st.slider("\uc0\u25235 \u22238 \u25991 \u29563 \u25976 \u37327  (records identified)", 10, 200, 50)\
\
st.session_state.pico = \{"P": p, "I": i, "C": c, "O": o\}\
\
st.markdown("#### \uc0\u24314 \u35696 \u30340  PICO \u23565 \u25033 \u25628 \u23563 \u22602 \u65288 \u31777 \u26131  MeSH / \u21516 \u32681 \u35422 \u65289 ")\
with st.expander("\uc0\u40670 \u38283 \u30475 \u24314 \u35696 \u25628 \u23563 \u22602 "):\
    st.write("P block:", suggest_mesh_block(p, "P") if p else "(\uc0\u26410 \u36664 \u20837 )")\
    st.write("I block:", suggest_mesh_block(i, "I") if i else "(\uc0\u26410 \u36664 \u20837 )")\
    st.write("C block:", suggest_mesh_block(c, "C") if c else "(\uc0\u26410 \u36664 \u20837 )")\
    st.write("O block:", suggest_mesh_block(o, "O") if o else "(\uc0\u26410 \u36664 \u20837 )")\
\
st.markdown("---")\
st.header("Step 2\uc0\u65306 \u29986 \u29983  PubMed \u25628 \u23563 \u24335 \u20006 \u25628 \u23563 ")\
\
\
# =========================\
#  Step 2: \uc0\u25628 \u23563  PubMed + AI \u21021 \u31721 \
# =========================\
if st.button("\uc0\u29986 \u29983 \u25628 \u23563 \u24335 \u20006 \u25628 \u23563  PubMed"):\
    query = build_pubmed_query(p, i, c, o, extra, add_rct_filter)\
    st.session_state.last_query = query\
\
    st.session_state.decisions = \{\}\
    st.session_state.ai_suggestions = \{\}\
    st.session_state.prisma = \{\}\
\
    with st.spinner("\uc0\u21521  PubMed \u26597 \u35426 \u20013 \u65292 \u35531 \u31245 \u20505 \'85"):\
        try:\
            pmids = pubmed_esearch(query, retmax=retmax)\
            records = pubmed_efetch(pmids)\
            if not records:\
                st.warning("\uc0\u27794 \u26377 \u25235 \u21040 \u20219 \u20309 \u25991 \u29563 \u20839 \u23481 \u65292 \u21487 \u33021 \u26159 \u25628 \u23563 \u24335 \u22826 \u22196 \u26684 \u12290 ")\
                st.session_state.results_df = None\
            else:\
                df_raw = pd.DataFrame(records)\
\
                identified = len(df_raw)\
\
                df_dedup = df_raw.drop_duplicates(\
                    subset=["PMID", "DOI", "Title"], keep="first"\
                )\
                after_dedup = len(df_dedup)\
\
                ai_suggestions = \{\}\
                decisions = \{\}\
                for idx, row in df_dedup.iterrows():\
                    pmid = row["PMID"] or f"idx_\{idx\}"\
                    label, reason = ai_screening_suggestion(\
                        row["Title"],\
                        row["Abstract"],\
                        st.session_state.pico,\
                        st.session_state.llm_provider,\
                    )\
                    ai_suggestions[pmid] = (label, reason)\
                    decisions[pmid] = label\
\
                st.session_state.results_df = df_dedup\
                st.session_state.ai_suggestions = ai_suggestions\
                st.session_state.decisions = decisions\
                st.session_state.prisma = \{\
                    "identified": identified,\
                    "after_dedup": after_dedup,\
                \}\
\
                st.success(\
                    f"\uc0\u25214 \u21040  \{identified\} \u31687 \u25991 \u29563 \u65292 \u21435 \u37325 \u24460 \u21097  \{after_dedup\} \u31687 \u65292 "\
                    f"\uc0\u24050 \u23436 \u25104  \{'LLM \u39493 \u21205 ' if st.session_state.llm_provider != '\u19981 \u20351 \u29992  LLM\u65288 \u20677 \u35215 \u21063 \u29256 \u65289 ' else '\u35215 \u21063 \u29256 '\} AI \u21021 \u31721 \u12290 "\
                )\
\
        except Exception as e:\
            st.session_state.results_df = None\
            st.error(f"\uc0\u25628 \u23563 \u25110 \u19979 \u36617  PubMed \u36039 \u26009 \u26178 \u30332 \u29983 \u37679 \u35492 \u65306 \{e\}")\
\
\
# =========================\
#  Step 3: \uc0\u39023 \u31034 \u32080 \u26524 \u12289 \u36880 \u31687 \u26657 \u27491  & PRISMA\
# =========================\
df = st.session_state.results_df\
\
if df is not None and not df.empty:\
    st.subheader("\uc0\u29986 \u29983 \u30340  PubMed \u25628 \u23563 \u24335 ")\
    st.code(st.session_state.last_query or "(\uc0\u23578 \u26410 \u25628 \u23563 )", language="text")\
\
    st.subheader("\uc0\u25628 \u23563 \u32080 \u26524 \u32317 \u35261 \u65288 \u21547  AI \u24314 \u35696 \u65289 ")\
    overview_df = df[["PMID", "Title", "DOI"]].copy()\
    ai_labels = []\
    for idx, row in df.iterrows():\
        pmid = row["PMID"] or f"idx_\{idx\}"\
        label, _ = st.session_state.ai_suggestions.get(pmid, ("\uc0\u26410 \u38928 \u28204 ", ""))\
        ai_labels.append(label)\
    overview_df["AI_suggestion"] = ai_labels\
    st.dataframe(overview_df, use_container_width=True)\
\
    st.markdown("---")\
    st.header("Step 3\uc0\u65306 \u36880 \u31687 \u31721 \u36984  & \u26657 \u27491  AI \u21028 \u35712 ")\
\
    include_count = 0\
    exclude_count = 0\
    unsure_count = 0\
\
    for idx, row in df.iterrows():\
        pmid = row["PMID"] or f"idx_\{idx\}"\
        title = row["Title"]\
        abstract = row["Abstract"]\
\
        ai_label, ai_reason = st.session_state.ai_suggestions.get(\
            pmid, ("\uc0\u26410 \u38928 \u28204 ", "\u65288 \u28961  AI \u29702 \u30001 \u65289 ")\
        )\
        current_decision = st.session_state.decisions.get(pmid, ai_label)\
\
        with st.expander(f"\{idx+1\}. \{title[:120]\}"):\
            st.write(f"**PMID:** \{pmid\}")\
            st.write(f"**AI \uc0\u21021 \u31721 \u24314 \u35696 \u65306 \{ai_label\}**")\
            st.caption(f"\uc0\u29702 \u30001 \u65306 \{ai_reason\}")\
\
            if abstract:\
                st.write("**Abstract**")\
                st.write(abstract)\
            else:\
                st.write("_No abstract available._")\
\
            decision = st.radio(\
                "\uc0\u20320 \u30340 \u26368 \u32066 \u21028 \u26039 \u65288 \u21487 \u35206 \u23531  AI \u24314 \u35696 \u65289 ",\
                ["Include", "Exclude", "Unsure"],\
                index=["Include", "Exclude", "Unsure"].index(\
                    current_decision\
                    if current_decision in ["Include", "Exclude", "Unsure"]\
                    else "Unsure"\
                ),\
                key=f"decision_radio_\{pmid\}",\
            )\
            st.session_state.decisions[pmid] = decision\
\
    for d in st.session_state.decisions.values():\
        if d == "Include":\
            include_count += 1\
        elif d == "Exclude":\
            exclude_count += 1\
        elif d == "Unsure":\
            unsure_count += 1\
\
    st.markdown("---")\
    st.subheader("PRISMA \uc0\u32113 \u35336 \u65288 \u31777 \u21270 \u29256 \u65289 ")\
    prisma = st.session_state.prisma or \{\}\
    st.write(\
        f"Records identified (database searching)\uc0\u65306 \{prisma.get('identified', 'NA')\}"\
    )\
    st.write(\
        f"Records after duplicates removed\uc0\u65306 \{prisma.get('after_dedup', 'NA')\}"\
    )\
    st.write(\
        f"Records included for full-text review (Include)\uc0\u65306 \{include_count\}"\
    )\
    st.write(\
        f"Records marked as Unsure\uc0\u65288 \u21487 \u35222 \u28858 \u24453 \u30906 \u35469  full-text\u65289 \u65306 \{unsure_count\}"\
    )\
    st.write(f"Records excluded at screening\uc0\u65306 \{exclude_count\}")\
\
else:\
    st.info("\uc0\u35531 \u20808 \u22312  Step 2 \u29986 \u29983 \u25628 \u23563 \u24335 \u20006 \u25628 \u23563  PubMed\u65292 \u32080 \u26524 \u25165 \u26371 \u39023 \u31034 \u22312 \u36889 \u35041 \u12290 ")\
\
\
# =========================\
#  Step 4: \uc0\u19978 \u20659  effect size CSV\u65292 \u35336 \u31639  pooled effect\u65288 \u19981 \u30059 \u22294 \u65289 \
# =========================\
st.markdown("---")\
st.header("Step 4\uc0\u65306 \u19978 \u20659  effect size \u36039 \u26009 \u20006 \u35336 \u31639  pooled effect\u65288 \u31034 \u24847 \u29256 \u65289 ")\
\
st.write(\
    """\
    \uc0\u35531 \u20808 \u20154 \u24037 \u25110 \u20351 \u29992 \u20854 \u20182 \u24037 \u20855 \u23436 \u25104 \u36039 \u26009 \u33795 \u21462 \u65292 \u25972 \u29702 \u25104  CSV\u65292 \u21253 \u21547 \u20197 \u19979 \u27396 \u20301 \u65306 \
    - study\uc0\u65306 \u30740 \u31350 \u21517 \u31281 \
    - effect\uc0\u65306 \u25928 \u26524 \u37327 \u65288 \u20363 \u22914  mean difference, log RR \u31561 \u65289 \
    - ci_lower\uc0\u65306 95% \u20449 \u36084 \u21312 \u38291 \u19979 \u38480 \
    - ci_upper\uc0\u65306 95% \u20449 \u36084 \u21312 \u38291 \u19978 \u38480 \
\
    \uc0\u19978 \u20659 \u24460 \u65292 \u31995 \u32113 \u26371 \u35336 \u31639  fixed-effect pooled effect \u20006 \u39023 \u31034 \u25976 \u20540 \u32080 \u26524 \u65288 \u26283 \u19981 \u30059 \u26862 \u26519 \u22294 \u65289 \u12290 \
    """\
)\
\
uploaded = st.file_uploader("\uc0\u19978 \u20659  effect size CSV", type="csv")\
\
if uploaded is not None:\
    try:\
        eff_df = pd.read_csv(uploaded)\
    except Exception as e:\
        st.error(f"\uc0\u35712 \u21462  CSV \u26178 \u30332 \u29983 \u37679 \u35492 \u65306 \{e\}")\
        eff_df = None\
\
    if eff_df is not None:\
        required_cols = \{"study", "effect", "ci_lower", "ci_upper"\}\
        if not required_cols.issubset(set(eff_df.columns)):\
            st.error(f"CSV \uc0\u38656 \u21253 \u21547 \u27396 \u20301 \u65306 \{required_cols\}")\
        else:\
            st.subheader("\uc0\u19978 \u20659 \u30340 \u36039 \u26009 ")\
            st.dataframe(eff_df, use_container_width=True)\
\
            effects = eff_df["effect"].tolist()\
            lowers = eff_df["ci_lower"].tolist()\
            uppers = eff_df["ci_upper"].tolist()\
\
            weights = []\
            for eff, lo, up in zip(effects, lowers, uppers):\
                se = (up - lo) / (2 * 1.96)\
                if se <= 0:\
                    se = 1e-6\
                weights.append(1.0 / (se ** 2))\
\
            num = sum(w * e for w, e in zip(weights, effects))\
            den = sum(weights)\
            pooled_effect = num / den if den > 0 else float("nan")\
            pooled_var = 1.0 / den if den > 0 else float("nan")\
            pooled_se = math.sqrt(pooled_var) if pooled_var > 0 else float("nan")\
            pooled_ci_lower = pooled_effect - 1.96 * pooled_se\
            pooled_ci_upper = pooled_effect + 1.96 * pooled_se\
\
            st.subheader("Fixed-effect pooled effect")\
            st.write(\
                f"Effect = \{pooled_effect:.3f\} "\
                f"(95% CI \{pooled_ci_lower:.3f\} ~ \{pooled_ci_upper:.3f\})"\
            )\
\
            st.write(\
                "\uc0\u20043 \u24460 \u33509 \u22312 \u26412 \u27231 \u25110  lab server \u19978 \u37096 \u32626 \u65292 \u21487 \u20197 \u22312 \u36889 \u19968 \u27573 \u21152 \u19978  matplotlib / plotly \u30059 \u26862 \u26519 \u22294 \u65292 "\
                "\uc0\u20006 \u25509 \u19978  full-text parser \u20570 \u33258 \u21205  extract\u12290 "\
            )\
}