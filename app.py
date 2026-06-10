import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()


from resume_parser import extract_text_from_pdf, preprocess_text, extract_sections
from matcher import compute_similarity, find_skill_gaps, section_wise_similarity
from feedback_generator import generate_overall_feedback, generate_section_feedback, generate_gap_advice

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Resume Feedback Analyzer",
    page_icon="📄",
    layout="wide",
)

# ── Cache heavy models so they load once per session, not per interaction ─────
@st.cache_resource(show_spinner="Loading NLP models (first run only)...")
def load_sentence_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('all-MiniLM-L6-v2')

# Trigger model load immediately (not lazily on first analysis)
load_sentence_model()

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .score-card {
        background: linear-gradient(135deg, #1e2130, #252a40);
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        border: 1px solid #2e3250;
    }
    .score-number { font-size: 3.5rem; font-weight: 700; line-height: 1; }
    .score-label  { color: #8b93b0; font-size: 0.85rem; margin-top: 6px; }
    .tag {
        display: inline-block; padding: 3px 10px; border-radius: 20px;
        font-size: 0.78rem; margin: 3px;
    }
    .tag-match { background:#0d3320; color:#34d399; border:1px solid #065f46; }
    .tag-miss  { background:#3b1515; color:#f87171; border:1px solid #7f1d1d; }
    .feedback-box {
        background: #1a1d2e; border-left: 3px solid #6366f1;
        border-radius: 8px; padding: 16px; margin-bottom: 12px;
    }
    .stButton > button {
        background: #6366f1; color: white; border: none;
        border-radius: 8px; padding: 10px 28px;
        font-weight: 600; width: 100%;
    }
    .stButton > button:hover { background: #4f46e5; }
</style>
""", unsafe_allow_html=True)



# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 📄 Resume Feedback Assistant")
st.markdown("Upload your resume and paste a job description for ATS-style analysis and AI-powered feedback.")
st.markdown("---")

# ── Inputs ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### Upload Resume (PDF)")
    uploaded_file = st.file_uploader("", type=["pdf"], label_visibility="collapsed")

with col2:
    st.markdown("### Job Description")
    job_description = st.text_area(
        "", placeholder="Paste the full job description here...",
        height=220, label_visibility="collapsed",
    )

st.markdown("")
run_btn = st.button("🔍 Analyse Resume")

# ── Analysis ──────────────────────────────────────────────────────────────────
if run_btn:
    if not uploaded_file:
        st.error("Please upload a PDF resume.")
        st.stop()
    if not job_description.strip():
        st.error("Please paste a job description.")
        st.stop()

    has_api_key = bool(os.environ.get("GROQ_API_KEY"))

    # 1. Parse
    with st.spinner("Parsing resume..."):
        raw_text = extract_text_from_pdf(uploaded_file)
        clean_text = preprocess_text(raw_text)
        sections = extract_sections(clean_text)

    if len(clean_text) < 100:
        st.error("Could not extract enough text. Use a text-based PDF, not a scanned image.")
        st.stop()

    # 2. Match
    with st.spinner("Computing match score and keyword analysis..."):
        score = compute_similarity(clean_text, job_description)
        gaps = find_skill_gaps(clean_text, job_description)
        sec_scores = section_wise_similarity(sections, job_description)

    # 3. AI feedback (only if key available)
    overall_feedback = gap_advice = None
    if has_api_key:
        with st.spinner("Generating AI feedback..."):
            overall_feedback = generate_overall_feedback(clean_text, job_description, score)
            gap_advice = generate_gap_advice(gaps["missing"], job_description)

    st.markdown("---")

    # ── Score cards ───────────────────────────────────────────────────────────
    score_pct = int(score * 100)
    color = "#34d399" if score_pct >= 70 else "#fbbf24" if score_pct >= 45 else "#f87171"

    r1, r2, r3 = st.columns(3, gap="large")
    with r1:
        st.markdown(f'<div class="score-card"><div class="score-number" style="color:{color}">{score_pct}%</div><div class="score-label">ATS Match Score</div></div>', unsafe_allow_html=True)
    with r2:
        st.markdown(f'<div class="score-card"><div class="score-number" style="color:#34d399">{len(gaps["matched"])}</div><div class="score-label">Keywords Matched</div></div>', unsafe_allow_html=True)
    with r3:
        st.markdown(f'<div class="score-card"><div class="score-number" style="color:#f87171">{len(gaps["missing"])}</div><div class="score-label">Keyword Gaps</div></div>', unsafe_allow_html=True)

    st.markdown("")

    # ── Keyword tags ──────────────────────────────────────────────────────────
    st.markdown("### 🎯 Keyword Analysis")
    tab1, tab2 = st.tabs([f"✅ Matched ({len(gaps['matched'])})", f"❌ Missing ({len(gaps['missing'])})"])

    with tab1:
        if gaps["matched"]:
            tags = " ".join([f'<span class="tag tag-match">{kw}</span>' for kw in gaps["matched"]])
            st.markdown(tags, unsafe_allow_html=True)
        else:
            st.info("No strong keyword matches found.")

    with tab2:
        if gaps["missing"]:
            tags = " ".join([f'<span class="tag tag-miss">{kw}</span>' for kw in gaps["missing"]])
            st.markdown(tags, unsafe_allow_html=True)
        else:
            st.success("No significant keyword gaps detected.")

    # ── Section scores ────────────────────────────────────────────────────────
    if sec_scores:
        st.markdown("### 📊 Section-wise Relevance")
        for sec, s in sorted(sec_scores.items(), key=lambda x: x[1], reverse=True):
            pct = int(s * 100)
            bar_color = "#34d399" if pct >= 60 else "#fbbf24" if pct >= 35 else "#f87171"
            st.markdown(f"**{sec.title()}** — {pct}%")
            st.markdown(
                f'<div style="background:#1e2130;border-radius:6px;height:10px">'
                f'<div style="width:{pct}%;background:{bar_color};border-radius:6px;height:10px"></div></div>',
                unsafe_allow_html=True,
            )
            st.markdown("")

    # ── AI Feedback ───────────────────────────────────────────────────────────
    st.markdown("### 🤖 AI Improvement Suggestions")
    if overall_feedback:
        st.markdown(f'<div class="feedback-box">{overall_feedback}</div>', unsafe_allow_html=True)
    else:
        st.info("Add your Anthropic API key in the sidebar to unlock AI-generated suggestions.")

    if gap_advice:
        st.markdown("### 🧩 How to Close Skill Gaps")
        st.markdown(f'<div class="feedback-box">{gap_advice}</div>', unsafe_allow_html=True)

    # ── Per-section AI feedback ───────────────────────────────────────────────
    if has_api_key and sections:
        st.markdown("### 🔍 Section-Level Feedback")
        for sec, content in sections.items():
            if not content or len(content) < 30:
                continue
            with st.expander(f"📌 {sec.title()}"):
                with st.spinner(f"Analysing {sec}..."):
                    sec_fb = generate_section_feedback(sec, content, job_description)
                st.markdown(sec_fb)
