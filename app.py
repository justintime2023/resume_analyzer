# app.py
# Streamlit Web UI for the LangGraph Resume Screening Agent
# Users upload multiple resumes + paste a job description → get ranked results

import streamlit as st
import PyPDF2
import io
import os
from resume_agent_core import run_screening

# --- Page Config ---
st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="🎯",
    layout="wide"
)

# --- Custom Styling ---
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .score-badge {
        display: inline-block;
        padding: 4px 16px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1em;
        color: white;
    }
    .score-high { background-color: #2ecc71; }
    .score-mid { background-color: #f39c12; }
    .score-low { background-color: #e74c3c; }
    .top-pick-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# Sidebar: API Configuration
# =====================================================

with st.sidebar:
    st.header("🔑 API Configuration")
    st.caption("Enter your GenAI credentials below. These are never stored or logged.")

    with st.expander("Configure API Settings", expanded=not all([
        st.session_state.get("genai_api_key"),
        st.session_state.get("genai_base_url"),
        st.session_state.get("genai_model"),
    ])):
        api_key = st.text_input(
            "GENAI API Key",
            type="password",
            value=st.session_state.get("genai_api_key", os.getenv("GENAI_API_KEY", "")),
            placeholder="Enter your API key",
            key="input_api_key"
        )
        base_url = st.text_input(
            "GENAI Base URL",
            value=st.session_state.get("genai_base_url", os.getenv("GENAI_BASE_URL", "")),
            placeholder="https://your-api-endpoint.com/v1",
            key="input_base_url"
        )
        model = st.text_input(
            "GENAI Model",
            value=st.session_state.get("genai_model", os.getenv("GENAI_MODEL", "")),
            placeholder="e.g. gpt-4o-mini, llama-3, etc.",
            key="input_model"
        )

        if st.button("💾 Save Configuration", use_container_width=True):
            st.session_state["genai_api_key"] = api_key
            st.session_state["genai_base_url"] = base_url
            st.session_state["genai_model"] = model
            st.success("Configuration saved!")

    # Show connection status
    has_key = bool(api_key or st.session_state.get("genai_api_key"))
    has_url = bool(base_url or st.session_state.get("genai_base_url"))
    has_model = bool(model or st.session_state.get("genai_model"))

    if has_key and has_url and has_model:
        st.success("✅ API configured")
    else:
        missing = []
        if not has_key: missing.append("API Key")
        if not has_url: missing.append("Base URL")
        if not has_model: missing.append("Model")
        st.warning(f"Missing: {', '.join(missing)}")

    st.divider()
    st.caption("ℹ️ Credentials are kept in your browser session only. Nothing is saved to disk or sent anywhere except to your configured API endpoint.")

# Set env vars from sidebar inputs (so resume_agent_core picks them up)
_api_key = api_key or st.session_state.get("genai_api_key", "")
_base_url = base_url or st.session_state.get("genai_base_url", "")
_model = model or st.session_state.get("genai_model", "")

if _api_key:
    os.environ["GENAI_API_KEY"] = _api_key
if _base_url:
    os.environ["GENAI_BASE_URL"] = _base_url
if _model:
    os.environ["GENAI_MODEL"] = _model


# =====================================================
# Helper: Extract text from uploaded files
# =====================================================

def extract_text_from_upload(uploaded_file):
    """Extract text from an uploaded PDF or TXT file."""
    if uploaded_file.type == "application/pdf":
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
            pages = []
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
            uploaded_file.seek(0)  # Reset for potential re-read
            return "\n".join(pages)
        except Exception as e:
            return ""
    elif uploaded_file.type == "text/plain":
        text = uploaded_file.read().decode("utf-8")
        uploaded_file.seek(0)
        return text
    elif uploaded_file.name.endswith(".docx"):
        try:
            import docx
            doc = docx.Document(io.BytesIO(uploaded_file.read()))
            uploaded_file.seek(0)
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception:
            return ""
    return ""


def get_score_class(score):
    """Return CSS class based on score."""
    if score >= 70:
        return "score-high"
    elif score >= 40:
        return "score-mid"
    return "score-low"


def get_score_emoji(score):
    """Return emoji based on score."""
    if score >= 70:
        return "🟢"
    elif score >= 40:
        return "🟡"
    return "🔴"


# =====================================================
# Main App
# =====================================================

st.title("🎯 AI Resume Screening Agent")
st.markdown(
    "Upload multiple resumes and a job description. "
    "The AI agent will analyze each resume, score them, and rank the best candidates."
)

st.divider()

# --- Two Column Layout ---
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    # --- Resume Upload ---
    st.subheader("📄 Step 1: Upload Resumes")
    st.caption("Upload all candidate resumes (PDF, TXT, or DOCX). You can select multiple files at once.")

    uploaded_files = st.file_uploader(
        "Upload resume files",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) uploaded")
        with st.expander("Preview uploaded files", expanded=False):
            for f in uploaded_files:
                text = extract_text_from_upload(f)
                word_count = len(text.split()) if text else 0
                status = "✅" if word_count >= 20 else "⚠️ too short"
                st.markdown(f"**{f.name}** — {word_count} words {status}")

with col_right:
    # --- Job Description ---
    st.subheader("📋 Step 2: Job Description")
    st.caption("Paste the full job description for the role you're hiring for.")

    job_description = st.text_area(
        "Paste the job description",
        height=250,
        placeholder="Example: We are looking for a Senior Python Developer with 5+ years experience in cloud platforms, microservices architecture, and DevOps practices...",
        label_visibility="collapsed"
    )

st.divider()

# --- Analyze Button ---
col_btn, col_info = st.columns([1, 2])
with col_btn:
    analyze_button = st.button("🚀 Screen All Resumes", type="primary", use_container_width=True)
with col_info:
    if uploaded_files and job_description:
        st.caption(f"Ready: {len(uploaded_files)} resume(s) + job description")
    elif not uploaded_files:
        st.caption("⬆️ Upload resumes to get started")
    else:
        st.caption("⬆️ Paste a job description")

# =====================================================
# Run Analysis
# =====================================================

if analyze_button:
    # Validate API config first
    if not _api_key or not _base_url or not _model:
        st.error("⚠️ Please configure your API credentials in the sidebar before running analysis.")
        st.stop()

    # Validation
    if not uploaded_files:
        st.warning("Please upload at least one resume.")
        st.stop()
    if not job_description or not job_description.strip():
        st.warning("Please paste a job description.")
        st.stop()
    if len(job_description.split()) < 10:
        st.warning("Job description seems too short. Please provide more details.")
        st.stop()

    # Extract text from all uploads
    resumes = []
    skipped = []
    for f in uploaded_files:
        text = extract_text_from_upload(f)
        if text.strip() and len(text.split()) >= 20:
            resumes.append({"filename": f.name, "text": text})
        else:
            skipped.append(f.name)

    if skipped:
        st.warning(f"Skipped {len(skipped)} file(s) (empty/unreadable): {', '.join(skipped)}")

    if not resumes:
        st.error("No valid resumes found. Please check your uploaded files.")
        st.stop()

    # Run the LangGraph agent
    st.divider()
    st.subheader("⚙️ Agent Processing")

    progress_container = st.empty()
    status_text = st.empty()

    with st.spinner(f"Screening {len(resumes)} resume(s)... This may take a minute."):
        try:
            result = run_screening(resumes, job_description)
        except Exception as e:
            st.error(f"Agent encountered an error: {str(e)}")
            st.error("Please check your API credentials in the sidebar (🔑).")
            st.stop()

    # Check for errors
    if result.get("error"):
        st.error(f"Agent error: {result['error']}")
        st.stop()

    # --- Show Progress Log ---
    with st.expander("📜 Agent Log", expanded=False):
        for log in result.get("progress_log", []):
            st.text(log)

    # --- Results ---
    evaluations = result.get("evaluations", [])
    sorted_evals = sorted(evaluations, key=lambda x: x.get("score", 0), reverse=True)

    st.divider()

    # =====================================================
    # Top Pick Card
    # =====================================================
    if sorted_evals:
        top = sorted_evals[0]
        st.subheader("🏆 Top Candidate")
        st.markdown(f"""
        <div class="top-pick-card">
            <h2 style="margin:0; color:white;">#{1} {top['filename']}</h2>
            <h3 style="margin:5px 0; color:rgba(255,255,255,0.9);">Score: {top.get('score', 'N/A')}/100</h3>
            <p style="color:rgba(255,255,255,0.85); font-size:1.05em;">{top.get('summary', '')}</p>
            <p style="color:rgba(255,255,255,0.8); font-style:italic;">{top.get('recommendation', '')}</p>
        </div>
        """, unsafe_allow_html=True)

    # =====================================================
    # All Candidates - Detailed Cards
    # =====================================================
    st.subheader("📊 All Candidates (Ranked)")

    for i, ev in enumerate(sorted_evals, 1):
        score = ev.get("score", 0)
        score_class = get_score_class(score)
        emoji = get_score_emoji(score)

        with st.container():
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"### {emoji} #{i} — {ev['filename']}")
            with c2:
                st.markdown(
                    f'<span class="score-badge {score_class}">{score}/100</span>',
                    unsafe_allow_html=True
                )

            st.markdown(f"**Summary:** {ev.get('summary', 'N/A')}")
            st.markdown(f"**Recommendation:** {ev.get('recommendation', 'N/A')}")

            col_s, col_g = st.columns(2)
            with col_s:
                strengths = ev.get("strengths", [])
                if strengths:
                    st.markdown("**✅ Strengths:**")
                    for s in strengths:
                        st.markdown(f"- {s}")
            with col_g:
                gaps = ev.get("gaps", [])
                if gaps:
                    st.markdown("**⚠️ Gaps:**")
                    for g in gaps:
                        st.markdown(f"- {g}")

            st.divider()

    # =====================================================
    # Final AI Ranking Report
    # =====================================================
    if len(sorted_evals) > 1 and result.get("final_ranking"):
        st.subheader("📝 AI Comparative Analysis")
        st.markdown(result["final_ranking"])

    # =====================================================
    # Download Report
    # =====================================================
    st.divider()
    st.subheader("💾 Export Results")

    # Build text report
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("RESUME SCREENING REPORT")
    report_lines.append("=" * 60)
    report_lines.append(f"\nJob Description:\n{job_description}\n")
    report_lines.append(f"Total Candidates Screened: {len(sorted_evals)}\n")
    report_lines.append("=" * 60)
    report_lines.append("INDIVIDUAL EVALUATIONS (Ranked by Score)")
    report_lines.append("=" * 60)

    for i, ev in enumerate(sorted_evals, 1):
        report_lines.append(f"\n{'─' * 50}")
        report_lines.append(f"#{i} | {ev['filename']} | Score: {ev.get('score', 'N/A')}/100")
        report_lines.append(f"{'─' * 50}")
        report_lines.append(f"Summary: {ev.get('summary', 'N/A')}")
        report_lines.append(f"Recommendation: {ev.get('recommendation', 'N/A')}")
        if ev.get("strengths"):
            report_lines.append("Strengths:")
            for s in ev["strengths"]:
                report_lines.append(f"  + {s}")
        if ev.get("gaps"):
            report_lines.append("Gaps:")
            for g in ev["gaps"]:
                report_lines.append(f"  - {g}")

    if result.get("final_ranking"):
        report_lines.append(f"\n{'=' * 60}")
        report_lines.append("COMPARATIVE ANALYSIS")
        report_lines.append(f"{'=' * 60}\n")
        report_lines.append(result["final_ranking"])

    report_text = "\n".join(report_lines)

    st.download_button(
        label="📥 Download Full Report (.txt)",
        data=report_text,
        file_name="resume_screening_report.txt",
        mime="text/plain",
        use_container_width=True
    )
