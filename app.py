import io
import re
import math
import base64
from typing import Tuple, Dict, List
import streamlit as st

from scoring import (
    extract_text_from_any,
    preprocess_text,
    extract_skills_from_text,
    compute_similarity_scores,
    ats_compliance_checks,
    build_report_pdf_bytes
)

st.set_page_config(page_title="ATS Resume Tracker", page_icon="📄", layout="wide")

# ------------------------
# Sidebar – App Info
# ------------------------
with st.sidebar:
    st.title("📄 ATS Resume Tracker")
    st.write(
        "Upload your **resume** and a **job description** to get:\n"
        "- Overall ATS Score\n"
        "- JD Match Percentage\n"
        "- Matched & Missing Skills\n"
        "- Formatting/ATS compliance tips\n"
    )
    st.caption("Built with Streamlit + scikit-learn (TF-IDF).")

# ------------------------
# Main – Inputs
# ------------------------
st.header("Upload & Analyze")

col1, col2 = st.columns(2, gap="large")

with col1:
    resume_file = st.file_uploader(
        "Upload Resume (PDF/DOCX/TXT)",
        type=["pdf", "docx", "txt"],
        help="Your resume will be processed locally."
    )

with col2:
    jd_source = st.radio("Job Description Input", ["Paste text", "Upload file"], horizontal=True)
    jd_text = ""
    jd_file = None
    if jd_source == "Paste text":
        jd_text = st.text_area(
            "Paste Job Description Text",
            height=220,
            placeholder="Paste JD here..."
        )
    else:
        jd_file = st.file_uploader(
            "Upload Job Description (PDF/DOCX/TXT)",
            type=["pdf", "docx", "txt"],
            key="jd_file"
        )

analyze_btn = st.button("Analyze Now 🚀", use_container_width=True)

# ------------------------
# Run Analysis
# ------------------------
if analyze_btn:
    if not resume_file:
        st.error("Please upload a resume.")
        st.stop()

    # Read resume text
    resume_text, resume_meta = extract_text_from_any(resume_file)

    # Read JD text
    if jd_source == "Paste text":
        if not jd_text.strip():
            st.error("Please paste a job description.")
            st.stop()
        jd_raw_text = jd_text
    else:
        if jd_file is None:
            st.error("Please upload a Job Description file.")
            st.stop()
        jd_raw_text, _ = extract_text_from_any(jd_file)

    # Preprocess
    resume_clean = preprocess_text(resume_text)
    jd_clean = preprocess_text(jd_raw_text)

    # Compute vector similarities + keyword overlap
    sim = compute_similarity_scores(resume_clean, jd_clean)

    # Skill extraction
    resume_skills = extract_skills_from_text(resume_clean)
    jd_skills = extract_skills_from_text(jd_clean)

    matched_skills = sorted(list(set(resume_skills).intersection(set(jd_skills))))
    missing_skills = sorted(list(set(jd_skills) - set(resume_skills)))

    # ATS compliance
    compliance = ats_compliance_checks(resume_text, resume_meta)

    # Aggregate scoring (weights)
    # Feel free to tune weights per your domain
    weight_skills = 0.45
    weight_semantic = 0.40
    weight_compliance = 0.15

    # Skills subscore: proportion of JD skills present
    skills_subscore = 0.0 if len(jd_skills) == 0 else len(matched_skills) / len(jd_skills)

    # Semantic subscore: cosine similarity (TF-IDF)
    semantic_subscore = sim["cosine_tfidf"]

    # Compliance subscore: average of booleans
    compliance_points = sum(1 for k, v in compliance["checks"].items() if v)
    compliance_possible = len(compliance["checks"])
    compliance_subscore = (compliance_points / compliance_possible) if compliance_possible else 0.0

    overall_score = (
        (skills_subscore * weight_skills) +
        (semantic_subscore * weight_semantic) +
        (compliance_subscore * weight_compliance)
    )
    overall_score_pct = round(overall_score * 100)

    # Match percentage – emphasize skills + TF-IDF equally
    match_pct = round(((skills_subscore + semantic_subscore) / 2) * 100)

    # ------------------------
    # Display Results
    # ------------------------
    st.success("Analysis complete ✅")

    m1, m2, m3 = st.columns([1, 1, 1], gap="large")
    with m1:
        st.metric("Overall ATS Score", f"{overall_score_pct} / 100")
    with m2:
        st.metric("JD Match %", f"{match_pct}%")
    with m3:
        st.metric("TF-IDF Similarity", f"{round(sim['cosine_tfidf']*100)}%")

    st.subheader("Skills Overview")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**Matched Skills**")
        if matched_skills:
            st.write(", ".join(matched_skills))
        else:
            st.write("_None detected._")

    with c2:
        st.markdown("**Missing (from JD)**")
        if missing_skills:
            st.write(", ".join(missing_skills))
        else:
            st.write("_No missing skills detected — great!_")

    # Simple bar chart
    st.markdown("### Skill Match Chart")
    chart_data = {
        "Category": ["Matched", "Missing"],
        "Count": [len(matched_skills), len(missing_skills)]
    }
    st.bar_chart(chart_data, x="Category", y="Count")

    # Compliance
    st.subheader("ATS Compliance Check")
    cc1, cc2 = st.columns([1.2, 1], gap="large")
    with cc1:
        for label, value in compliance["checks"].items():
            st.write(f"- {'✅' if value else '❌'} {label}")

        if compliance["notes"]:
            st.markdown("**Notes / Suggestions:**")
            for n in compliance["notes"]:
                st.write(f"- {n}")

    with cc2:
        st.caption("File Info")
        st.json(resume_meta)

    # Suggestions
    st.subheader("Suggestions to Improve")
    suggestions: List[str] = []

    if missing_skills:
        suggestions.append(
            f"Add these JD-aligned skills (where truthful): {', '.join(missing_skills[:15])}"
            + (" ..." if len(missing_skills) > 15 else "")
        )

    if not compliance["checks"].get("Has clear section headings (e.g., Experience, Education, Skills)", False):
        suggestions.append("Add standard, ATS-friendly headings: **Experience**, **Education**, **Skills**, **Projects**.")

    if not compliance["checks"].get("Contains contact info (email/phone/linkedin)", False):
        suggestions.append("Include **email**, **phone**, and optionally **LinkedIn/GitHub** in top section.")

    if not compliance["checks"].get("Avoids complex formatting (tables/multiple columns)", False):
        suggestions.append("Avoid **tables**, **text boxes**, or **multi-column** layouts that can confuse ATS.")

    if compliance["checks"].get("File format is ATS-friendly", True) is False:
        suggestions.append("Use **PDF (text-based)** or **DOCX** exported from Word/Google Docs (not scanned images).")

    if not suggestions:
        st.write("Looks good! Minor tweaks only.")
    else:
        for s in suggestions:
            st.write(f"- {s}")

    # ------------------------
    # Export Report (PDF)
    # ------------------------
    st.markdown("---")
    st.markdown("### Export")
    if st.button("Generate PDF Report"):
        pdf_bytes = build_report_pdf_bytes(
            overall_score_pct,
            match_pct,
            sim,
            matched_skills,
            missing_skills,
            compliance,
        )
        b64 = base64.b64encode(pdf_bytes).decode("utf-8")
        href = f'<a href="data:application/pdf;base64,{b64}" download="ATS_Resume_Report.pdf">📥 Download ATS_Resume_Report.pdf</a>'
        st.markdown(href, unsafe_allow_html=True)
