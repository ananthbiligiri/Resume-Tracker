import io
import re
from typing import Tuple, Dict, List

# File parsing
import PyPDF2
import docx2txt

# NLP / ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Report export
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

# ------------------------
# Common skill dictionary (expand as needed)
# ------------------------
COMMON_SKILLS = {
    # Programming
    "python", "java", "c", "c++", "c#", "javascript", "typescript", "go", "rust",
    "html", "css", "sql", "r", "matlab",

    # Data / ML
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras",
    "nlp", "opencv", "data analysis", "data visualization",
    "machine learning", "deep learning", "etl", "airflow",

    # BI / Analytics
    "power bi", "tableau", "excel", "dax", "power query",

    # Web / Backend
    "react", "node", "express", "django", "flask", "fastapi",
    "rest api", "graphql",

    # Cloud / DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "git", "ci/cd", "linux",

    # Extras
    "yolov8", "opencv", "streamlit", "postman", "jira", "spark", "hadoop"
}

SECTION_HEADINGS = [
    "experience", "work experience", "professional experience",
    "education", "skills", "projects", "certifications",
    "summary", "objective", "achievements", "publications"
]

EMAIL_REGEX = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
PHONE_REGEX = r"(\+?\d{1,3}[\s-]?)?(\(?\d{3,5}\)?[\s-]?)?\d{3,4}[\s-]?\d{4}"
LINK_REGEX = r"(linkedin\.com\/in\/|github\.com\/|gitlab\.com\/|portfolio|\.dev|\.io)"

# ------------------------
# Text Extraction
# ------------------------
def extract_text_from_pdf(file_bytes: io.BytesIO) -> str:
    reader = PyPDF2.PdfReader(file_bytes)
    text = []
    for page in reader.pages:
        try:
            text.append(page.extract_text() or "")
        except Exception:
            text.append("")
    return "\n".join(text)


def extract_text_from_docx(file_bytes: io.BytesIO) -> str:
    # docx2txt expects a file path or file-like; it can read file-like via docx2txt.process
    # but simplest is to write to temp – Streamlit runs ephemeral so we try in-memory
    # Fallback: save to buffer
    # However, docx2txt.process() accepts path only. We'll write to temp file.
    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(file_bytes.read())
        tmp.flush()
        path = tmp.name
    text = docx2txt.process(path) or ""
    try:
        os.remove(path)
    except Exception:
        pass
    return text


def extract_text_from_txt(file_bytes: io.BytesIO) -> str:
    try:
        return file_bytes.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""


def extract_text_from_any(upload) -> Tuple[str, Dict]:
    """
    Returns:
        text (str), meta (dict)
    """
    filename = upload.name.lower()
    meta = {
        "filename": upload.name,
        "mimetype": upload.type,
        "size_bytes": upload.size
    }

    bytes_data = upload.read()

    if filename.endswith(".pdf"):
        text = extract_text_from_pdf(io.BytesIO(bytes_data))
        meta["filetype"] = "pdf"
    elif filename.endswith(".docx"):
        text = extract_text_from_docx(io.BytesIO(bytes_data))
        meta["filetype"] = "docx"
    elif filename.endswith(".txt"):
        text = extract_text_from_txt(io.BytesIO(bytes_data))
        meta["filetype"] = "txt"
    else:
        # Attempt best effort
        text = extract_text_from_txt(io.BytesIO(bytes_data))
        meta["filetype"] = "unknown"

    # Very short text likely indicates a scanned PDF or image-based resume
    meta["suspected_scanned"] = (len(text.strip()) < 200)

    return text, meta

# ------------------------
# Preprocess
# ------------------------
def preprocess_text(text: str) -> str:
    text = text.lower()
    # normalize bullets/dashes
    text = text.replace("•", " ").replace("–", "-").replace("—", "-")
    # remove non-alphanumerics (keep basic punctuation for keyword density)
    text = re.sub(r"[^a-z0-9\s\.\-,/+#]", " ", text)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ------------------------
# Keyword / Skills
# ------------------------
def extract_skills_from_text(text: str) -> List[str]:
    """
    Heuristic: look for known skills and common bigrams unrolled from skills set.
    """
    found = set()
    txt = " " + text + " "

    # unigram match
    for s in COMMON_SKILLS:
        # ensure we match full tokens
        s_norm = s.lower().strip()
        # allow spaces in skills (bigrams)
        if s_norm in txt:
            # stronger check: regex word boundaries for single tokens
            if " " in s_norm:
                found.add(s_norm)
            else:
                if re.search(rf"\b{s_norm}\b", txt):
                    found.add(s_norm)

    # Also capture capitalized tech names that might not be in the list (simple heuristic)
    caps = set(re.findall(r"\b([A-Z][A-Za-z0-9\+\#]{2,})\b", text))
    # filter noise
    caps = {c.lower() for c in caps if c.lower() not in {"i", "ii", "iii"} and len(c) > 2}
    found |= caps

    # normalize
    return sorted({f.strip() for f in found})

# ------------------------
# Similarity
# ------------------------
def compute_similarity_scores(resume_clean: str, jd_clean: str) -> Dict[str, float]:
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    tfidf = vectorizer.fit_transform([jd_clean, resume_clean])
    # cosine between JD (row 0) and Resume (row 1)
    cos = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return {"cosine_tfidf": float(cos)}

# ------------------------
# Compliance Checks
# ------------------------
def has_section_headings(text: str) -> bool:
    ltxt = text.lower()
    return any(h in ltxt for h in SECTION_HEADINGS)

def has_contact_info(text: str) -> bool:
    if re.search(EMAIL_REGEX, text): 
        return True
    if re.search(PHONE_REGEX, text): 
        return True
    if re.search(LINK_REGEX, text.lower()): 
        return True
    return False

def avoids_complex_format(meta: Dict) -> bool:
    """
    We can't fully inspect layout without a full parser; use heuristics:
    - If PDF is suspected scanned (very short text), it's bad for ATS.
    - Prefer docx/txt or text-based PDF.
    """
    if meta.get("suspected_scanned", False):
        return False
    return True

def is_ats_friendly_format(meta: Dict) -> bool:
    f = meta.get("filetype", "")
    return f in {"pdf", "docx", "txt"}

def ats_compliance_checks(text: str, meta: Dict) -> Dict:
    checks = {
        "Has clear section headings (e.g., Experience, Education, Skills)": has_section_headings(text),
        "Contains contact info (email/phone/linkedin)": has_contact_info(text),
        "Avoids complex formatting (tables/multiple columns)": avoids_complex_format(meta),
        "File format is ATS-friendly": is_ats_friendly_format(meta),
        "Reasonable text length (not empty)": len(text.strip()) > 200
    }

    notes = []
    if meta.get("suspected_scanned", False):
        notes.append("Resume text appears very short; PDF may be scanned/image-based. Export a text-based PDF or DOCX.")
    if not checks["Has clear section headings (e.g., Experience, Education, Skills)"]:
        notes.append("Use standard headings: Experience, Education, Skills, Projects.")
    if not checks["Contains contact info (email/phone/linkedin)"]:
        notes.append("Add contact info at the top: email, phone, LinkedIn/GitHub.")
    if not checks["Avoids complex formatting (tables/multiple columns)"]:
        notes.append("Avoid tables, text boxes, and multi-column layouts that can confuse ATS.")
    if not checks["File format is ATS-friendly"]:
        notes.append("Use PDF (text-based) or DOCX.")
    if not checks["Reasonable text length (not empty)"]:
        notes.append("Content looks too short to evaluate properly.")

    return {"checks": checks, "notes": notes}

# ------------------------
# PDF Report
# ------------------------
def build_report_pdf_bytes(
    overall_score_pct: int,
    match_pct: int,
    sim: Dict[str, float],
    matched_skills: List[str],
    missing_skills: List[str],
    compliance: Dict
) -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    y = height - 2 * cm
    def line(txt, size=12, leading=14):
        nonlocal y
        c.setFont("Helvetica", size)
        c.drawString(2*cm, y, txt[:1100])
        y -= leading
        if y < 2*cm:
            c.showPage()
            y = height - 2*cm

    c.setTitle("ATS Resume Report")

    line("ATS Resume Report", 18, 22)
    line(f"Overall ATS Score: {overall_score_pct}/100", 14, 18)
    line(f"JD Match Percentage: {match_pct}%", 14, 18)
    line(f"TF-IDF Similarity: {round(sim['cosine_tfidf']*100)}%", 12, 16)
    line("")

    line("Matched Skills:", 14, 18)
    if matched_skills:
        for s in matched_skills:
            line(f"  • {s}")
    else:
        line("  (None)")

    line("")
    line("Missing Skills (from JD):", 14, 18)
    if missing_skills:
        for s in missing_skills:
            line(f"  • {s}")
    else:
        line("  (None)")

    line("")
    line("ATS Compliance:", 14, 18)
    for k, v in compliance["checks"].items():
        line(f"  {'[OK] ' if v else '[X]  '}{k}")

    if compliance["notes"]:
        line("")
        line("Notes / Suggestions:", 14, 18)
        for n in compliance["notes"]:
            line(f"  • {n}")

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.read()
