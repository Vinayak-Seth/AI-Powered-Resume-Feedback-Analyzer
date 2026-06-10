import re
import PyPDF2
import io


def extract_text_from_pdf(uploaded_file) -> str:
    """Extract raw text from an uploaded PDF file object (Streamlit UploadedFile)."""
    reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def preprocess_text(text: str) -> str:
    """Clean and normalize extracted text."""
    # Remove excessive whitespace and blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    # Remove non-printable characters
    text = re.sub(r'[^\x20-\x7E\n]', '', text)
    return text.strip()


def extract_sections(text: str) -> dict:
    """
    Heuristically extract common resume sections.
    Returns a dict with keys like 'skills', 'experience', 'education', 'summary'.
    """
    section_headers = {
        "summary": r"(summary|objective|profile|about)",
        "experience": r"(experience|work history|employment|professional background)",
        "education": r"(education|academic|qualification)",
        "skills": r"(skills|technical skills|core competencies|technologies)",
        "projects": r"(projects|personal projects|key projects)",
        "certifications": r"(certifications|certificates|licenses)",
    }

    sections = {}
    lines = text.split('\n')
    current_section = "other"
    buffer = []

    for line in lines:
        stripped = line.strip()
        matched_section = None

        for section, pattern in section_headers.items():
            if re.match(rf'^{pattern}', stripped, re.IGNORECASE) and len(stripped) < 60:
                matched_section = section
                break

        if matched_section:
            if buffer:
                sections[current_section] = '\n'.join(buffer).strip()
            current_section = matched_section
            buffer = []
        else:
            buffer.append(line)

    if buffer:
        sections[current_section] = '\n'.join(buffer).strip()

    return sections


def extract_skills_list(skills_text: str) -> list:
    """Parse comma/bullet separated skills into a clean list."""
    # Split on commas, bullets, pipes, newlines
    raw = re.split(r'[,•|\n/]+', skills_text)
    skills = [s.strip() for s in raw if len(s.strip()) > 1]
    return skills