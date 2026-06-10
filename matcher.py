from sentence_transformers import SentenceTransformer, util
import re
import torch

# ── Model singleton ────────────────────────────────────────────────────────
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model


# ── Tech synonym map: maps variants → canonical form ──────────────────────
# This handles "ML" == "machine learning", "js" == "javascript", etc.
SYNONYMS = {
    # Languages
    "js": "javascript", "ts": "typescript", "py": "python",
    "golang": "go", "c#": "csharp", "c++": "cpp",
    # ML/AI
    "ml": "machine learning", "dl": "deep learning",
    "nlp": "natural language processing", "cv": "computer vision",
    "llm": "large language models", "genai": "generative ai",
    "ai": "artificial intelligence",
    # Web
    "reactjs": "react", "react.js": "react", "vuejs": "vue",
    "node.js": "nodejs", "node": "nodejs",
    "nextjs": "next", "next.js": "next",
    # Cloud
    "gcp": "google cloud", "aws": "amazon web services",
    "k8s": "kubernetes",
    # DB
    "mongo": "mongodb", "postgres": "postgresql",
    # Data
    "sklearn": "scikit-learn", "scikit learn": "scikit-learn",
    "tf": "tensorflow", "pytorch": "torch",
    # Methodologies
    "ci/cd": "cicd", "rest api": "restapi", "rest": "restapi",
    "oop": "object oriented programming",
    "agile/scrum": "agile",
}


def normalize_token(token: str) -> str:
    """Lowercase, strip punctuation, map through synonym table."""
    t = token.lower().strip()
    t = re.sub(r'[^\w\+\#\.]', '', t)
    return SYNONYMS.get(t, t)


def extract_keywords(text: str) -> set:
    """
    Extract meaningful keywords from text with:
    1. Bigram extraction (captures 'machine learning', 'data science')
    2. Synonym normalization
    3. Stopword filtering
    """
    stopwords = {
        'the', 'and', 'for', 'with', 'that', 'have', 'this', 'will', 'are',
        'from', 'your', 'not', 'but', 'can', 'all', 'was', 'has', 'been',
        'they', 'our', 'you', 'their', 'more', 'also', 'into', 'its', 'who',
        'use', 'using', 'work', 'team', 'strong', 'good', 'well', 'able',
        'experience', 'working', 'years', 'required', 'must', 'etc', 'such',
        'candidate', 'role', 'position', 'responsibilities', 'opportunity',
        'please', 'apply', 'join', 'help', 'new', 'great', 'company',
    }

    # Clean text
    text_clean = re.sub(r'[^\w\s\+\#\./]', ' ', text.lower())
    tokens = text_clean.split()

    keywords = set()

    # Unigrams
    for t in tokens:
        normalized = normalize_token(t)
        if len(normalized) > 2 and normalized not in stopwords:
            keywords.add(normalized)

    # Bigrams (catches "machine learning", "data engineering", etc.)
    for i in range(len(tokens) - 1):
        bigram = f"{tokens[i]} {tokens[i+1]}"
        normalized = normalize_token(bigram)
        # Also try normalizing the bigram directly
        if bigram in SYNONYMS:
            keywords.add(SYNONYMS[bigram])
        # Keep bigrams that look like tech phrases (both words > 3 chars, neither a stopword)
        w1, w2 = tokens[i].lower(), tokens[i+1].lower()
        if len(w1) > 3 and len(w2) > 3 and w1 not in stopwords and w2 not in stopwords:
            keywords.add(f"{normalize_token(w1)} {normalize_token(w2)}")

    return keywords


def find_skill_gaps(resume_text: str, job_description: str) -> dict:
    """
    Identify gaps and matches using normalized + synonym-aware keyword matching.
    Returns {'matched': [...], 'missing': [...]}
    """
    resume_kw = extract_keywords(resume_text)
    jd_kw = extract_keywords(job_description)

    matched = sorted(resume_kw & jd_kw)
    missing = sorted(jd_kw - resume_kw)

    # Filter missing: only terms length > 3, remove noise
    missing = [m for m in missing if len(m) > 3]

    return {
        "matched": matched[:35],
        "missing": missing[:35],
    }


def compute_similarity(resume_text: str, job_description: str) -> float:
    """Cosine similarity between full resume and JD embeddings."""
    model = get_model()
    embeddings = model.encode([resume_text, job_description], convert_to_tensor=True)
    score = util.cos_sim(embeddings[0], embeddings[1]).item()
    return round(score, 4)


def section_wise_similarity(resume_sections: dict, job_description: str) -> dict:
    """Per-section similarity scores against the JD."""
    model = get_model()
    jd_embedding = model.encode(job_description, convert_to_tensor=True)

    scores = {}
    for section, content in resume_sections.items():
        if not content or len(content) < 20:
            continue
        sec_embedding = model.encode(content, convert_to_tensor=True)
        score = util.cos_sim(sec_embedding, jd_embedding).item()
        scores[section] = round(score, 4)

    return scores