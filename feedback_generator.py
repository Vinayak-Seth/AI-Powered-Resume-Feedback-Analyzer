from groq import Groq
from dotenv import load_dotenv
import os
load_dotenv()

_client = None

def get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    return _client


def _ask_groq(prompt: str) -> str:
    client = get_client()
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",   # free tier model on Groq
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


def generate_overall_feedback(resume_text: str, job_description: str, similarity_score: float) -> str:
    score_pct = int(similarity_score * 100)
    prompt = f"""You are an expert ATS resume coach. A candidate's resume scores {score_pct}% match against this job description.

Job Description (first 600 chars):
{job_description[:600]}

Resume (first 600 chars):
{resume_text[:600]}

Give exactly 3 specific, numbered, actionable improvements the candidate should make to this resume to better match the job.
Be direct. No filler. Reference actual content from both documents."""
    return _ask_groq(prompt)


def generate_section_feedback(section_name: str, section_content: str, job_description: str) -> str:
    prompt = f"""You are a resume expert. Review the '{section_name}' section below against the job description.

Job Description (first 400 chars):
{job_description[:400]}

{section_name.title()} Section:
{section_content[:400]}

Give ONE specific, actionable improvement for this section. Be concrete — reference actual words or gaps."""
    return _ask_groq(prompt)


def generate_gap_advice(missing_skills: list, job_description: str) -> str:
    if not missing_skills:
        return "No critical skill gaps detected based on keyword analysis."
    skills_str = ", ".join(missing_skills[:12])
    prompt = f"""A job seeker's resume is missing these keywords found in the job description: {skills_str}

Job context: {job_description[:300]}

Give 2 practical, specific tips on how they can address these gaps — either by adding genuine experience they forgot to mention, or by building the missing skills quickly. No generic advice."""
    return _ask_groq(prompt)