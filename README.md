# 📄 AI-Powered Resume Feedback Assistant

An intelligent resume evaluation system that analyses resumes against job descriptions using NLP, semantic similarity, and AI-generated feedback — deployed as an interactive web app.

---

## 🚀 Live Demo

[Click here to try it →](https://resume-feedback-2zrebrgycsytrmslmpwdux.streamlit.app/)

---

## 🧠 What It Does

- Upload your resume as a PDF
- Paste any job description
- Get an instant ATS-style match score, keyword gap analysis, and section-wise relevance breakdown
- Receive AI-generated, personalised improvement suggestions powered by Llama 3 via Groq

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core language |
| Streamlit | Interactive web app |
| Sentence Transformers (`all-MiniLM-L6-v2`) | Semantic similarity scoring |
| Groq API (Llama 3.1) | AI feedback generation |
| PyPDF2 | PDF text extraction |
| Scikit-Learn | Supporting NLP utilities |

---

## ⚙️ How It Works

1. **PDF Parsing** — Extracts and preprocesses text from uploaded resumes, detects sections (Skills, Experience, Education, Projects)
2. **Semantic Matching** — Generates sentence embeddings for both resume and job description, computes cosine similarity score
3. **Keyword Analysis** — Synonym-normalised bigram extraction identifies matched and missing keywords between resume and JD
4. **AI Feedback** — Sends context to Llama 3.1 via Groq API, returns 3 specific actionable improvements
5. **Section Scoring** — Scores each resume section independently against the JD to pinpoint weakest areas

---

## 📁 Project Structure

```
resume_feedback/
├── app.py                  # Streamlit UI
├── resume_parser.py        # PDF extraction and section detection
├── matcher.py              # Similarity scoring and keyword gap analysis
├── feedback_generator.py   # Groq API integration for AI feedback
├── requirements.txt        # Dependencies
├── .env.example            # Environment variable template
└── .gitignore
```

---

## 🔧 Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/yourusername/resume-feedback.git
cd resume-feedback
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set up your API key**
```bash
cp .env.example .env
```
Open `.env` and add your free Groq API key (get one at [console.groq.com](https://console.groq.com) — no credit card needed):
```
GROQ_API_KEY=your_key_here
```

**4. Run the app**
```bash
streamlit run app.py
```

---

## 📊 Features

- ✅ ATS match score (0–100%)
- ✅ Matched and missing keyword tags
- ✅ Section-wise relevance bars (Skills, Experience, Education, Projects)
- ✅ 3 specific AI-generated resume improvements
- ✅ Skill gap advice
- ✅ Per-section AI feedback (expandable)

---

## 🔑 Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Free API key from [console.groq.com](https://console.groq.com) |

---

## 📌 Notes

- Works best with text-based PDFs (not scanned images)
- First run downloads the Sentence Transformer model (~90MB, cached after)
- Free Groq tier supports 30 requests/minute — sufficient for personal use

---

## 👤 Author

**Vinayak**  
[LinkedIn](https://www.linkedin.com/in/vinayak-seth2005/) · [GitHub](https://github.com/Vinayak-Seth)
