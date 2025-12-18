# SHL Assessment Recommendation Engine

This repository contains an **AI-powered assessment recommendation system** built for the **SHL AI Intern – Generative AI Assignment**.

The system recommends the most relevant **SHL Individual Test Solutions** based on a given **job description or hiring query**, using semantic search and optional LLM-based query enhancement.

---

## 🚀 Live Demo (Frontend + API)

🔗 **Streamlit App URL**: *(add your deployed Streamlit URL here)*

The Streamlit application serves as:
- A **web frontend** for evaluation
- An **API-like interface** to test recommendations interactively

---

## 📌 Problem Statement

Given a **job description or hiring requirement**, recommend the **top 10 SHL Individual Test Solutions** that best match the role.

**Key constraints:**
- Only **Individual Test Solutions** (no job solutions)
- Catalogue size ≥ **377 assessments**
- System must be explainable, scalable, and reproducible

---

## 🧠 Solution Overview

The system uses a **semantic retrieval pipeline** built on top of vector embeddings and fast similarity search.

High-level flow:

Job Description / Query
↓
SentenceTransformers Embeddings
↓
FAISS Vector Index
↓
Hybrid Ranking (Semantic + Keyword Overlap)
↓
(Optional) LLM-based Query Rewriting (Gemini, silent fallback)
↓
Top-10 SHL Assessment Recommendations


---

## 🔍 Key Features

- Semantic search using **SentenceTransformers**
- Fast vector similarity search using **FAISS**
- Hybrid ranking for better relevance
- Optional **LLM-based query rewriting**
- Graceful fallback when LLM is unavailable
- Simple and interactive **Streamlit frontend**

---

## 🧪 Evaluation

- Performance evaluated using **Recall@10**
- Used the provided **labeled training dataset**
- Iterative improvements included:
  - Query expansion
  - Slug-based matching
  - Hybrid ranking logic

---

## 📂 Project Structure

shl-reco-engine/
│
├── app.py                      # Streamlit frontend (UI + API)
├── requirements.txt
├── README.md
│
├── data/
│   └── shl_catalog_raw.json    # SHL catalogue snapshot (377 assessments)
│
├── embeddings/
│   ├── embedding_utils.py      # SentenceTransformers embeddings
│   ├── build_faiss_index.py    # FAISS index builder
│   ├── faiss_index/
│   │   ├── index.faiss         # Vector index
│   │   └── metadata.json       # Assessment metadata
│   ├── gemini_embedding_utils.py       # Deprecated / experimental
│   └── build_faiss_index_gemini.py     # Deprecated / experimental
│
├── retrieval/
│   ├── retrieve_and_rank.py    # Retrieval + ranking logic
│   └── rank_utils.py           # Balanced re-ranking utilities
│
├── llm/
│   └── query_rewriter.py       # Optional Gemini query rewriting
│
├── evaluation/
│   └── evaluate_recall.py      # Recall@10 evaluation
│
└── submission/
    └── shl_test_predictions.csv



---

## 🤖 LLM Usage (Design Decision)

- **SentenceTransformers** are used for bulk embedding and indexing due to:
  - Free, offline, scalable embeddings
  - Deterministic and reproducible results

- **Gemini LLM** is used **optionally at query time** for:
  - Rewriting long or unstructured job descriptions
  - Improving semantic focus of queries

The LLM integration is:
- Optional
- Limited to **1 API call per query**
- Designed to **fail silently and fall back** to the baseline system

This mirrors real-world production trade-offs between cost, scalability, and performance.

---

## ▶️ How to Run Locally

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
python embeddings/build_faiss_index.py
python embeddings/build_faiss_index.py

📈 Future Improvements

Fine-tuning embeddings on SHL-specific text

Skill taxonomy and assessment alias mapping

LLM-based re-ranking of top candidates

Improved metadata-based filtering (duration, test type)

👤 Author

Kalpesh Sharma
SHL AI Intern Assignment Submission

📜 License

This project is created for evaluation purposes as part of the SHL AI Intern assignment.
