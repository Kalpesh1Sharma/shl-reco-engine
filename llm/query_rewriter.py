import os
import google.generativeai as genai

# Configure Gemini (API key via environment variable)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

MODEL_NAME = "gemini-pro"


def rewrite_query(query: str):
    """
    Optional LLM-based query rewriting.
    Fails silently and returns None if unavailable.
    """

    prompt = f"""
Rewrite the following job description or query into a SHORT, keyword-focused
search query containing:
- role
- key skills
- assessment intent

Rules:
- Do NOT add explanations
- Do NOT exceed 20 words
- Output only the rewritten query

Original query:
\"\"\"{query}\"\"\"
"""

    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.2,
                "max_output_tokens": 50,
            },
        )

        rewritten = response.text.strip()
        if rewritten:
            return rewritten

    except Exception:
        # Silent failure by design
        pass

    return None
