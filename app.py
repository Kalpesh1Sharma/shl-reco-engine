import os
import sys
import streamlit as st

# --------------------------------
# Fix Python path for Streamlit Cloud
# --------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from retrieval.retrieve_and_rank import recommend

# --------------------------------
# Streamlit Page Config
# --------------------------------
st.set_page_config(
    page_title="SHL Assessment Recommendation Engine",
    layout="centered"
)

st.title("SHL Assessment Recommendation Engine")
st.write(
    "This tool recommends **SHL Individual Test Solutions** based on "
    "job descriptions or hiring requirements."
)

# --------------------------------
# 🔌 API MODE — JSON RESPONSE
# --------------------------------
query_param = st.query_params.get("query")

if query_param:
    query_text = query_param.strip()

    results = recommend(query_text, k=10)

    response = {
        "query": query_text,
        "recommendations": [
            {
                "rank": idx + 1,
                "name": r["name"],
                "score": round(float(r["score"]), 4),
                "url": r["url"]
            }
            for idx, r in enumerate(results)
        ]
    }

    st.json(response)
    st.stop()

# --------------------------------
# 🖥️ UI MODE
# --------------------------------
st.subheader("Job Description / Hiring Query")

user_query = st.text_area(
    "Enter a JD or hiring query below and click Recommend",
    height=180,
    placeholder="Looking to hire mid-level professionals proficient in Python, SQL, and JavaScript..."
)

if st.button("🔍 Recommend Assessments"):
    if not user_query.strip():
        st.warning("Please enter a job description or hiring query.")
    else:
        with st.spinner("Finding best matching SHL assessments..."):
            recommendations = recommend(user_query, k=10)

        st.success("Top Recommended Assessments")

        for idx, rec in enumerate(recommendations, start=1):
            st.markdown(
                f"""
**{idx}. {rec['name']}**  
🔗 {rec['url']}  
_Relevance score_: `{rec['score']:.3f}`
"""
            )

st.caption(
    "Recommendations are generated using semantic similarity over "
    "SHL Individual Test Solutions."
)
