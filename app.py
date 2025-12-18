import streamlit as st
import sys
import os

# --------------------------------------------------
# Ensure project root is on PYTHONPATH
# --------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from retrieval.retrieve_and_rank import recommend

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.set_page_config(
    page_title="SHL Assessment Recommendation Engine",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 SHL Assessment Recommendation Engine")
st.markdown(
    """
This tool recommends **SHL Individual Test Solutions** based on  
job descriptions or hiring requirements.

Enter a **JD / hiring query** below and click **Recommend**.
"""
)

query = st.text_area(
    "Job Description / Hiring Query",
    height=200,
    placeholder="e.g. Looking to hire a data analyst with strong SQL, Excel, and analytical skills..."
)

if st.button("🔍 Recommend Assessments"):
    if not query.strip():
        st.warning("Please enter a job description or query.")
    else:
        with st.spinner("Finding best assessments..."):
            results = recommend(query, k=10)

        st.success("Top Recommended Assessments")

        for i, r in enumerate(results, 1):
            st.markdown(
                f"""
**{i}. {r['name']}**  
🔗 {r['url']}
"""
            )

st.markdown("---")
st.caption("Built for SHL AI Intern Assignment")
