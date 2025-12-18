import streamlit as st
from retrieval.retrieve_and_rank import recommend

st.set_page_config(
    page_title="SHL Assessment Recommendation Engine",
    layout="centered"
)

st.title("SHL Assessment Recommendation Engine")
st.write(
    "This tool recommends **SHL Individual Test Solutions** based on "
    "job descriptions or hiring requirements."
)

# ---------------- API MODE ----------------
query_param = st.query_params.get("query")

if query_param:
    results = recommend(query_param, k=10)
    st.json({
        "query": query_param,
        "recommendations": [
            {
                "rank": i + 1,
                "name": r["name"],
                "score": round(float(r["score"]), 4),
                "url": r["url"]
            }
            for i, r in enumerate(results)
        ]
    })
    st.stop()

# ---------------- UI MODE ----------------
st.subheader("Job Description / Hiring Query")

user_query = st.text_area(
    "Enter a JD or hiring query",
    height=180
)

if st.button("🔍 Recommend Assessments"):
    if not user_query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Finding best matching SHL assessments..."):
            results = recommend(user_query, k=10)

        for i, r in enumerate(results, 1):
            st.markdown(
                f"""
**{i}. {r['name']}**  
🔗 {r['url']}  
_Relevance score_: `{r['score']:.3f}`
"""
            )
