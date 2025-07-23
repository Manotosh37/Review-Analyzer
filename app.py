import streamlit as st, pickle
import pandas as pd 
import torch 
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

@st.cache_resource
def load_bert_model():
    return pipeline("sentiment-analysis",
                    model= "nlptown/bert-base-multilingual-uncased-sentiment",
                    return_all_scores=True)

sentiment_analyzer = load_bert_model()

st.title("Sentiment Analysis")
st.markdown("**Enter lines** or **upload a CSV** for analysis.")

multi = st.text_area("Enter Reviews:", height=150)
uploaded = st.file_uploader("Or upload CSV file", type=["csv"] )

texts = []
if multi:
    texts += [l for l in multi.splitlines() if l.strip()]
if uploaded:
    df = pd.read_csv(uploaded)
    if "review" in df.columns:
        texts += df["review"].astype(str).tolist()
    else:
        st.error("CSV needs a 'review' column")
        st.stop()

if not texts:
    st.info("Add text lines or upload a CSV to get started.")
    st.stop()

with st.spinner("Analyzing with BERT..."):
    results = sentiment_analyzer(texts)
    
processed_results = []
for i, text in enumerate(texts):
    scores = {scores['label']: scores['score'] for score in results[i]}
    best_label = max(scores, key=scores.get)
    confidence = scores[best_label]
    
    processed_results.append({
        "review": text,
        "sentiment": best_label,
        "confidence": f"{confidence:.2f}",
        **{f"{label}_score": f"{scores:.2f}" for label, scores in scores.items()}
    })

result_df = pd.DataFrame({"review": texts, "sentiment": labels})
st.dataframe(result_df)

sentiment_count = result_df['sentiment'].value_counts()
st.subheader("Sentiment Distribution")
st.bar_chart(sentiment_count)