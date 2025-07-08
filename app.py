import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Philosopher Quote Finder", page_icon="🧠")

@st.cache_resource
def load_resources():
    df = pd.read_csv("philosopher_quotes_top3_labeled.csv")
    model = joblib.load("svm_top100_model.pkl")
    mlb = joblib.load("label_encoder.pkl")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return df, model, mlb, embedder

df, svm_model, mlb, embedder = load_resources()

st.title("🧠 Philosopher Quote Finder")
st.write("Describe how you feel or what you’re going through, and get wisdom from history’s greatest minds.")

user_input = st.text_input("What are you feeling or searching for?", "")

try:
    if user_input:
        input_clean = user_input.lower().strip()
        known_tags = set(mlb.classes_)

        if input_clean in known_tags:
            matched_df = df[df["predicted_tags"].apply(lambda tags: input_clean in tags)]
            st.success(f"Found {len(matched_df)} quotes with tag: '{input_clean}'")
        else:
            emb = embedder.encode([user_input])
            scores = svm_model.decision_function(emb)
            top_indices = np.argsort(scores[0])[-3:]
            predicted = mlb.inverse_transform(np.array([[1 if i in top_indices else 0 for i in range(len(scores[0]))]]))[0]

            if predicted:
                st.info(f"Predicted related tags: {', '.join(predicted)}")
                matched_df = df[df["predicted_tags"].apply(lambda tags: any(tag in tags for tag in predicted))]
            else:
                matched_df = pd.DataFrame()

        if matched_df.empty:
            st.warning("No matching quotes found. Try a different word.")
        else:
            st.subheader("✨ Recommended Quotes")
            quote_vecs = embedder.encode(matched_df["quote"].tolist(), batch_size=64)
            sim_scores = np.dot(quote_vecs, emb.T).flatten()
            matched_df["similarity"] = sim_scores
            top_quotes = matched_df.sort_values("similarity", ascending=False).head(5)

            for _, row in top_quotes.iterrows():
                st.markdown(f"> *{row['quote']}* — **{row['author']}**")
except Exception as e:
    st.error("Something went wrong.")
    st.exception(e)
