import streamlit as st
import pandas as pd
import numpy as np
import faiss
import joblib
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

st.set_page_config(page_title="MiniLM-Powered Quote Finder", page_icon="🧠")

@st.cache_resource
def load_resources():
    df = pd.read_csv("philosopher_quotes_top3_labeled.csv")
    embedder = SentenceTransformer("minilm-pos-neg")
    quotes = df["quote"].tolist()
    authors = df["author"].tolist()
    embeddings = embedder.encode(quotes, batch_size=64, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return df, embedder, index, quotes, authors
    # model = joblib.load("svm_top100_model.pkl")
    # mlb = joblib.load("label_encoder.pkl")
    # embedder = SentenceTransformer("all-MiniLM-L6-v2")
    #return df, model, mlb, embedder

#df, svm_model, mlb, embedder = load_resources()
df, embedder, faiss_index, quotes, authors = load_resources()

st.title("🧠 MiniLM-Powered Quote Finder")
st.write("Describe how you feel or what you’re going through, and get wisdom from history’s greatest minds.")

authors = sorted(df["author"].unique().tolist())
selected_authors = st.multiselect("🔍 Filter by author:", authors)
unique_authors_only = st.checkbox("Show quotes from different authors only", value=True)

user_input = st.text_input("What are you feeling or searching for?", "")

def retrieve_quotes(text: str, k: int = 5) -> List[Tuple[str, str, float]]:
    emb = embedder.encode([text], convert_to_numpy=True)
    faiss.normalize_L2(emb)
    D, I = faiss_index.search(emb, k)
    return [(quotes[i], authors[i], float(D[0][j])) for j, i in enumerate(I[0])]

try:
    if user_input:
        input_clean = user_input.lower().strip()

        st.subheader("✨ Recommended Quotes")

        emb = embedder.encode([user_input], convert_to_numpy=True)
        faiss.normalize_L2(emb)

        k = min(20, faiss_index.ntotal)
        D, I = faiss_index.search(emb, k)  
        results = []
        for score, idx in zip(D[0], I[0]):
            if 0 <= idx < len(quotes):
                results.append({
                    "quote": quotes[idx],
                    "author": authors[idx],
                    "similarity": float(score)
                })

        results_df = pd.DataFrame(results)

        if selected_authors:
            results_df = results_df[results_df["author"].isin(selected_authors)]

        if unique_authors_only:
            results_df = results_df.drop_duplicates(subset="author")

        if results_df.empty:
            st.warning("No matching quotes found. Try a different word or author.")
        else:
            top_quotes = results_df.sort_values("similarity", ascending=False).head(5)
            for _, row in top_quotes.iterrows():
                st.markdown(f"> *{row['quote']}* — **{row['author']}**")

except Exception as e:
    st.error("Something went wrong.")
    st.exception(e)


# try:
#     if user_input:
#         input_clean = user_input.lower().strip()
#         known_tags = set(mlb.classes_)
#         emb = embedder.encode([user_input])
#         if input_clean in known_tags:
#             matched_df = df[df["predicted_tags"].apply(lambda tags: input_clean in tags)]
#             st.success(f"Found {len(matched_df)} quotes with tag: '{input_clean}'")
#         else:
#             scores = svm_model.decision_function(emb)
#             top_indices = np.argsort(scores[0])[-3:]
#             predicted = mlb.inverse_transform(np.array([[1 if i in top_indices else 0 for i in range(len(scores[0]))]]))[0]

#             if predicted:
#                 st.info(f"Predicted related tags: {', '.join(predicted)}")
#                 matched_df = df[df["predicted_tags"].apply(lambda tags: any(tag in tags for tag in predicted))]
#             else:
#                 matched_df = pd.DataFrame()
                
#         if selected_authors:
#             matched_df = matched_df[matched_df["author"].isin(selected_authors)]
            
#         if matched_df.empty:
#             st.warning("No matching quotes found. Try a different word.")
            
#         else:
#             st.subheader("✨ Recommended Quotes")
#             quote_vecs = embedder.encode(matched_df["quote"].tolist(), batch_size=64)
#             sim_scores = np.dot(quote_vecs, emb.T).flatten()
#             matched_df["similarity"] = sim_scores
#             sorted_df = matched_df.sort_values("similarity", ascending=False)
#             if unique_authors_only:
#                 sorted_df = sorted_df.drop_duplicates(subset="author")
#             top_quotes = sorted_df.head(5)
#             for _, row in top_quotes.iterrows():
#                 st.markdown(f"> *{row['quote']}* — **{row['author']}**")
# except Exception as e:
#     st.error("Something went wrong.")
#     st.exception(e)
