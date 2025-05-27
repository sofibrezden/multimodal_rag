import sys
import streamlit as st
import openai

import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.utils import load_faiss_index, load_json, load_model
from scripts.search_articles import query_index, select_top_articles_and_images, rerank_results

# configs
INDEX_PATH = "output/textonly_blip.index"
META_PATH = "output/textonly_meta_blip.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gpt-4o-mini"

openai.api_key = st.secrets["OPENAI_API_KEY"]
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


@st.cache_resource
def load_model_and_data():
    model = load_model(EMBEDDING_MODEL)
    index = load_faiss_index(INDEX_PATH)
    metadata = load_json(META_PATH)
    return model, index, metadata


# limit content width
with st.container():
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1000px;
            padding-left: 3rem;
            padding-right: 3rem;
            margin-left: auto;
            margin-right: auto;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

st.title("🔍 Multimodal Retrieval System (Text + Images)")

model, index, metadata = load_model_and_data()
tab1, tab2 = st.tabs(["🔎 Search (RAG)", "🧠 LLM Answer"])

with tab1:
    query1 = st.text_input("Enter your search query", key="query_tab1")
    pool_size = 50

    if query1:
        # Allow user to filter results by type
        filter_option = st.radio(
            "Filter results by type:",
            ["Show All", "Only Articles", "Only Images"],
            horizontal=True
        )
        st.markdown("---")
        D, I = query_index(query1, pool_size, model, index)
        text_results, image_results = select_top_articles_and_images(D, I, metadata, top_articles=20, top_images=10)

        text_results = rerank_results(query1, text_results, model, top_k=5)
        image_results = rerank_results(query1, image_results, model, top_k=3)

        # photos
        if filter_option in ["Show All", "Only Images"]:
            if image_results:
                st.subheader("🖼️ Most relevant images:")
                for idx, entry, _ in image_results:
                    st.image(entry["meta"]["image_url"], caption=entry["raw_text"], use_container_width=True)
                    st.markdown("---")
            else:
                st.info("No relevant images found for your query.")

        # articles
        if filter_option in ["Show All", "Only Articles"]:
            if text_results:
                st.subheader("📰 Top relevant articles:")
                for idx, entry, _ in text_results:
                    st.markdown("📰 Article:")
                    st.markdown(f"**{entry['meta'].get('title', '')}**")
                    st.markdown(entry["raw_text"][:1000] + "...")
                    article_url = entry["meta"].get("url", "")
                    st.markdown(f"[Read more]({article_url})")
                    st.markdown("---")
            else:
                st.info("No relevant articles found for your query.")
with tab2:
    query2 = st.text_input("Ask your question directly", key="query_tab2")
    pool_size = 50

    if query2:
        st.markdown("---")
        D, I = query_index(query2, pool_size, model, index)
        text_results, image_results = select_top_articles_and_images(D, I, metadata, top_articles=20,
                                                                     top_images=10)

        # uses rerank
        text_results = rerank_results(query2, text_results, model, top_k=5)
        image_results = rerank_results(query2, image_results, model, top_k=3)

        context_chunks = []
        for idx, entry, _ in image_results:
            context_chunks.append(f"[Image] {entry['raw_text']}")

        for idx, entry, _ in text_results:
            context_chunks.append(f"[Article] {entry['raw_text']}")

        full_context = "\n\n".join(context_chunks)
        with st.spinner("Generating answer..."):
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system",
                     "content": "You are a helpful assistant who summarizes AI news and describes visual content."},
                    {"role": "user", "content": f"Context:\n{full_context}\n\nQuestion: {query2}"},
                ],
                max_tokens=500,
            )
            answer = response.choices[0].message.content
            st.markdown("### 💡 Answer from GPT")
            st.write(answer)

        if image_results:
            st.subheader("🖼️ Images referenced in the context:")
            for idx, entry, _ in image_results:
                st.image(entry["meta"]["image_url"], caption=entry["raw_text"], use_container_width=True)
                st.markdown("---")
        else:
            st.info("No images found for your query.")