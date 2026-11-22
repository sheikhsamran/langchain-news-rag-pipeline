import os

import streamlit as st
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
)
from sentence_transformers import CrossEncoder

load_dotenv()
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")


# 1. MODELS

llm = init_chat_model(
    "openai/gpt-oss-120b",
    model_provider="openai",
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_api_key,
    temperature=0.6,
    max_tokens=500,
)

embeddings = OpenAIEmbeddings(
    model="openai/text-embedding-3-small",
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_api_key,
)

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# LangChain contextual compression (no LLM call: splits each chunk into small pieces
# and keeps only the pieces similar to the question, using embeddings)
sentence_splitter = CharacterTextSplitter(
    separator=". ",
    chunk_size=300,
    chunk_overlap=0,
    keep_separator="end",
)

embeddings_filter = EmbeddingsFilter(
    embeddings=embeddings,
    similarity_threshold=0.45,  # raise = stricter, lower = keeps more text
)

compressor = DocumentCompressorPipeline(
    transformers=[sentence_splitter, embeddings_filter]
)


# 2. PAGE

st.set_page_config(page_title="News RAG Pipeline", layout="wide")
st.title("News RAG Pipeline")
st.write("Retrieve 10 chunks, rerank to 5, compress the text, and generate an answer.")


# 3. LOAD ARTICLE URLS

st.sidebar.header("Add Article URLs")

url1 = st.sidebar.text_input("URL 1", placeholder="https://example.com/article")
url2 = st.sidebar.text_input("URL 2", placeholder="https://example.com/article")
url3 = st.sidebar.text_input("URL 3", placeholder="https://example.com/article")

if st.sidebar.button("Load URLs"):

    urls = [url1, url2, url3]
    urls = [url.strip() for url in urls if url.strip()]

    if not urls:
        st.sidebar.warning("Add at least one URL.")

    else:

        with st.spinner("Loading articles and building vector database..."):

            loader = UnstructuredURLLoader(
                urls=urls,
                continue_on_failure=True,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0 Safari/537.36"
                    )
                },
            )

            documents = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100,
            )

            chunks = splitter.split_documents(documents)

            vector_store = FAISS.from_documents(chunks, embeddings)

            st.session_state["vector_store"] = vector_store
            st.session_state["article_count"] = len(documents)
            st.session_state["chunk_count"] = len(chunks)

# Short sidebar message after loading

if "vector_store" in st.session_state:

    st.sidebar.success(
        f"Loaded {st.session_state['article_count']} articles "
        f"and created {st.session_state['chunk_count']} chunks."
    )


# 4. ASK QUESTION

st.header("Ask a Question")

question = st.text_area("Enter your question:")

if st.button("Run RAG Pipeline"):

    if "vector_store" not in st.session_state:
        st.warning("Load articles first.")

    elif not question.strip():
        st.warning("Enter a question.")

    else:

        vector_store = st.session_state["vector_store"]

        # STEP 1: RETRIEVE 10 CHUNKS (MMR)

        with st.spinner("Step 1: Retrieving 10 chunks..."):

            query_vector = embeddings.embed_query(question)

            results = vector_store.max_marginal_relevance_search_with_score_by_vector(
                query_vector,
                k=10,
                fetch_k=20,
            )

            chunks = [doc for doc, distance in results]

        st.subheader("Step 1: Retrieved Chunks (MMR)")

        for i, chunk in enumerate(chunks, 1):
            source = chunk.metadata.get("source", "Unknown")

            with st.expander(f"Retrieved Chunk {i} | {source}"):
                st.write(chunk.page_content)


        # STEP 2: RERANK AND KEEP TOP 5

        with st.spinner("Step 2: Reranking chunks..."):

            pairs = []

            for chunk in chunks:
                pairs.append((question, chunk.page_content))

            scores = reranker.predict(pairs)

            ranked_chunks = sorted(
                zip(scores, chunks),
                key=lambda item: float(item[0]),
                reverse=True,
            )

            top_chunks = ranked_chunks[:5]

        st.subheader("Step 2: Reranked Top 5 Chunks")

        selected_chunks = []

        for rank, (score, chunk) in enumerate(top_chunks, 1):
            selected_chunks.append(chunk)

            source = chunk.metadata.get("source", "Unknown")

            with st.expander(f"Rank {rank} | {source}"):
                st.write(chunk.page_content)


        # STEP 3: COMPRESS CHUNKS (LangChain contextual compression)

        st.subheader("Step 3: Contextual Compression")

        progress = st.progress(0)

        compressed_chunks = []

        for i, chunk in enumerate(selected_chunks, 1):

            with st.spinner(f"Compressing chunk {i} of {len(selected_chunks)}..."):
                result = compressor.compress_documents([chunk], question)

            if result:
                # the compressor returns several small pieces, join them back into one chunk
                merged = result[0]
                merged.page_content = " ".join(piece.page_content for piece in result)

                compressed_chunks.append(merged)

            progress.progress(i / len(selected_chunks))

        for i, chunk in enumerate(compressed_chunks, 1):
            source = chunk.metadata.get("source", "Unknown")

            with st.expander(f"Compressed Chunk {i} | {source}"):
                st.write(chunk.page_content)


        # STEP 4: COMBINE CHUNKS

        context = ""

        for i, chunk in enumerate(compressed_chunks, 1):
            context += (
                f"\n\n[Source {i}: "
                f"{chunk.metadata.get('source', 'Unknown')}]\n"
                f"{chunk.page_content}"
            )


        # STEP 5: FINAL ANSWER

        prompt = f"""
Answer the question using only the context below.

Do not invent facts.
If the answer is missing, say so.
Cite the source URLs that support your answer.

Context:
{context}

Question:
{question}

Answer:
"""

        with st.spinner("Step 5: Generating final answer..."):
            response = llm.invoke(prompt)

        st.subheader("Final Answer")
        st.markdown(response.content)


        # STEP 6: SOURCES

        st.subheader("Sources Used")

        used_sources = []

        for chunk in compressed_chunks:
            source = chunk.metadata.get("source", "Unknown")

            if source not in used_sources:
                used_sources.append(source)

        for source in used_sources:
            st.write(f"- {source}")