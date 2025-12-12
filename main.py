import streamlit as st
import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from groq import Groq
import pickle
import time

# ----------------- LOAD ENVIRONMENT VARIABLES -----------------
# ----------------- LOAD STREAMLIT SECRETS -----------------
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY missing in Streamlit Secrets")
    st.stop()

# ----------------- INITIALIZE GROQ CLIENT -----------------
groq_client = Groq(api_key=GROQ_API_KEY)

# ----------------- FILE FOR SAVING DOCUMENTS -----------------
DOCS_FILE = "documents.pkl"

# ----------------- STREAMLIT UI SETUP -----------------
st.set_page_config(page_title="INSIGHT FLOW 🔎", layout="wide")
st.title("INSIGHT FLOW 🔎 : AI News Research Tool (Groq-only)")
st.markdown(
    "Fetch news articles from URLs, get AI summaries first, and optionally view full article chunks."
)

# ----------------- SIDEBAR SETTINGS -----------------
st.sidebar.header("⚙️ Settings")
url1 = st.sidebar.text_input("Enter URL 1:")
url2 = st.sidebar.text_input("Enter URL 2:")
url3 = st.sidebar.text_input("Enter URL 3:")
urls = [u.strip() for u in (url1, url2, url3) if u and u.strip()]

# ----------------- HELPER FUNCTIONS -----------------
HEADERS = {"User-Agent": "Mozilla/5.0"}

def extract_title(soup: BeautifulSoup) -> str:
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        return og.get("content").strip()
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)
    return "(no title)"

def fetch_and_extract(url: str):
    resp = requests.get(url, headers=HEADERS, timeout=12)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "aside", "form", "svg"]):
        tag.decompose()
    title = extract_title(soup)
    article = soup.find("article")
    container = article if article else soup
    paragraphs = [p.get_text(" ", strip=True) for p in container.find_all("p")]
    text = "\n".join(p for p in paragraphs if p).strip()
    return title, text

def ask_groq(prompt: str, context: str = "") -> str:
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": f"Question: {prompt}\n\nContext:\n{context}"}
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Groq error: {e}"

def save_documents_to_disk(docs):
    try:
        with open(DOCS_FILE, "wb") as f:
            pickle.dump(docs, f)
        st.sidebar.success(f"💾 Saved {len(docs)} documents to disk.")
    except Exception as e:
        st.sidebar.error(f"❌ Failed to save documents: {e}")

def load_documents_from_disk():
    if os.path.exists(DOCS_FILE):
        try:
            with open(DOCS_FILE, "rb") as f:
                docs = pickle.load(f)
            return docs
        except Exception as e:
            st.sidebar.warning(f"⚠️ Could not load saved documents: {e}")
    return []

def clear_documents():
    if "docs" in st.session_state:
        del st.session_state["docs"]
    if os.path.exists(DOCS_FILE):
        os.remove(DOCS_FILE)
    st.sidebar.success("🗑️ Cleared all documents from memory and disk.")

# ----------------- LOAD SAVED DOCUMENTS -----------------
saved_docs = load_documents_from_disk()
if saved_docs:
    st.session_state["docs"] = saved_docs
    st.sidebar.info(f"Loaded {len(saved_docs)} saved documents from disk.")

# ----------------- FETCH & PROCESS DATA -----------------
if st.sidebar.button("Fetch and Process Data"):
    if not urls:
        st.sidebar.warning("⚠️ Please enter at least one URL")
    else:
        documents = []
        progress_bar = st.progress(0)
        total_urls = len(urls)

        for idx, url in enumerate(urls, 1):
            st.sidebar.info(f"Fetching {url} ...")
            try:
                title, text = fetch_and_extract(url)
                if not text:
                    st.sidebar.warning(f"⚠️ No text found at {url}")
                    continue

                # Split into chunks
                splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
                chunks = splitter.split_text(text)

                # Save chunks as Document objects
                for chunk_idx, chunk in enumerate(chunks, 1):
                    documents.append(Document(
                        page_content=chunk,
                        metadata={"source": url, "title": title, "chunk": chunk_idx, "length": len(chunk)}
                    ))

                st.sidebar.success(f"✅ Fetched {len(chunks)} chunks from {url}")
            except Exception as e:
                st.sidebar.error(f"❌ Error fetching {url}: {e}")
            progress_bar.progress(idx / total_urls)
            time.sleep(0.3)

        if documents:
            st.session_state["docs"] = documents
            save_documents_to_disk(documents)
            st.success(f"✅ Total {len(documents)} chunks ready for querying!")

# ----------------- CLEAN DATA -----------------
if st.sidebar.button("Clean Data"):
    clear_documents()

# ----------------- QUERY INTERFACE -----------------
multi_queries = st.text_area(
    "🔍 Enter your questions (one per line):",
    placeholder="What is the target price for Tata Motors?\nWhat is the outlook for Mahindra?"
)

k = st.slider("Top results to display per question (k)", min_value=1, max_value=10, value=3)

if st.button("Get Summaries") and multi_queries.strip():
    if "docs" not in st.session_state or not st.session_state["docs"]:
        st.error("No data available. Please fetch and process URLs first.")
    else:
        docs = st.session_state["docs"]

        # Process each query
        queries = [q.strip() for q in multi_queries.strip().split("\n") if q.strip()]
        st.write("### 📝 Groq AI Summaries:")
        context_text = "\n".join([d.page_content for d in docs])
        for query_text in queries:
            answer = ask_groq(query_text, context_text)
            st.markdown(f"**Question:** {query_text}")
            st.success(answer)
            st.divider()

        # Option to show all chunks after summaries
        if st.checkbox("Show all article chunks"):
            st.write("### 📄 Full Text from Articles:")
            for doc in docs:
                src = doc.metadata.get("source", "Unknown")
                title = doc.metadata.get("title", src)
                chunk_no = doc.metadata.get("chunk", 0)
                with st.expander(f"{title} | Chunk #{chunk_no} | Source: {src}"):
                    st.write(doc.page_content)
