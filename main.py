import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import requests
from bs4 import BeautifulSoup
import os
import shutil
import google.genai as genai

# ----------------- SETUP -----------------
# ✅ Use Streamlit secrets instead of .env
api_key = st.secrets["general"]["GOOGLE_API_KEY"]
if not api_key:
    raise ValueError("❌ GOOGLE_API_KEY not found in Streamlit secrets.")
os.environ["GOOGLE_API_KEY"] = api_key
print("✅ GOOGLE_API_KEY loaded successfully from Streamlit secrets")

# Create Gemini client
client = genai.Client(api_key=api_key)

# Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=api_key)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

# ----------------- HELPERS -----------------
def extract_title(soup: BeautifulSoup) -> str:
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        return og.get("content").strip()
    tw = soup.find("meta", attrs={"name": "twitter:title"})
    if tw and tw.get("content"):
        return tw.get("content").strip()
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)
    return "(no title)"

def validate_url(url: str) -> bool:
    return bool(url and url.startswith("http"))

def fetch_and_extract(url: str, timeout: int = 12):
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    for tag in soup(["script", "style", "noscript", "header", "footer", "aside", "form", "svg"]):
        tag.decompose()

    title = extract_title(soup)

    container = soup.find("article") or soup.find("div", class_="article-content") or soup
    paragraphs = [p.get_text(" ", strip=True) for p in container.find_all("p")]
    text = "\n".join(p for p in paragraphs if p).strip()
    return title, text

def get_index_path() -> str:
    return "faiss_index"

def ask_gemini(prompt: str, context: str = "") -> str:
    contents = f"Answer the question clearly.\n\nQuestion: {prompt}\n\nContext:\n{context}"
    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=contents,
            config={"max_output_tokens": 256}
        )
        return response.candidates[0].content.parts[0].text
    except Exception as e:
        return f"⚠️ Error generating response: {e}"

# ----------------- STREAMLIT APP -----------------
st.title("INSIGHT FLOW🔎 : AI News Research Tool")

st.sidebar.header("⚙️ Settings")
url1 = st.sidebar.text_input("Enter URL 1:")
url2 = st.sidebar.text_input("Enter URL 2:")
url3 = st.sidebar.text_input("Enter URL 3:")
urls = [u.strip() for u in (url1, url2, url3) if u and u.strip()]

if st.sidebar.button("Fetch and Process Data"):
    documents = []
    for url in urls:
        if not validate_url(url):
            st.sidebar.warning(f"⚠️ Skipped invalid URL: {url}")
            continue
        try:
            title, text = fetch_and_extract(url)
            if not text:
                st.sidebar.warning(f"⚠️ No text found at {url}")
                continue

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_text(text)

            for chunk in chunks:
                documents.append(Document(page_content=chunk, metadata={"source": url, "title": title}))

            st.sidebar.success(f"✅ Data fetched from {url} ({len(chunks)} chunks)")
        except Exception as e:
            st.sidebar.error(f"❌ Error fetching {url}: {e}")

    if documents:
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local(get_index_path())
        st.success("✅ Data embedded and stored in FAISS!")
        st.info("💾 FAISS index saved locally.")

if st.sidebar.button("Clear Stored Data"):
    if os.path.exists(get_index_path()):
        shutil.rmtree(get_index_path())
        st.sidebar.success("🗑️ FAISS index cleared.")
    else:
        st.sidebar.warning("⚠️ No stored FAISS index found.")

k = st.slider("Top results to retrieve (k)", min_value=1, max_value=10, value=3)
query = st.text_input("🔍 Ask a question about the fetched articles:")

if st.button("Search") and query:
    try:
        vectorstore = FAISS.load_local(get_index_path(), embeddings, allow_dangerous_deserialization=True)
    except Exception:
        st.error("No FAISS index found. Please 'Fetch and Process Data' first.")
        st.stop()

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": max(2 * k, 20), "lambda_mult": 0.6}
    )
    docs = retriever.invoke(query)

    if not docs:
        st.warning("No relevant content found in indexed pages. Using Gemini directly.")
        answer = ask_gemini(query)
        st.write("### 🤖 Gemini Answer:")
        st.write(answer)
    else:
        st.write("### 📌 Top Results:")
        context_text = ""
        for i, doc in enumerate(docs, 1):
            src = doc.metadata.get("source", "Unknown")
            title = doc.metadata.get("title", src)
            st.markdown(f"**Result {i}: [{title}]({src})**")
            snippet = doc.page_content.strip()
            st.write(snippet[:500] + ("…" if len(snippet) > 500 else ""))
            st.divider()
            context_text += snippet + "\n"

        st.write("### Summary of the results🤖:")
        answer = ask_gemini(query, context_text)
        st.success(answer)

        unique_sources = {d.metadata.get("source") for d in docs}
        if len(unique_sources) == 1:
            st.warning("All top results come from the same source.")
