import asyncio
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv

# -----------------------
# Load environment variables
# -----------------------
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("❌ GOOGLE_API_KEY not found. Check your .env file.")
os.environ["GOOGLE_API_KEY"] = api_key
print("✅ GOOGLE_API_KEY loaded successfully")

# -----------------------
# Ensure asyncio loop exists (fix for Streamlit + gRPC)
# -----------------------
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# -----------------------
# Initialize embeddings AFTER event loop is set
# -----------------------
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# -----------------------
# Streamlit UI
# -----------------------
st.title("🔎 News Research Tool (Gemini + FAISS)")

# Sidebar input
st.sidebar.header("⚙️ Settings")
urls = st.sidebar.text_area("Enter URLs (comma separated):").split(",")

if st.sidebar.button("Fetch and Process Data"):
    all_text = ""
    for url in urls:
        url = url.strip()
        if not url:
            continue
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            paragraphs = soup.find_all("p")
            text = " ".join([p.get_text() for p in paragraphs])
            all_text += text + "\n"
            st.sidebar.success(f"✅ Data fetched from {url}")
        except Exception as e:
            st.sidebar.error(f"❌ Error fetching {url}: {e}")

    if all_text:
        # Split text into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_text(all_text)

        # Store in FAISS
        vectorstore = FAISS.from_texts(chunks, embeddings)

        st.success("✅ Data embedded and stored in FAISS!")
        vectorstore.save_local("faiss_index")
        st.info("💾 FAISS index saved locally.")

# Search section
query = st.text_input("🔍 Ask a question about the fetched articles:")
if st.button("Search") and query:
    vectorstore = FAISS.load_local("faiss_index", embeddings)
    docs = vectorstore.similarity_search(query, k=3)

    st.write("### 📌 Top Results:")
    for i, doc in enumerate(docs, 1):
        st.write(f"**Result {i}:** {doc.page_content[:500]}...")
        st.write(f"**Result {i}:** {doc.page_content[:500]}...")