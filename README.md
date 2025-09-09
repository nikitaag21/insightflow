# 🚀 INSIGHTFLOW: AI-Powered News Research for Equity Analysts  

## 📌 Introduction  
*InsightFlow* is a user-friendly *AI-powered news research tool* designed for effortless financial information retrieval.  

- Users can input article URLs and ask questions to receive *relevant insights* from the stock market and financial domain.  
- It ingests diverse data like *news articles, reports, and CSV files* using *LangChain loaders*.  
- Texts are broken into *manageable chunks* → converted into *vector embeddings* stored in *FAISS* for *fast semantic search*.  
- When analysts ask questions → relevant chunks are retrieved → fed to *Google Generative AI* → concise, accurate answers with *source citations* are generated.  
- The entire process is accessible via a *Streamlit web interface*, making research efficient, transparent, and cost-effective.  

---

## 🛠 Installation Guide  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/nikitaag21/insightflow
cd insightflow


2️⃣ Check Python Version

Run:

python --version


If Python < 3.13 → use requirements A.

If Python ≥ 3.13 → use requirements B.

3️⃣ Install Requirements
✅ For Python < 3.13

Create requirements.txt with:

streamlit==1.36.0
python-dotenv==1.0.0
unstructured==0.11.2
faiss-cpu==1.8.0
python-magic==0.4.27
protobuf==4.25.3

langchain==0.3.7
langchain-community==0.3.7
langchain-core==0.3.19
langchain-google-genai==2.0.1

✅ For Python ≥ 3.13
streamlit==1.38.0
python-dotenv==1.0.1
unstructured==0.18.14       # latest version that supports Python < 3.13
faiss-cpu==1.12.0
libmagic==1.0
python-magic==0.4.27
python-magic-bin==0.4.14

langchain==0.3.7
langchain-community==0.3.7
langchain-core==0.3.15
langchain-google-genai==2.0.5


Then install dependencies:

pip install -r requirements.txt

4️⃣ Set Up API Key

Go to Google Cloud Console

Create a project & generate a Google Generative AI API key

Create a .env file in your project root and add:

GOOGLE_API_KEY=your_api_key_here

5️⃣ Run the Application

Start Streamlit server:

streamlit run main.py


The app will open in your default browser 🎉
