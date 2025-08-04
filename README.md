# 🗞️ RAG News Assistant

**RAG News Assistant** is an AI-powered, real-time news query system that combines the power of **Retrieval-Augmented Generation (RAG)** with **Google Search** and **ChromaDB** for delivering accurate and up-to-date answers to news-related questions.

---

## 🚀 Features

- 🔍 Semantic understanding of user queries using embeddings
- 🧠 Intelligent similarity search using **ChromaDB**
- 🌐 Real-time news retrieval using **Google Search API**
- 💬 Chat-based interface with query history
- 🗃️ Persistent vector storage and retrieval
- ⚡ Built with **Streamlit**, **LLMs**, and **RAG pipeline**

---

## 📌 How It Works

### 🧭 Flow Overview

1. **User Input**
   - The user enters a news-related question via the Streamlit interface.

2. **Similarity Search (ChromaDB)**
   - The system embeds the query using a sentence transformer.
   - It performs a **vector similarity search** against previously stored queries in **ChromaDB**.

3. **Decision: Retrieve or Search**
   - If a similar query exists (e.g., similarity score > 0.65):
     -Use cached data to respond, or regenerate a response with stored content.
   - If no similar question is found:
     - Use the **Google Search API** (or equivalent web search tool) to retrieve the latest news articles.
     - Parse and embed those results.
     - Parse, embed, and store the documents and query in ChromaDB.
4. **RAG Generation**
   - Retrieved documents are passed along with the user’s query to a **Large Language Model (LLM)**.
   - The LLM generates a context-aware, natural language response.

5. **Display**
   - The result is shown in a chat-style format.
   - The conversation is stored in session state and optionally persisted.

---

## 📦 Tech Stack

| Component        | Technology                   |
|------------------|------------------------------|
| Frontend UI      | Streamlit                    |
| Embeddings       | Sentence Transformers (openAi) |
| Vector Store     | ChromaDB (Persistent)        |
| Search           | Google Search API            |
| LLM              | OpenAI                        |
| Backend Logic    | Python                       |

---

## 🛠️ Setup Instructions

```bash
# Clone the repo
git clone https://github.com/your-username/rag-news-assistant.git
cd rag-news-assistant

# Install dependencies
pip install -r requirements.txt
