import os
import sys
import time
import uuid
from typing import List, Tuple

import requests
from newspaper import Article
from dotenv import load_dotenv
import tiktoken

import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings
from openai import OpenAI
import streamlit as st

# Load environment variables
load_dotenv()

# =======================
# Backend: RAGNewsBot
# =======================

class RAGNewsBot:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.cse_id = os.getenv("GOOGLE_CSE_ID")
        self.openai_key = os.getenv("OPENAI_API_KEY")

        if not self.api_key or not self.cse_id or not self.openai_key:
            st.error("[FATAL] Missing keys in .env file (GOOGLE_API_KEY, GOOGLE_CSE_ID, OPENAI_API_KEY).")
            sys.exit(1)

        self.openai_client = OpenAI(api_key=self.openai_key)

        # Initialize Chroma vector DB
        chroma_host = os.getenv("CHROMA_HOST")
        self.client = chromadb.HttpClient(
            host=chroma_host,
            port=443,
            ssl=True
        )
        self.embedder = embedding_functions.OpenAIEmbeddingFunction(api_key=self.openai_key)
        self.collection = self.client.get_or_create_collection(
            name="news",
            embedding_function=self.embedder
        )

        self.chat_history = []

    def trim_to_token_limit(self, text: str, max_tokens: int = 7000, model: str = "gpt-3.5-turbo"):
        enc = tiktoken.encoding_for_model(model)
        tokens = enc.encode(text)
        trimmed_tokens = tokens[:max_tokens]
        return enc.decode(trimmed_tokens)

    def google_search(self, query: str, max_results: int = 5) -> List[str]:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "q": query,
            "key": self.api_key,
            "cx": self.cse_id,
            "num": max_results,
            "lr": "lang_en"
        }
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return [item["link"] for item in data.get("items", [])]
        except requests.RequestException as e:
            st.warning(f"[ERROR] Google Search failed: {e}")
            return []

    def extract_article(self, url: str) -> Tuple[str, str]:
        try:
            art = Article(url, language="en")
            art.download()
            art.parse()
            return art.title, art.text
        except Exception as e:
            st.warning(f"[WARN] Article extraction failed for {url}: {e}")
            return "", ""

    def ask_llm(self, query: str, context: str) -> str:
        try:
            prompt = f"Use the following news context to answer the question.\n\nContext:\n{context}\n\nQuestion: {query}"

            messages = [{"role": "system", "content": "You are a helpful news assistant."}]
            messages.extend(self.chat_history)
            messages.append({"role": "user", "content": prompt})

            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.3,
                max_tokens=500
            )

            reply = response.choices[0].message.content.strip()
            self.chat_history.append({"role": "user", "content": prompt})
            self.chat_history.append({"role": "assistant", "content": reply})

            return reply
        except Exception as e:
            return f"[ERROR] OpenAI call failed: {e}"



    def handle_query(self, query: str) -> str:
        try:
        # Search in ChromaDB
            matches = self.collection.query(query_texts=[query], n_results=3, include=['documents', 'distances', 'metadatas'])

            documents = matches.get("documents", [[]])[0]
            scores = matches.get("distances", [[]])[0]  # Lower distance = higher similarity
            sources = matches.get("metadatas", [[]])[0]

            is_rag = False
        # Convert distances to similarity scores (Chroma uses cosine distance)
            
            similarity_scores = [1 - score for score in scores]  # cosine similarity
            if(similarity_scores):
                is_rag = True
                st.info(f"smilarity score: {max(similarity_scores)}")
        # Determine if results are good enough

            if documents and similarity_scores and max(similarity_scores) > 0.65:
                context = "\n\n".join(documents)
            else:
                urls = self.google_search(query)
                if not urls:
                  return "No relevant articles found from the web."

                context = ""
                for url in urls:
                    title, text = self.extract_article(url)
                    if text:
                        trimmed_text = self.trim_to_token_limit(text, max_tokens=7000)
                        doc_id = str(uuid.uuid4())
                        self.collection.add(
                            documents=[trimmed_text],
                            metadatas=[{"source": url}],
                            ids=[doc_id]
                        )
                        context += f"\n\n{trimmed_text}"
                        time.sleep(1)


            if(is_rag):
                st.info(f"🔍 Results from ChromaDB (similarity: {max(similarity_scores):.2f})")
            else:
                st.info("🌐 Results from Web")
            
            return self.ask_llm(query, context)

        except Exception as e:
            return f"[ERROR] Query handling failed: {e}"

 