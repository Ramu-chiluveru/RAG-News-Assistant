import streamlit as st
from Rag_implementation import RAGNewsBot

st.set_page_config(page_title="RAG News Assistant", layout="centered")
st.title("🗞️ RAG News Assistant")
st.markdown("Ask anything about current events. Powered by Google Search + AI 🤖")

# Initialize session state
if "bot" not in st.session_state:
    st.session_state.bot = RAGNewsBot()

if "chat" not in st.session_state:
    st.session_state.chat = []

# Input box
query = st.text_input("Enter a news-related question")

if query:
    st.session_state.chat.append(("user", query))
    with st.spinner("Searching and generating answer..."):
        reply = st.session_state.bot.handle_query(query)
    st.session_state.chat.append(("bot", reply))

# Display chat
for role, message in st.session_state.chat:
    if role == "user":
        st.markdown(f"**🧑 You:** {message}")
    else:
        st.markdown(f"**🤖 Bot:** {message}")

# Clear chat
if st.button("🧹 Clear Chat"):
    st.session_state.chat = []
    st.session_state.bot.chat_history = []
