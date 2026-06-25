import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Vectorless Chatbot",
    page_icon="🤖"
)

st.title("🤖 Vectorless AI Chatbot") 

# Initialize LLM
llm = ChatGroq(
    api_key=os.getenv("gsk_fDAcM9z2eemI6WILaC0GWGdyb3FYDC0pB2S7TOoEwktyucmd8DBoc"),
    model_name="llama-3.1-8b-instant"
)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input
prompt = st.chat_input("Ask me anything...")

if prompt:
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.write(prompt)

    try:
        response = llm.invoke(prompt)

        answer = response.content

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        with st.chat_message("assistant"):
            st.write(answer)

    except Exception as e:
        st.error(f"Error: {e}")