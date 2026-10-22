import time
import streamlit as st
from google import genai

client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
MODEL = "gemini-3.8-flash"

st.title("My AI Chatbot")

if "chat" not in st.session_state:
    st.session_state.chat = client.chats.create(model=MODEL)
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask me anything...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    reply = "Sorry, server is busy. Try again."
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            for _ in range(5):
                try:
                    reply = st.session_state.chat.send_message(prompt).text
                    break
                except Exception as e:
                    reply = "Error: " + str(e)[:300]
                    time.sleep(4)
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
