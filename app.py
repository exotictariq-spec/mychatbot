import time
import streamlit as st
from google import genai

st.set_page_config(page_title="My AI Chat", page_icon="🤖")

st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
.stApp {background-color: #212121; color: #ECECEC;}
.block-container {padding-top: 1.5rem; max-width: 720px;}
[data-testid="stChatMessage"] {background: transparent;}
[data-testid="stChatInput"] textarea {border-radius: 24px;}
h1 {text-align: center; font-size: 1.6rem;}
</style>
""", unsafe_allow_html=True)

MODEL = "gemini-3.8-flash"

st.title("🤖 My AI Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.button("＋ New chat"):
    st.session_state.messages = []
    st.rerun()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Message...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    contents = [
        {"role": "user" if m["role"] == "user" else "model",
         "parts": [{"text": m["content"]}]}
        for m in st.session_state.messages
    ]

    reply = "Sorry, server is busy. Try again."
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            for _ in range(5):
                try:
                    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
                    reply = client.models.generate_content(
                        model=MODEL, contents=contents
                    ).text
                    break
                except Exception as e:
                    reply = "Error: " + str(e)[:300]
                    time.sleep(4)
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
