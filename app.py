import time
import streamlit as st
from google import genai

st.set_page_config(page_title="Sparkle AI", page_icon="✨")

st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
.stApp {background-color: #FFFFFF; color: #1F2937;}
.block-container {padding-top: 1.5rem; max-width: 720px;}
h1 {text-align: center; font-size: 1.7rem; color: #1F2937;}
p, li, span, label {color: #1F2937;}
[data-testid="stChatMessage"] {
    background: #F3F4F6;
    border-radius: 20px;
    padding: 12px 16px;
    margin-bottom: 10px;
}
[data-testid="stChatInput"] {background: #FFFFFF;}
[data-testid="stChatInput"] > div {
    border-radius: 28px;
    border: 1px solid #E5E7EB;
    background: #F9FAFB;
}
[data-testid="stChatInput"] textarea {
    color: #1F2937;
    background: #F9FAFB;
}
.stButton > button {
    border-radius: 999px;
    border: 1px solid #E5E7EB;
    background: #FFFFFF;
    color: #1F2937;
    padding: 6px 18px;
}
.stButton > button:hover {background: #F3F4F6; border-color: #D1D5DB;}
</style>
""", unsafe_allow_html=True)

MODEL = "gemini-3.8-flash"


@st.cache_resource
def get_client():
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])


def stream_reply(contents):
    client = get_client()
    for attempt in range(3):
        got_text = False
        try:
            for chunk in client.models.generate_content_stream(
                model=MODEL, contents=contents
            ):
                if chunk.text:
                    got_text = True
                    yield chunk.text
            return
        except Exception as e:
            if got_text:
                return
            if attempt == 2:
                yield "Error: " + str(e)[:300]
                return
            time.sleep(1)


st.title("✨ Sparkle AI")

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.button("＋ New chat"):
    st.session_state.messages = []
    st.rerun()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask me anything...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    contents = [
        {"role": "user" if m["role"] == "user" else "model",
         "parts": [{"text": m["content"]}]}
        for m in st.session_state.messages[-20:]
    ]

    with st.chat_message("assistant"):
        reply = st.write_stream(stream_reply(contents))

    st.session_state.messages.append({"role": "assistant", "content": reply})
