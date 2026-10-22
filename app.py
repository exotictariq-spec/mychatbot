import time
import streamlit as st
from groq import Groq

st.set_page_config(page_title="Sparkle AI", page_icon="✨")

st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
.stApp {background-color: #000000; color: #F3F4F6;}
.block-container {padding-top: 1.5rem; max-width: 720px;}
h1 {text-align: center; font-size: 1.7rem; color: #FFFFFF;}
p, li, span, label, div {color: #F3F4F6;}
[data-testid="stChatMessage"] {
    background: #1A1A1A;
    border-radius: 20px;
    padding: 12px 16px;
    margin-bottom: 10px;
}
[data-testid="stBottom"], [data-testid="stBottom"] > div {
    background: #000000;
}
[data-testid="stChatInput"] {background: #000000;}
[data-testid="stChatInput"] > div {
    border-radius: 28px;
    border: 1px solid #333333;
    background: #1A1A1A;
}
[data-testid="stChatInput"] textarea {
    color: #FFFFFF;
    background: #1A1A1A;
}
.stButton > button {
    border-radius: 999px;
    border: 1px solid #333333;
    background: #000000;
    color: #FFFFFF;
    padding: 6px 18px;
}
.stButton > button:hover {background: #1A1A1A; border-color: #555555;}
code {color: #FFFFFF;}
</style>
""", unsafe_allow_html=True)

MODEL = "openai/gpt-oss-120b"


@st.cache_resource
def get_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])


def stream_reply(messages):
    client = get_client()
    for attempt in range(3):
        got_text = False
        try:
            stream = client.chat.completions.create(
                model=MODEL, messages=messages, stream=True
            )
            for chunk in stream:
                text = chunk.choices[0].delta.content
                if text:
                    got_text = True
                    yield text
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

    messages = st.session_state.messages[-20:]

    with st.chat_message("assistant"):
        reply = st.write_stream(stream_reply(messages))

    st.session_state.messages.append({"role": "assistant", "content": reply})
