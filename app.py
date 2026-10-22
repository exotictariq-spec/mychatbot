import time
import uuid
import json
import httpx
import streamlit as st
from groq import Groq

st.set_page_config(page_title="Sparkle AI", page_icon="✨", layout="centered")

# ---------- Look: black, ChatGPT-style ----------
st.markdown("""
<style>
footer {visibility: hidden;}
.stApp {background-color: #000000; color: #F3F4F6;}
section[data-testid="stSidebar"] {background-color: #111111;}
.block-container {padding-top: 2rem; max-width: 720px;}
</style>
""", unsafe_allow_html=True)


# ---------- Groq client (key stays in Streamlit Secrets) ----------
def get_key():
    for name in ("GROQ_API_KEY", "GROQ_KEY", "API_KEY"):
        if name in st.secrets:
            return st.secrets[name]
    return list(st.secrets.values())[0]


client = Groq(api_key=get_key())

MODELS = {
    "Smarter (120B)": "openai/gpt-oss-120b",
    "Faster (20B)": "openai/gpt-oss-20b",
}
LANGUAGES = {
    "Auto": None,
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr",
}
FILE_TYPES = ["txt", "md", "csv", "json", "py", "html", "js", "log"]

BASE_STYLE = (
    "Answer directly and briefly. Give only the information the user needs. "
    "No long introductions, no repeated points, no unnecessary warnings or extra tips. "
    "Use simple words and short sentences. Use bullet points only when they really help. "
    "If the question is simple, answer in one or two sentences. "
    "Reply in the same language the user writes in."
)

ss = st.session_state


# ---------- Permanent storage (GitHub Gist) ----------
def gist_ready():
    return "GITHUB_TOKEN" in st.secrets and "GIST_ID" in st.secrets


def gh_headers():
    return {
        "Authorization": "Bearer " + st.secrets["GITHUB_TOKEN"],
        "Accept": "application/vnd.github+json",
    }


def gist_url():
    return "https://api.github.com/gists/" + st.secrets["GIST_ID"]


def load_chats():
    """Returns a dict of chats, or None if loading failed."""
    try:
        r = httpx.get(gist_url(), headers=gh_headers(), timeout=20)
        r.raise_for_status()
        f = r.json()["files"]["chats.json"]
        text = f["content"]
        if f.get("truncated"):
            text = httpx.get(f["raw_url"], timeout=20).text
        data = json.loads(text or "{}")
        return data if isinstance(data, dict) else {}
    except Exception:
        return None


def save_chats():
    if not gist_ready() or not ss.can_save:
        return
    keep = {cid: c for cid, c in ss.chats.items() if c["messages"]}
    try:
        r = httpx.patch(
            gist_url(),
            headers=gh_headers(),
            json={"files": {"chats.json": {"content": json.dumps(keep)}}},
            timeout=20,
        )
        r.raise_for_status()
    except Exception as e:
        st.toast("Could not save chats: " + str(e)[:100])


# ---------- State ----------
def new_chat():
    cid = uuid.uuid4().hex[:8]
    ss.chats[cid] = {"title": "New chat", "messages": []}
    ss.current = cid


if "chats" not in ss:
    ss.can_save = False
    ss.chats = {}
    if gist_ready():
        loaded = load_chats()
        if loaded is not None:
            ss.chats = loaded
            ss.can_save = True
        else:
            st.warning("Could not load saved chats, so saving is off for now. Refresh to try again.")
    new_chat()

# ---------- Sidebar: new chat, recent chats, settings ----------
with st.sidebar:
    st.markdown("### ✨ Sparkle AI")
    if st.button("➕ New chat", use_container_width=True):
        if ss.chats[ss.current]["messages"]:
            new_chat()
        st.rerun()

    st.markdown("**Recent chats**")
    for cid in reversed(list(ss.chats.keys())):
        chat = ss.chats[cid]
        c1, c2 = st.columns([5, 1])
        label = ("● " if cid == ss.current else "") + chat["title"]
        if c1.button(label, key=f"open_{cid}", use_container_width=True):
            ss.current = cid
            st.rerun()
        if c2.button("🗑", key=f"del_{cid}"):
            del ss.chats[cid]
            if not ss.chats:
                new_chat()
            elif ss.current == cid:
                ss.current = list(ss.chats.keys())[-1]
            save_chats()
            st.rerun()

    st.divider()
    with st.expander("⚙️ Settings"):
        model_name = st.selectbox("Model", list(MODELS.keys()))
        instructions = st.text_area(
            "Instructions for the AI",
            placeholder="Example: Always answer in Hindi.",
        )
        lang_name = st.selectbox("Voice language", list(LANGUAGES.keys()))
        if gist_ready() and ss.can_save:
            st.caption("✅ Chats are saved permanently")
        else:
            st.caption("⚠️ Chats are NOT saved (check Secrets)")
        if st.button("Delete all chats", use_container_width=True):
            ss.chats = {}
            new_chat()
            save_chats()
            st.rerun()

model = MODELS[model_name]
lang = LANGUAGES[lang_name]
chat = ss.chats[ss.current]

# ---------- Show the conversation ----------
for m in chat["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m.get("show", m["content"]))


# ---------- Ask the model (streaming, with retry) ----------
def stream_reply(messages):
    system_text = BASE_STYLE
    if instructions.strip():
        system_text += " " + instructions.strip()
    msgs = [{"role": "system", "content": system_text}]
    msgs += [{"role": m["role"], "content": m["content"]} for m in messages]
    err = ""
    for _ in range(3):
        got_any = False
        try:
            stream = client.chat.completions.create(model=model, messages=msgs, stream=True)
            for chunk in stream:
                piece = chunk.choices[0].delta.content
                if piece:
                    got_any = True
                    yield piece
            return
        except Exception as e:
            err = str(e)
            if got_any:
                break
            time.sleep(2)
    yield "\n\nError: " + err[:300]


# ---------- Message bar with 📎 attach and 🎤 mic inside ----------
prompt = st.chat_input(
    "Message Sparkle...",
    accept_file="multiple",
    file_type=FILE_TYPES,
    accept_audio=True,
)

if prompt:
    user_text = (prompt.text or "").strip()
    files = prompt.files or []
    audio = getattr(prompt, "audio", None)

    if audio is not None:
        with st.spinner("Listening..."):
            try:
                kwargs = {
                    "file": ("voice.wav", audio.getvalue()),
                    "model": "whisper-large-v3-turbo",
                }
                if lang:
                    kwargs["language"] = lang
                heard = client.audio.transcriptions.create(**kwargs).text
                user_text = (user_text + " " + heard).strip()
            except Exception as e:
                st.error("Could not understand the voice: " + str(e)[:200])

    if user_text or files:
        if not user_text:
            user_text = "Please read the attached file."
        full = user_text
        shown = user_text
        for f in files:
            body = f.getvalue().decode("utf-8", errors="ignore")[:20000]
            full += f"\n\n[Attached file: {f.name}]\n{body}"
            shown += f"\n\n📎 {f.name}"

        if not chat["messages"]:
            chat["title"] = user_text[:28]
        chat["messages"].append({"role": "user", "content": full, "show": shown})

        with st.chat_message("user"):
            st.markdown(shown)
        with st.chat_message("assistant"):
            reply = st.write_stream(stream_reply(chat["messages"]))
        chat["messages"].append({"role": "assistant", "content": reply})

        save_chats()
        st.rerun()
