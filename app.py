import base64
import html
import io
import os
import tempfile
from datetime import datetime, timezone

import cv2
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
from supabase import create_client

st.set_page_config(page_title="Sparkle AI", page_icon="✨")

TEXT_MODEL = "openai/gpt-oss-120b"
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
VOICE_MODEL = "whisper-large-v3-turbo"
SEARCH_MODEL = "openai/gpt-oss-120b"

# ---------- Look: no icons, you on the right, Sparkle on the left ----------
st.markdown("""
<style>
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] { display: none; }
[data-testid="stChatMessage"] { background: transparent; padding: 0.25rem 0; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    flex-direction: row-reverse;
    align-self: flex-end;
    width: fit-content;
    max-width: 85%;
    margin-left: auto;
    background: rgba(128,128,128,0.2);
    border-radius: 18px;
    padding: 0.6rem 1rem;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_llm():
    return OpenAI(api_key=st.secrets["GROQ_API_KEY"],
                  base_url="https://api.groq.com/openai/v1")


@st.cache_resource
def get_db():
    return create_client(st.secrets["SUPABASE_URL"],
                         st.secrets["SUPABASE_SECRET"])


def new_auth_client():
    return create_client(st.secrets["SUPABASE_URL"],
                         st.secrets["SUPABASE_KEY"])


llm = get_llm()
db = get_db()


# ---------- Login screen ----------
def login_screen():
    st.markdown("<h1 style='text-align:center'>✨ Sparkle AI</h1>",
                unsafe_allow_html=True)
    if st.button("Continue with Google", use_container_width=True):
        st.login()
    st.caption("or use your email")
    tab1, tab2 = st.tabs(["Log in", "Sign up"])
    with tab1:
        email = st.text_input("Email", key="li_email")
        pw = st.text_input("Password", type="password", key="li_pw")
        if st.button("Log in", use_container_width=True):
            try:
                res = new_auth_client().auth.sign_in_with_password(
                    {"email": email.strip(), "password": pw})
                st.session_state.user = {"id": res.user.id,
                                         "email": res.user.email}
                st.rerun()
            except Exception:
                st.error("Login failed. Check your email and password.")
    with tab2:
        email2 = st.text_input("Email", key="su_email")
        pw2 = st.text_input("Password (6+ characters)", type="password",
                            key="su_pw")
        if st.button("Create account", use_container_width=True):
            try:
                res = new_auth_client().auth.sign_up(
                    {"email": email2.strip(), "password": pw2})
                if res.session and res.user:
                    st.session_state.user = {"id": res.user.id,
                                             "email": res.user.email}
                    st.rerun()
                else:
                    st.info("Check your email to confirm, then log in.")
            except Exception as e:
                st.error(str(e)[:200])


# ---------- Check if the user came back from Google ----------
try:
    if st.user.is_logged_in and "user" not in st.session_state:
        gmail = str(st.user.email).lower()
        st.session_state.user = {"owner": "g:" + gmail, "email": gmail,
                                 "google": True}
except Exception:
    pass

if "user" not in st.session_state:
    login_screen()
    st.stop()

user = st.session_state.user
owner = user.get("owner") or ("sb:" + user["id"])

st.session_state.setdefault("messages", [])
st.session_state.setdefault("chat_id", None)
st.session_state.setdefault("name", user["email"].split("@")[0].title())


# ---------- Saved chats (database) ----------
def list_chats():
    try:
        return (db.table("chats").select("id,title")
                .eq("user_email", owner)
                .order("updated_at", desc=True).limit(30).execute().data)
    except Exception:
        return []


def load_chat(cid):
    r = (db.table("chats").select("messages").eq("id", cid)
         .eq("user_email", owner).limit(1).execute().data)
    if not r:
        return []
    return [{"role": m["role"], "text": m["text"], "show": m["show"],
             "parts": [], "images": []} for m in r[0]["messages"]]


def save_chat():
    msgs = [{"role": m["role"], "text": m["text"], "show": m["show"]}
            for m in st.session_state.messages]
    now = datetime.now(timezone.utc).isoformat()
    if st.session_state.chat_id is None:
        title = (msgs[0]["show"] or "New chat").strip()[:40] or "New chat"
        r = db.table("chats").insert({
            "user_email": owner, "title": title,
            "messages": msgs, "updated_at": now}).execute()
        st.session_state.chat_id = r.data[0]["id"]
        return True
    (db.table("chats").update({"messages": msgs, "updated_at": now})
     .eq("id", st.session_state.chat_id).eq("user_email", owner).execute())
    return False


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### ✨ Sparkle AI")
    if st.button("➕ New chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_id = None
        st.rerun()
    st.caption("Recent chats")
    for c in list_chats():
        label = ("● " if c["id"] == st.session_state.chat_id else "") + c["title"]
        if st.button(label, key="c_" + c["id"], use_container_width=True):
            st.session_state.messages = load_chat(c["id"])
            st.session_state.chat_id = c["id"]
            st.rerun()
    st.divider()
    with st.expander("⚙️ Settings"):
        name = st.text_input("Your name", key="name")
        style = st.radio("Answer style", ["Detailed", "Short"])
        web = st.toggle("🌐 Web search", value=False)
        if st.session_state.chat_id and st.button("🗑 Delete this chat"):
            (db.table("chats").delete()
             .eq("id", st.session_state.chat_id)
             .eq("user_email", owner).execute())
            st.session_state.messages = []
            st.session_state.chat_id = None
            st.rerun()
    st.caption(user["email"])
    if st.button("Log out"):
        is_google = user.get("google", False)
        for k in ["user", "messages", "chat_id", "name"]:
            st.session_state.pop(k, None)
        if is_google:
            st.logout()
        else:
            st.rerun()


# ---------- Greeting ----------
def greeting():
    try:
        from zoneinfo import ZoneInfo
        hour = datetime.now(ZoneInfo("Asia/Kolkata")).hour
    except Exception:
        hour = datetime.now().hour
    if hour < 12:
        return "Good morning"
    if hour < 17:
        return "Good afternoon"
    return "Good evening"


if not st.session_state.messages:
    st.markdown(
        f"<h2 style='text-align:center;margin-top:30vh'>"
        f"{greeting()}, {html.escape(name)}</h2>",
        unsafe_allow_html=True)

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        for img in m.get("images", []):
            st.image(img, width=250)
        st.markdown(m["show"])


# ---------- Attached files ----------
def to_part(data, mime):
    b64 = base64.b64encode(data).decode()
    return {"type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"}}


def video_frames(data, ext, n=4):
    frames = []
    path = None
    try:
        with tempfile.NamedTemporaryFile(suffix="." + ext, delete=False) as t:
            t.write(data)
            path = t.name
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(n):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (i + 0.5) / n))
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            scale = 640 / max(h, w)
            if scale < 1:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            ok2, buf = cv2.imencode(".jpg", frame)
            if ok2:
                frames.append(buf.tobytes())
        cap.release()
    except Exception:
        pass
    finally:
        if path and os.path.exists(path):
            os.remove(path)
    return frames


def read_files(files):
    extra, parts, shown, names = "", [], [], []
    for f in files:
        names.append(f.name)
        ext = f.name.lower().rsplit(".", 1)[-1]
        data = f.getvalue()
        if ext in ("png", "jpg", "jpeg", "webp"):
            mime = "image/jpeg" if ext in ("jpg", "jpeg") else "image/" + ext
            parts.append(to_part(data, mime))
            shown.append(data)
        elif ext in ("mp4", "mov", "webm"):
            frames = video_frames(data, ext)
            for fr in frames:
                parts.append(to_part(fr, "image/jpeg"))
            if frames:
                shown.append(frames[0])
            extra += (f"\n\n[Video: {f.name}. You can see {len(frames)} "
                      f"pictures taken from it. You cannot hear audio.]")
        elif ext == "pdf":
            reader = PdfReader(io.BytesIO(data))
            txt = "\n".join((p.extract_text() or "") for p in reader.pages)
            extra += f"\n\n[File: {f.name}]\n{txt[:12000]}"
        else:
            txt = data.decode("utf-8", errors="ignore")
            extra += f"\n\n[File: {f.name}]\n{txt[:12000]}"
    return extra, parts[:5], shown, names


# ---------- Message bar: + for files, mic for voice ----------
prompt = st.chat_input(
    "Ask Sparkle AI...",
    accept_file="multiple",
    file_type=["png", "jpg", "jpeg", "webp", "mp4", "mov", "webm",
               "pdf", "txt", "md", "csv", "json", "py"],
    accept_audio=True,
)

if prompt:
    user_text = prompt.text or ""
    files = prompt.files or []

    audio = getattr(prompt, "audio", None)
    if audio is not None:
        try:
            tr = llm.audio.transcriptions.create(
                model=VOICE_MODEL, file=("voice.wav", audio.getvalue()))
            user_text = (user_text + " " + tr.text).strip()
        except Exception as e:
            st.warning("Could not read the voice message: " + str(e)[:150])

    if not user_text and not files:
        st.stop()

    extra, parts, shown, names = read_files(files)
    api_text = (user_text or "Please look at what I attached.") + extra
    show_text = user_text
    if names:
        show_text += ("\n\n" if show_text else "") + "📎 " + ", ".join(names)

    st.session_state.messages.append({
        "role": "user", "text": api_text, "parts": parts,
        "images": shown, "show": show_text})
    with st.chat_message("user"):
        for img in shown:
            st.image(img, width=250)
        st.markdown(show_text)

    use_web = web and not parts

    if style == "Short":
        how = ("Keep replies brief and direct, 1 to 4 sentences unless the "
               "user asks for more. Do not add extra information, warnings "
               "or follow-up suggestions.")
        effort = "low"
    else:
        how = ("Answer like a thoughtful expert. Start straight with the "
               "answer in plain words. Never write labels or headings such "
               "as 'Direct answer', 'Answer' or 'Summary'. Then explain "
               "clearly and in an organised way. Use Markdown when it "
               "helps: bullet lists, numbered steps, tables for "
               "comparisons, and code blocks with a language name. Use "
               "headings only for long answers with several parts. Give a "
               "short example when it helps. Be accurate, say so when you "
               "are unsure, and never make things up. Skip filler, "
               "flattery and repeating the question.")
        effort = "medium"
    system = (f"You are Sparkle AI. Your name is Sparkle AI. The user's name "
              f"is {name}; do not use it in every reply. Reply in the same "
              f"language as the user. {how}")
    if use_web:
        today = datetime.now(timezone.utc).strftime("%d %B %Y")
        system += (f" Today's date is {today}. Use web search for anything "
                   f"current or recent. When you used the web, end with a "
                   f"short list of the source links.")

    history = st.session_state.messages[-12:]
    while history and history[0]["role"] == "assistant":
        history = history[1:]

    api_messages = [{"role": "system", "content": system}]
    for i, m in enumerate(history):
        if i == len(history) - 1 and m["parts"]:
            content = [{"type": "text", "text": m["text"]}] + m["parts"]
        else:
            content = m["text"]
        api_messages.append({"role": m["role"], "content": content})

    has_image = bool(history[-1]["parts"])
    if has_image:
        model, extra_args = VISION_MODEL, {}
    elif use_web:
        model = SEARCH_MODEL
        extra_args = {"tools": [{"type": "browser_search"}],
                      "reasoning_effort": "medium"}
    else:
        model, extra_args = TEXT_MODEL, {"reasoning_effort": effort}

    def stream_reply():
        resp = llm.chat.completions.create(
            model=model, messages=api_messages, stream=True, **extra_args)
        for chunk in resp:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    with st.chat_message("assistant"):
        try:
            if use_web and not has_image:
                reply = None
                with st.spinner("Searching the web..."):
                    try:
                        r = llm.chat.completions.create(
                            model=model, messages=api_messages, **extra_args)
                        reply = r.choices[0].message.content
                    except Exception as e:
                        st.caption("Web search failed (" + str(e)[:80]
                                   + "). Answering without it.")
                if reply:
                    st.markdown(reply)
                else:
                    model = TEXT_MODEL
                    extra_args = {"reasoning_effort": effort}
                    reply = st.write_stream(stream_reply())
            else:
                reply = st.write_stream(stream_reply())
        except Exception as e:
            reply = "Error: " + str(e)[:300]
            st.markdown(reply)

    st.session_state.messages.append({
        "role": "assistant", "text": reply, "parts": [],
        "images": [], "show": reply})

    try:
        created = save_chat()
    except Exception as e:
        created = False
        st.warning("Could not save this chat: " + str(e)[:150])
    if created:
        st.rerun()
