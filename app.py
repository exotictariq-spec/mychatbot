import base64
import html
import io
import os
import tempfile
from datetime import datetime

import cv2
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader

st.set_page_config(page_title="Sparkle AI", page_icon="✨")

TEXT_MODEL = "openai/gpt-oss-120b"
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


@st.cache_resource
def get_client():
    return OpenAI(api_key=st.secrets["GROQ_API_KEY"],
                  base_url="https://api.groq.com/openai/v1")


client = get_client()

# ---------- Settings ----------
with st.sidebar:
    st.header("Settings")
    name = st.text_input("Your name", value="friend")
    style = st.radio("Answer style", ["Short", "Detailed"])
    if st.button("New chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []


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

# ---------- Show old messages ----------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        for img in m["images"]:
            st.image(img, width=250)
        st.markdown(m["show"])


# ---------- Helpers for attached files ----------
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
    extra, parts, shown = "", [], []
    for f in files:
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
    return extra, parts[:5], shown


# ---------- Message bar with + button ----------
prompt = st.chat_input(
    "Ask Sparkle AI...",
    accept_file="multiple",
    file_type=["png", "jpg", "jpeg", "webp", "mp4", "mov", "webm",
               "pdf", "txt", "md", "csv", "json", "py"],
)

if prompt:
    user_text = prompt.text or ""
    files = prompt.files or []
    extra, parts, shown = read_files(files)
    api_text = (user_text or "Please look at what I attached.") + extra

    st.session_state.messages.append({
        "role": "user", "text": api_text, "parts": parts,
        "images": shown, "show": user_text or "(file attached)"})
    with st.chat_message("user"):
        for img in shown:
            st.image(img, width=250)
        st.markdown(user_text or "(file attached)")

    if style == "Short":
        how = "Keep replies brief and to the point, 1 to 4 sentences unless asked for more."
    else:
        how = "Give clear and thorough answers."
    system = (f"You are Sparkle AI. Your name is Sparkle AI. "
              f"The user's name is {name}. Reply in the same language as the user. "
              f"{how} Do not add extra information, warnings or follow-up "
              f"suggestions that the user did not ask for.")

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
    model = VISION_MODEL if has_image else TEXT_MODEL
    extra_args = {} if has_image else {"reasoning_effort": "low"}

    def stream_reply():
        resp = client.chat.completions.create(
            model=model, messages=api_messages, stream=True, **extra_args)
        for chunk in resp:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    with st.chat_message("assistant"):
        try:
            reply = st.write_stream(stream_reply())
        except Exception as e:
            reply = "Error: " + str(e)[:300]
            st.markdown(reply)

    st.session_state.messages.append({
        "role": "assistant", "text": reply, "parts": [],
        "images": [], "show": reply})
