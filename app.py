import os
import streamlit as st
import random
import time
import base64
import uuid
from dotenv import load_dotenv
import io
import json
from pathlib import Path
from typing import List, Tuple
try:
    import fitz  # PyMuPDF for PDF text extraction
except ImportError:  # graceful fallback if PyMuPDF isn't installed
    fitz = None

from lawglance_main import Lawglance
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import HumanMessage, AIMessage

# Set page configuration
st.set_page_config(page_title="Lexora", page_icon="logo/logo.png", layout="wide")

# Load environment variables
load_dotenv()

# Custom CSS for UI
def add_custom_css():
    custom_css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        body { font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; background-color: #f7f9fc; }
        .main .block-container { max-width: 920px; padding-top: 1.2rem; padding-bottom: 2rem; }
        .st-chat-input {
            border-radius: 12px; padding: 12px 14px;
            border: 1px solid #d8dee6; margin-bottom: 10px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }
        .stButton > button {
            background: linear-gradient(180deg, #0B5ED7 0%, #0A4EB5 100%); color: #ffffff;
            font-size: 15px; border-radius: 10px;
            padding: 10px 16px; margin-top: 5px; border: none;
            transition: background-color 0.3s ease;
        }
        .stButton > button:hover { filter: brightness(0.95); }
        .st-chat-message-assistant {
            background: #ffffff; border-radius: 14px; border: 1px solid #e5e9f0;
            padding: 16px 16px; margin-bottom: 16px;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.06);
        }
        .st-chat-message-user {
            background: #eef5ff; border-radius: 14px; border: 1px solid #d6e6ff;
            padding: 16px 16px; margin-bottom: 16px;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.06);
        }
        .chat-input-container {
            position: fixed; bottom: 0; left: 0; right: 0;
            background: rgba(255,255,255,0.9); padding: 14px 16px;
            box-shadow: 0 -6px 24px rgba(16, 24, 40, 0.08);
            display: flex; gap: 10px; backdrop-filter: blur(6px);
        }
        .chat-input { flex-grow: 1; }
        .st-title {
            font-weight: 700; letter-spacing: -0.02em;
            color: #0f172a; display: flex; align-items: center;
            gap: 12px; margin-top: 12px; margin-bottom: 16px;
        }
        .logo { width: 42px; height: 32px; }
        .st-sidebar {
            background-color: #f6f8fb; padding: 18px;
        }
        .st-sidebar header {
            font-size: 18px; font-weight: 700; margin-bottom: 10px;
        }
        .st-sidebar p {
            font-size: 13px; color: #475569;
        }
        .chip { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; background: #eef2ff; color: #3730a3; border: 1px solid #c7d2fe; }
        .sources { margin-top: 6px; }
        .sources li { margin-bottom: 2px; }
        .top-nav { position: sticky; top: 0; z-index: 999; background: rgba(255,255,255,0.85); backdrop-filter: blur(6px); border-bottom: 1px solid #e5e9f0; }
        .top-nav-inner { max-width: 920px; margin: 0 auto; display: flex; align-items: center; justify-content: space-between; padding: 10px 12px; }
        .brand { display: flex; align-items: center; gap: 8px; font-weight: 700; color: #0f172a; }
        .indicator { font-size: 12px; color: #475569; }
        .fade-in { animation: fadein 280ms ease-in; }
        @keyframes fadein { from { opacity: 0; transform: translateY(2px); } to { opacity: 1; transform: translateY(0);} }
        .dark body { background: #0b1220; color: #e5e7eb; }
        .dark .st-chat-message-assistant { background: #0f172a; border-color: #1f2937; }
        .dark .st-chat-message-user { background: #0b2542; border-color: #1f3a67; }
        .dark .top-nav { background: rgba(11, 18, 32, 0.8); border-bottom-color: #1f2937; }
        .dark .stButton > button { background: linear-gradient(180deg, #2563eb 0%, #1d4ed8 100%); }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

add_custom_css()

# Top Navbar
st.markdown(
    f"""
    <div class='top-nav'>
      <div class='top-nav-inner'>
        <div class='brand'>
          <span>⚖️ Lexora</span>
        </div>
        <div class='indicator'>Model: {model_choice if 'model_choice' in globals() else 'Gemini'}</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Title with Logo + subtitle (responsive)
logo_path = "logo/lexora.svg"
col_logo, col_title = st.columns([1, 6])
with col_logo:
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode()
        st.markdown(f"<img src=\"data:image/svg+xml;base64,{encoded_image}\" alt=\"Lexora Logo\" class=\"logo\">", unsafe_allow_html=True)
    else:
        st.markdown("<div class=\"logo\">📘</div>", unsafe_allow_html=True)
with col_title:
    st.markdown("**Lexora**", help="AI-Powered Legal Assistant")
    st.caption("AI-Powered Legal Assistant")

# Local history persistence (fallback when Redis not available)
LOCAL_HISTORY_DIR = Path(".lexora_sessions")
LOCAL_HISTORY_DIR.mkdir(exist_ok=True)

INDEX_PATH = LOCAL_HISTORY_DIR / "index.json"


def _local_history_path(session_id: str) -> Path:
    safe_id = session_id.replace("/", "_").replace("\\", "_")
    return LOCAL_HISTORY_DIR / f"{safe_id}.json"


def _load_local_history(session_id: str):
    try:
        p = _local_history_path(session_id)
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception:
        pass
    return []


def _save_local_history(session_id: str, messages):
    try:
        p = _local_history_path(session_id)
        data = messages
        if 'redact_pii' in globals() and redact_pii:
            # simple redaction for emails and phone-like numbers
            import re
            def _redact(text: str) -> str:
                t = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[redacted-email]", text)
                t = re.sub(r"\b(?:\+?\d[\s-]?){7,}\b", "[redacted-phone]", t)
                return t
            data = [{"role": m["role"], "content": _redact(m["content"]) } for m in messages]
        with p.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception:
        pass


def _load_index():
    try:
        if INDEX_PATH.exists():
            with INDEX_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def _save_index(index_data: dict):
    try:
        with INDEX_PATH.open("w", encoding="utf-8") as f:
            json.dump(index_data, f, ensure_ascii=False)
    except Exception:
        pass


def _ensure_thread_index(session_id: str):
    idx = _load_index()
    if session_id not in idx:
        idx[session_id] = {"title": "", "created_at": int(time.time())}
        _save_index(idx)


def _get_thread_title(session_id: str) -> str:
    idx = _load_index()
    meta = idx.get(session_id) or {}
    title = meta.get("title") or ""
    if not title:
        return session_id
    return title


def _set_thread_title(session_id: str, title: str):
    idx = _load_index()
    meta = idx.get(session_id, {"created_at": int(time.time())})
    meta["title"] = title.strip()
    idx[session_id] = meta
    _save_index(idx)


def _slugify(text: str) -> str:
    import re
    text = re.sub(r"\s+", " ", text).strip().lower()
    text = re.sub(r"[^a-z0-9\-\s]", "", text)
    text = text.replace(" ", "-")
    return text[:60] or "chat"

# Sidebar Info and controls
st.sidebar.header("About Lexora")
st.sidebar.markdown("""
**Lexora** helps answer legal questions using RAG and Gemini models.

Disclaimer: Not legal advice. Verify critical information.
""")

# Optional PII redaction toggle for local saves
redact_pii = st.sidebar.checkbox("Redact PII in saved chats", value=False)

# Sidebar: session controls and filters (future-wired)
def _reset_chat():
    st.session_state.messages = []
    st.session_state.thread_id = str(uuid.uuid4())
    _save_local_history(st.session_state.thread_id, st.session_state.messages)
    _ensure_thread_index(st.session_state.thread_id)

st.sidebar.button("New Chat", use_container_width=True, on_click=_reset_chat)

st.sidebar.divider()

# Conversations picker (local threads)
def _list_local_threads():
    try:
        return [p.stem for p in LOCAL_HISTORY_DIR.glob("*.json")]
    except Exception:
        return []

existing_threads = _list_local_threads()
current_thread = st.session_state.get("thread_id", "")
if existing_threads:
    # Map thread IDs to display titles
    display_titles = []
    for tid in existing_threads:
        display_titles.append(f"{_get_thread_title(tid)} ({tid[:6]})")
    selected_display = st.sidebar.selectbox("Conversations", display_titles, index=0)
    selected_thread = existing_threads[display_titles.index(selected_display)]
    if selected_thread and selected_thread != current_thread:
        # Switch to selected thread
        st.session_state.thread_id = selected_thread
        st.session_state.messages = _load_local_history(selected_thread)

jurisdiction = st.sidebar.selectbox("Jurisdiction", ["India"], index=0)
sources = st.sidebar.multiselect(
    "Sources", [
        "Constitution", "BNS 2023", "BNSS 2023", "BSA 2023",
        "Consumer Protection Act 2019", "Motor Vehicles Act 1988",
        "IT Act 2000", "POCSO 2012"
    ], default=["Constitution", "BNS 2023"]
)

# Persistent session ID
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    _ensure_thread_index(st.session_state.thread_id)

thread_id = st.session_state.thread_id

# Model controls
st.sidebar.divider()
st.sidebar.subheader("Model")
model_choice = st.sidebar.selectbox("Gemini Model", [
    "gemini-2.0-flash-exp",
    "gemini-1.5-pro",
    "gemini-1.5-flash"
], index=0)
temperature = st.sidebar.slider("Creativity (temperature)", 0.0, 1.0, 0.7, 0.1)



# Load Gemini models
gemini_api_key = os.getenv('GEMINI_API_KEY') or 'AIzaSyAmLvnW08x_vAqbTn5fqCB79FnmE3uuK60'
llm = ChatGoogleGenerativeAI(
    model=model_choice, 
    temperature=temperature, 
    google_api_key=gemini_api_key
)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=gemini_api_key
)

# Ensure Chroma tenancy/env and persistence directory
os.environ.setdefault("CHROMA_TENANT", "default")
os.environ.setdefault("CHROMA_DATABASE", "default")
persist_dir = "chroma_db_legal_bot_part1"
os.makedirs(persist_dir, exist_ok=True)

try:
    vector_store = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
except Exception as _e:
    st.sidebar.warning("Chroma persistence unavailable. Using in-memory DB this session.")
    vector_store = Chroma(embedding_function=embeddings)

# Create LawGlance instance
law = Lawglance(llm, embeddings, vector_store)

# Utility: extract text from uploaded files
def extract_text_from_pdf_bytes(data: bytes) -> str:
    if fitz is None:
        return "[PDF text extraction unavailable: install PyMuPDF with `pip install pymupdf`]"
    try:
        doc = fitz.open(stream=data, filetype="pdf")
        pages_text: List[str] = []
        for page in doc:
            pages_text.append(page.get_text("text"))
        return "\n\n".join(pages_text).strip()
    except Exception as e:
        return f"[Unable to extract PDF text: {e}]"


def build_prompt_with_uploads(base_prompt: str, uploads: List[Tuple[str, str]]) -> str:
    if not uploads:
        return base_prompt
    parts = ["Uploaded context:"]
    for name, content in uploads:
        if content:
            preview = content[:4000]
            parts.append(f"- {name}:\n{preview}")
        else:
            parts.append(f"- {name}: [image attached]")
    parts.append("")
    return "\n\n".join(parts) + base_prompt


# Get chat history from backend and display
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Try to load history from Redis if available
    try:
        history = law.get_session_history(thread_id).messages
        for msg in history:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            st.session_state.messages.append({"role": role, "content": msg.content})
    except Exception:
        # Fallback to local state if Redis not running
        st.session_state.messages = _load_local_history(thread_id)

# Display history with avatars
for message in st.session_state.messages:
    role = "user" if message["role"] == "user" else "assistant"
    avatar = "🙋" if role == "user" else "⚖️"
    with st.chat_message(role, avatar=avatar):
        st.markdown(message["content"])



# Display quick suggestions if available
suggestion_clicked = None
if "suggestions" in st.session_state and st.session_state.suggestions:
    st.markdown("---")
    st.markdown("##### You might also want to ask:")
    # Trim suggestions to max 4
    suggestions = st.session_state.suggestions[:4]
    cols = st.columns(len(suggestions))
    for i, suggestion in enumerate(suggestions):
        if cols[i].button(suggestion):
            suggestion_clicked = suggestion
    # Clear suggestions after displaying them
    st.session_state.suggestions = []

# File uploads
uploaded_files = st.file_uploader(
    "Attach documents or images (PDF, TXT, PNG, JPG)",
    type=["pdf", "txt", "png", "jpg", "jpeg", "gif", "webp"],
    accept_multiple_files=True,
)

uploads_payload: List[Tuple[str, str]] = []  # (name, extracted_text or "")
if uploaded_files:
    for uf in uploaded_files:
        name = uf.name
        data = uf.read()
        if name.lower().endswith(".pdf"):
            text = extract_text_from_pdf_bytes(data)
            uploads_payload.append((name, text))
        elif name.lower().endswith(".txt"):
            try:
                uploads_payload.append((name, data.decode(errors="ignore")))
            except Exception:
                uploads_payload.append((name, "[Unable to decode TXT file]"))
        else:
            # Image or other types: preview only for now
            uploads_payload.append((name, ""))
            try:
                st.image(io.BytesIO(data), caption=name, use_column_width=True)
            except Exception:
                st.info(f"Attached file: {name}")

# Prompt input
st.markdown("<div class='chat-input-container'>", unsafe_allow_html=True)
prompt = suggestion_clicked or st.chat_input("Ask Lexora a legal question…")
st.markdown("</div>", unsafe_allow_html=True)

if prompt and prompt.strip():
    with st.chat_message("user", avatar="🙋"):
        if uploads_payload:
            attached_names = ", ".join([name for name, _ in uploads_payload])
            st.markdown(f"{prompt}\n\n_Attached:_ {attached_names}")
        else:
            st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    _save_local_history(thread_id, st.session_state.messages)
    # Auto-title empty sessions from first user prompt
    if not _get_thread_title(thread_id) or _get_thread_title(thread_id) == thread_id:
        _set_thread_title(thread_id, prompt[:60])

    # Invoke LawGlance backend (with Redis error handling)
    try:
        enriched_prompt = build_prompt_with_uploads(prompt + "\n\nPlease use any uploaded context to improve the answer.", uploads_payload)
        # Build metadata filter from sidebar selections
        metadata_filter = {}
        if jurisdiction:
            metadata_filter["country"] = jurisdiction
        if sources:
            metadata_filter["part_name"] = {"$in": sources}

        llm_result, updated_history = law.conversational(
            enriched_prompt,
            thread_id,
            metadata_filter=metadata_filter,
            return_citations=True,
        )
        result = llm_result["answer"]
        citations = llm_result.get("citations", [])
        # Rebuild session messages from updated Redis chat
        st.session_state.messages = []
        for msg in updated_history:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            st.session_state.messages.append({"role": role, "content": msg.content})
        _save_local_history(thread_id, st.session_state.messages)
    except Exception as e:
        if "redis" in str(e).lower() or "10061" in str(e):
            # Fallback: Use LLM directly without Redis, but preserve context
            history_snippets = []
            for m in st.session_state.messages[-8:]:
                prefix = "User:" if m["role"] == "user" else "Assistant:"
                history_snippets.append(f"{prefix} {m['content']}")
            context_prompt = "\n".join(history_snippets + [f"User: {prompt}", "Assistant:"])
            result = llm.invoke(context_prompt).content
            st.session_state.messages.append({"role": "assistant", "content": result})
            _save_local_history(thread_id, st.session_state.messages)
        else:
            result = f"Error: {e}"
            st.session_state.messages.append({"role": "assistant", "content": result})
            _save_local_history(thread_id, st.session_state.messages)

    # Animate AI response
    final_response = f"Lexora: {result}"

    def response_generator(response):
        for word in response.split():
            yield word + " "
            time.sleep(0.05)

    with st.chat_message("assistant", avatar="⚖️"):
        with st.spinner("Lexora is thinking..."):
            animated = "".join(list(response_generator(final_response)))
            st.markdown(animated)
        # Render citations if available
        try:
            if citations:
                st.markdown("<span class='chip'>Sources</span>", unsafe_allow_html=True)
                for doc in citations[:5]:
                    meta = getattr(doc, "metadata", {}) if hasattr(doc, "metadata") else doc.get("metadata", {})
                    title = meta.get("source_name") or meta.get("part_name") or meta.get("source") or "Document"
                    part = meta.get("part")
                    st.markdown(f"- {title}{' - ' + part if part else ''}")
        except Exception:
            pass
        _save_local_history(thread_id, st.session_state.messages)

        # Generate and store follow-up suggestions
        if len(st.session_state.messages) >= 2:
            last_user_message = st.session_state.messages[-2]['content']
            last_ai_message = st.session_state.messages[-1]['content']
            suggestion_prompt = f'''
            Based on the last user question and the assistant's answer, suggest 3 relevant follow-up questions.
            Return the questions as a JSON list of strings. For example: ["Question 1", "Question 2", "Question 3"]

            USER: {last_user_message}
            ASSISTANT: {last_ai_message}

            QUESTIONS:
            '''
            try:
                response = llm.invoke(suggestion_prompt)
                # Clean the response to get only the JSON part
                cleaned_response = response.content.strip()
                if cleaned_response.startswith("```json"):
                    cleaned_response = cleaned_response[7:-4].strip()
                suggestions = json.loads(cleaned_response)
                if suggestions and isinstance(suggestions, list):
                    st.session_state.suggestions = suggestions
            except Exception as e:
                st.session_state.suggestions = [] # Clear suggestions on failure


# Export chat
st.sidebar.divider()
if st.sidebar.button("Export Chat (JSON)", use_container_width=True):
    try:
        p = _local_history_path(thread_id)
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                st.sidebar.download_button("Download Current Chat", data=f.read(), file_name=f"{thread_id}.json", mime="application/json")
        else:
            st.sidebar.info("No chat to export.")
    except Exception:
        st.sidebar.warning("Export failed.")

# Rename current conversation
st.sidebar.divider()
new_title = st.sidebar.text_input("Rename conversation", value=_get_thread_title(st.session_state.get("thread_id", "")))
if new_title:
    _set_thread_title(st.session_state.get("thread_id", ""), new_title)

# Footer
st.markdown("---")
st.caption("Lexora © 2025")
