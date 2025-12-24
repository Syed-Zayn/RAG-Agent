import streamlit as st
import requests
import uuid
from dotenv import load_dotenv
# Load Environment Variables
load_dotenv()
# Backend API URL
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Professional RAG Assistant", layout="wide")

# Persistent Session
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=50)
    st.title("User Profile")
    
    # User Identification (Acts as Login)
    username = st.text_input("👤 Your Username", value="zain", help="Files you upload will be linked to this name.")
    
    st.markdown("---")
    st.header("⚙️ Settings")
    
    provider = st.radio("Select AI Model:", ("OpenAI (GPT-3.5)", "Google (Gemini)"))
    provider_key = "openai" if "OpenAI" in provider else "gemini"
    api_key = st.text_input(f"🔑 {provider_key.title()} API Key", type="password")
    
    st.markdown("---")
    st.subheader("📁 Upload Knowledge")
    
    # Privacy Toggle
    privacy_mode = st.radio("Visibility:", ("Private (Only me)", "Public (Everyone)"), index=0)
    privacy_val = "private" if "Private" in privacy_mode else "public"
    
    uploaded_files = st.file_uploader("Choose PDF/TXT", accept_multiple_files=True)
    
    if st.button("🚀 Process Documents"):
        if uploaded_files and api_key and username:
            files = [("files", (file.name, file, file.type)) for file in uploaded_files]
            
            data = {
                "username": username,
                "privacy": privacy_val,
                "provider": provider_key, 
                "api_key": api_key
            }
            
            with st.spinner(f"Ingesting as {privacy_val.upper()} document..."):
                try:
                    res = requests.post(f"{BACKEND_URL}/upload/", files=files, data=data)
                    if res.status_code == 200:
                        st.success(res.json()["message"])
                    else:
                        st.error(f"Error: {res.text}")
                except Exception as e:
                    st.error(f"Connection Error: {e}")
        else:
            st.warning("Please enter Username and API Key.")

# --- Main Chat ---
st.title(f"🤖 Welcome, {username}!")
st.markdown("Secure RAG System with **Role-Based Access Control**.")

# Display History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("View Sources & Confidence"):
                st.write(f"**Confidence Score:** {msg['confidence']}%")
                for s in msg["sources"]:
                    st.caption(f"📄 {s['source']} (Relevance: {s['score']:.4f})")
                    st.text(s['content'][:200] + "...")

# Chat Input
if prompt := st.chat_input("Ask a question..."):
    if not api_key:
        st.error("Please enter API Key.")
    else:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("assistant"):
            with st.spinner("Searching secure knowledge base..."):
                payload = {
                    "query": prompt,
                    "session_id": st.session_state.session_id,
                    "username": username,  # Send current user identity
                    "provider": provider_key,
                    "api_key": api_key
                }
                try:
                    res = requests.post(f"{BACKEND_URL}/chat/", json=payload).json()
                    answer = res["answer"]
                    confidence = res["confidence"]
                    sources = res["sources"]
                    
                    st.markdown(answer)
                    
                    if confidence > 0:
                        color = "green" if confidence > 70 else "orange"
                        st.markdown(f":{color}[**Confidence: {confidence}%**]")
                        with st.expander("🔍 Verified Sources"):
                            for src in sources:
                                st.markdown(f"**Source:** `{src['source']}`")
                                st.info(src['content'])
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "confidence": confidence,
                        "sources": sources
                    })
                except Exception as e:

                    st.error(f"Error: {e}")
