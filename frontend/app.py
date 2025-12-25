import streamlit as st
import requests
import uuid
import os

# Backend URL Selection
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Professional RAG Assistant", layout="wide", page_icon="🤖")

# Custom CSS for Professional Look
st.markdown("""
<style>
    .stChatFloatingInputContainer {bottom: 20px;}
    .block-container {padding-top: 2rem;}
</style>
""", unsafe_allow_html=True)

# Persistent Session State
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar ---
with st.sidebar:
    st.title("📂 Knowledge Control")
    
    # User Identity
    username = st.text_input("👤 Username", placeholder="Enter identity...", help="Access Control uses this name.")
    
    st.divider()
    st.subheader("⚙️ Configuration")
    
    # Model Provider
    provider = st.selectbox("AI Model", ("OpenAI (GPT-4o)", "Google (Gemini)"))
    provider_key_type = "openai" if "OpenAI" in provider else "gemini"
    
    # API Key Logic
    api_key = ""
    env_key_name = "OPENAI_API_KEY" if provider_key_type == "openai" else "GOOGLE_API_KEY"
    env_key_val = os.getenv(env_key_name)

    if env_key_val:
        api_key = env_key_val
        st.success(f"✅ {provider_key_type.upper()} Key Active")
    else:
        api_key = st.text_input(f"🔑 Enter {provider_key_type.title()} Key", type="password")

    st.divider()
    
    # Upload Section
    st.subheader("📄 Upload Documents")
    privacy_mode = st.radio("Access Level:", ("Private (Session Only)", "Public (Organization)"), index=0)
    privacy_val = "private" if "Private" in privacy_mode else "public"
    
    uploaded_files = st.file_uploader("Supported: PDF, DOCX, TXT", accept_multiple_files=True)
    
    if st.button("🚀 Process & Ingest"):
        if uploaded_files and api_key and username:
            files = [("files", (file.name, file, file.type)) for file in uploaded_files]
            data = {
                "username": username,
                "privacy": privacy_val,
                "provider": provider_key_type, 
                "api_key": api_key
            }
            
            with st.spinner("Processing vectors & indexing..."):
                try:
                    res = requests.post(f"{BACKEND_URL}/upload/", files=files, data=data)
                    if res.status_code == 200:
                        st.success("✅ Ingestion Complete!")
                    else:
                        st.error(f"Error: {res.text}")
                except Exception as e:
                    st.error(f"Connection Error: {e}")
        else:
            st.warning("⚠️ Missing Username or API Key.")

# --- Main Chat Interface ---
st.title("🧠 Enterprise RAG Agent")
st.caption(f"Session ID: {st.session_state.session_id} | Connected to: {provider}")

if not username:
    st.info("👋 Please enter a **Username** in the sidebar to start.")
    st.stop()

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Display Evidence (Only for Assistant)
        if msg.get("role") == "assistant" and msg.get("confidence", 0) > 0:
            conf = msg["confidence"]
            # Color Logic
            if conf >= 75: color = "green"
            elif conf >= 50: color = "orange"
            else: color = "red"
            
            st.markdown(f":{color}[**Confidence Score: {conf}%**]")
            
            # Citation Expander
            if msg.get("sources"):
                with st.expander("🔍 Verified Sources (Evidence)"):
                    # Show Top 2 Sources Only (UI Cleanliness)
                    for i, src in enumerate(msg["sources"][:2]):
                        score_display = f"{src['score']}% Match"
                        st.markdown(f"**{i+1}. {src['source']}** — *{score_display}*")
                        st.caption(f'"{src["content"][:200]}..."') # Snippet
                        st.divider()

# Chat Input Logic
if prompt := st.chat_input("Ask about your documents..."):
    if not api_key:
        st.error("Please provide an API Key in settings.")
        st.stop()
        
    # User Message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing vectors..."):
            payload = {
                "query": prompt,
                "session_id": st.session_state.session_id,
                "username": username,
                "provider": provider_key_type,
                "api_key": api_key
            }
            try:
                res = requests.post(f"{BACKEND_URL}/chat/", json=payload).json()
                
                answer = res["answer"]
                confidence = res["confidence"]
                sources = res["sources"]
                
                st.markdown(answer)
                
                # Dynamic Footer based on Confidence
                if confidence > 0:
                    if confidence >= 75: color = "green"
                    elif confidence >= 50: color = "orange"
                    else: color = "red"
                    
                    st.markdown(f":{color}[**Confidence Score: {confidence}%**]")
                    
                    if sources:
                        with st.expander("🔍 Verified Sources (Evidence)"):
                            for i, src in enumerate(sources[:2]): 
                                st.markdown(f"**{i+1}. {src['source']}** — *{src['score']}% Relevance*")
                                st.caption(f'"{src["content"][:200]}..."')
                                st.divider()
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "confidence": confidence,
                    "sources": sources
                })
                
            except Exception as e:
                st.error(f"System Error: {e}")
