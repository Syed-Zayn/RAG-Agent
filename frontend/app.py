import streamlit as st
import requests
import uuid
import os

# Backend URL Selection (Environment Variable or Fallback)
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
    
    # 1. Username Fix: No default value, placeholder added
    username = st.text_input("👤 Your Username", value="", placeholder="Enter your name...", help="Files you upload will be linked to this name.")
    
    st.markdown("---")
    st.header("⚙️ Settings")
    
    # Provider Selection
    provider = st.radio("Select AI Model:", ("OpenAI (GPT-3.5)", "Google (Gemini)"))
    provider_key_type = "openai" if "OpenAI" in provider else "gemini"
    
    # 2. Smart API Key Logic (Environment vs Manual)
    api_key = ""
    env_key_name = "OPENAI_API_KEY" if provider_key_type == "openai" else "GOOGLE_API_KEY"
    env_key_val = os.getenv(env_key_name)

    if env_key_val:
        # Agar Environment mein key majood hai
        api_key = env_key_val
        st.success(f"✅ {provider_key_type.title()} Key Loaded from System")
    else:
        # Agar Environment mein nahi hai, to Manual input dikhao
        api_key = st.text_input(f"🔑 {provider_key_type.title()} API Key", type="password")

    st.markdown("---")
    st.subheader("📁 Upload Knowledge")
    
    # Privacy Toggle
    privacy_mode = st.radio("Visibility:", ("Private (Only me)", "Public (Everyone)"), index=0)
    privacy_val = "private" if "Private" in privacy_mode else "public"
    
    uploaded_files = st.file_uploader("Choose PDF/TXT/DOCX", accept_multiple_files=True)
    
    if st.button("🚀 Process Documents"):
        if uploaded_files and api_key and username:
            files = [("files", (file.name, file, file.type)) for file in uploaded_files]
            
            data = {
                "username": username,
                "privacy": privacy_val,
                "provider": provider_key_type, 
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
            if not username:
                st.warning("⚠️ Please enter a Username.")
            elif not api_key:
                st.warning("⚠️ API Key is missing.")

# --- Main Chat ---
if username:
    st.title(f"🤖 Welcome, {username}!")
else:
    st.title("🤖 Welcome, Guest!")

st.markdown("Secure RAG System with **Role-Based Access Control**.")

# Display History (Clean Version)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["confidence"] > 0:
            color = "green" if msg["confidence"] > 70 else "orange"
            st.markdown(f":{color}[**Confidence Score: {msg['confidence']}%**]")
            with st.expander("🔍 Verified Sources (Click to expand)"):
                # Clean History View as well
                for i, src in enumerate(msg["sources"][:2]):
                    st.markdown(f"**{i+1}. {src['source']}**")
                    st.caption(f'"{src["content"][:80]}..."')
                    if i == 0:
                        st.divider()

# Chat Input
if prompt := st.chat_input("Ask a question..."):
    if not api_key:
        st.error("❌ API Key is missing. Please set it in Settings or Environment Variables.")
    elif not username:
        st.error("❌ Please enter a username in the sidebar first.")
    else:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("assistant"):
            with st.spinner("Searching secure knowledge base..."):
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
                    
                    if confidence > 0:
                        color = "green" if confidence > 70 else "orange"
                        st.markdown(f":{color}[**Confidence Score: {confidence}%**]")
                        
                        # Fix: Show only Top 2 Sources (CLEAN VERSION)
                        with st.expander("🔍 Verified Sources (Click to expand)"):
                            for i, src in enumerate(sources[:2]): 
                                st.markdown(f"**{i+1}. {src['source']}**")
                                st.caption(f'"{src["content"][:80]}..."') # Only 80 chars
                                if i == 0:
                                    st.divider()
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "confidence": confidence,
                        "sources": sources
                    })
                except Exception as e:
                    st.error(f"Error: {e}")
