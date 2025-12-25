import streamlit as st
import requests
import uuid
import os

# Backend URL
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG Knowledge Assistant", layout="centered")

# Custom CSS for Professional Look
st.markdown("""
    <style>
    .stChatMessage { border-radius: 10px; }
    .stButton>button { width: 100%; }
    </style>
""", unsafe_allow_html=True)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar ---
with st.sidebar:
    st.header("👤 User Access")
    username = st.text_input("Username", placeholder="Enter your identifier...")
    
    st.divider()
    
    st.subheader("⚙️ Configuration")
    provider = st.radio("AI Model Provider", ("OpenAI (GPT-4o)", "Google (Gemini)"))
    provider_key_type = "openai" if "OpenAI" in provider else "gemini"
    
    api_key = st.text_input(f"{provider_key_type.title()} API Key", type="password")
    
    st.divider()
    st.subheader("📄 Knowledge Base")
    
    privacy_mode = st.radio("Document Visibility", ("Private (Session)", "Public (Shared)"), index=0)
    privacy_val = "private" if "Private" in privacy_mode else "public"
    
    uploaded_files = st.file_uploader("Upload Documents (PDF/DOCX)", accept_multiple_files=True)
    
    if st.button("Processing & Ingestion"):
        if uploaded_files and api_key and username:
            files = [("files", (file.name, file, file.type)) for file in uploaded_files]
            data = {"username": username, "privacy": privacy_val, "provider": provider_key_type, "api_key": api_key}
            
            with st.spinner("Embedding and Indexing..."):
                try:
                    res = requests.post(f"{BACKEND_URL}/upload/", files=files, data=data)
                    if res.status_code == 200:
                        st.success("✅ Knowledge Base Updated!")
                    else:
                        st.error("Ingestion failed.")
                except:
                    st.error("Backend connection failed.")
        else:
            st.warning("Please provide Username and API Key.")

# --- Main Interface ---
st.title("🧠 Enterprise RAG Agent")
st.caption("Advanced Retrieval Augmented Generation with Grounding & Precision Metrics")

# Display History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Display Metrics for Assistant Messages
        if msg.get("confidence", 0) > 0:
            c_score = msg["confidence"]
            col1, col2 = st.columns(2)
            with col1:
                color = "green" if c_score > 70 else "orange"
                st.markdown(f"**Confidence:** :{color}[{c_score}%]")
                st.progress(int(c_score))
            
            with st.expander("📚 Evidence (Verified Sources)"):
                for src in msg.get("sources", [])[:2]: # Show Top 2 Only
                    st.markdown(f"**Source:** `{src['source']}`")
                    st.caption(f"...{src['content'][:200]}...") # Limit text length
                    st.markdown(f"*Relevance Score: {src['score']}%*")
                    st.divider()

# Input
if prompt := st.chat_input("Ask about your documents..."):
    if not username or not api_key:
        st.error("⚠️ Authentication Required: Please enter Username and API Key in sidebar.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing vector space..."):
                payload = {
                    "query": prompt,
                    "session_id": st.session_state.session_id,
                    "username": username,
                    "provider": provider_key_type,
                    "api_key": api_key
                }
                
                try:
                    response = requests.post(f"{BACKEND_URL}/chat/", json=payload).json()
                    
                    answer = response["answer"]
                    confidence = response.get("confidence", 0)
                    sources = response.get("sources", [])
                    
                    st.markdown(answer)
                    
                    if confidence > 0:
                        col1, col2 = st.columns(2)
                        with col1:
                            color = "green" if confidence > 70 else "orange"
                            st.markdown(f"**Confidence:** :{color}[{confidence}%]")
                            st.progress(int(confidence))
                        with col2:
                            # Precision Metric as requested
                            st.metric("Retrieval Precision", f"{response.get('retrieval_quality', 0):.1f}%")

                        # Improved Source Display
                        with st.expander("📚 Evidence (Verified Sources)"):
                            if sources:
                                for src in sources[:2]: # STRICT LIMIT: Top 2
                                    st.markdown(f"**📄 {src['source']}**")
                                    st.info(f'"{src["content"][:200]}..."')
                                    st.caption(f"Match Score: {src['score']}%")
                            else:
                                st.write("No direct sources cited.")
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer, 
                        "confidence": confidence,
                        "sources": sources
                    })
                    
                except Exception as e:
                    st.error(f"Error: {e}")
