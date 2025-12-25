import streamlit as st
import requests
import uuid
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Enterprise RAG Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- BACKEND CONNECTION ---
# Backend URL (Priority: Secrets -> Env -> Default)
if "BACKEND_URL" in st.secrets:
    BACKEND_URL = st.secrets["BACKEND_URL"]
else:
    BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# --- SESSION STATE INITIALIZATION ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- HELPER: SMART KEY LOADER ---
def get_api_key(provider_type):
    """
    Tries to load API Key from:
    1. Streamlit Secrets (Best for Cloud)
    2. Environment Variables (Best for Local)
    3. Return None if not found (Falls back to Manual Input)
    """
    key_name = "OPENAI_API_KEY" if provider_type == "openai" else "GOOGLE_API_KEY"
    
    # Check Streamlit Secrets first
    if key_name in st.secrets:
        return st.secrets[key_name]
    
    # Check OS Environment
    if os.getenv(key_name):
        return os.getenv(key_name)
        
    return None

# --- SIDEBAR: CONFIGURATION ---
with st.sidebar:
    st.title("🎛️ Control Panel")
    
    # 1. User Identity
    st.subheader("👤 User Profile")
    username = st.text_input(
        "Username", 
        value=st.session_state.get("username", ""), 
        placeholder="e.g., admin_user",
        help="Used for Role-Based Access Control (RBAC)"
    )
    if username:
        st.session_state.username = username # Persist in session

    st.divider()

    # 2. AI Provider Settings
    st.subheader("🧠 Model Settings")
    provider_option = st.selectbox("Select AI Provider", ["OpenAI (GPT-4o)", "Google (Gemini 1.5)"])
    provider_key_type = "openai" if "OpenAI" in provider_option else "gemini"

    # Smart Key Logic
    system_key = get_api_key(provider_key_type)
    
    if system_key:
        api_key = system_key
        st.success(f"✅ {provider_key_type.title()} Key Active (System Managed)")
    else:
        api_key = st.text_input(f"🔑 Enter {provider_key_type.title()} Key", type="password")
        if not api_key:
            st.warning("⚠️ API Key required for processing.")

    st.divider()

    # 3. Document Ingestion
    st.subheader("📂 Knowledge Base")
    privacy_mode = st.radio("Access Level:", ("Private (Session Only)", "Public (Organization)"), index=0)
    privacy_val = "private" if "Private" in privacy_mode else "public"
    
    uploaded_files = st.file_uploader("Upload Documents (PDF/DOCX/TXT)", accept_multiple_files=True)
    
    if st.button("🚀 Process & Ingest", type="primary"):
        if not username:
            st.error("❌ Username is required!")
        elif not api_key:
            st.error("❌ API Key is missing!")
        elif not uploaded_files:
            st.error("❌ Please select files first.")
        else:
            files = [("files", (file.name, file, file.type)) for file in uploaded_files]
            data = {
                "username": username,
                "privacy": privacy_val,
                "provider": provider_key_type,
                "api_key": api_key
            }
            
            with st.status("Processing Documents...", expanded=True) as status:
                try:
                    st.write("📤 Uploading files...")
                    res = requests.post(f"{BACKEND_URL}/upload/", files=files, data=data)
                    
                    if res.status_code == 200:
                        status.update(label="✅ Ingestion Complete!", state="complete", expanded=False)
                        st.success(res.json()["message"])
                    else:
                        status.update(label="❌ Upload Failed", state="error")
                        st.error(res.text)
                except Exception as e:
                    status.update(label="❌ Connection Error", state="error")
                    st.error(f"Could not connect to Backend: {e}")

# --- MAIN CHAT INTERFACE ---

# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🤖 Corporate RAG Assistant")
    st.caption("Secure Retrieval Augmented Generation with Precision Metrics")
with col2:
    if username:
        st.markdown(f"**Logged in as:** `{username}`")
        st.markdown(f"**Mode:** `{privacy_mode}`")

st.divider()

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Advanced Metrics Display for Assistant
        if msg.get("role") == "assistant" and msg.get("confidence", 0) > 0:
            c_score = msg["confidence"]
            
            # Metric Columns
            m_col1, m_col2, m_col3 = st.columns([2, 2, 4])
            with m_col1:
                st.metric("Confidence", f"{c_score}%")
            with m_col2:
                # Color coding based on score
                p_color = "green" if c_score > 70 else "orange" if c_score > 40 else "red"
                st.caption(f"Grounding Health: :{p_color}[{p_color.upper()}]")
                st.progress(int(c_score))
            
            # Sources Expander
            with st.expander("📚 View Verified Sources (Evidence)"):
                sources = msg.get("sources", [])
                if sources:
                    for idx, src in enumerate(sources[:2]): # Limit to top 2
                        st.markdown(f"**{idx+1}. {src['source']}** (Match: {src['score']}%)")
                        st.info(f"{src['content'][:250]}...") # Truncate text
                else:
                    st.write("No direct sources cited from documents.")

# Input Field
if prompt := st.chat_input("Ask a question about your uploaded documents..."):
    if not api_key:
        st.toast("❌ API Key is missing! Check sidebar.", icon="⚠️")
    elif not username:
        st.toast("❌ Username is required! Check sidebar.", icon="👤")
    else:
        # Add User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            with st.spinner("🧠 Analyzing Knowledge Base..."):
                payload = {
                    "query": prompt,
                    "session_id": st.session_state.session_id,
                    "username": username,
                    "provider": provider_key_type,
                    "api_key": api_key
                }
                
                try:
                    response = requests.post(f"{BACKEND_URL}/chat/", json=payload)
                    
                    if response.status_code == 200:
                        data = response.json()
                        answer = data["answer"]
                        confidence = data.get("confidence", 0)
                        sources = data.get("sources", [])
                        
                        # Display Answer
                        message_placeholder.markdown(answer)
                        
                        # Display Metrics if answer is grounded
                        if confidence > 0:
                            m_col1, m_col2, m_col3 = st.columns([2, 2, 4])
                            with m_col1:
                                st.metric("Confidence", f"{confidence}%")
                            with m_col2:
                                p_color = "green" if confidence > 70 else "orange"
                                st.caption("Relevance Score")
                                st.progress(int(confidence))
                            
                            with st.expander("📚 Verified Sources"):
                                for src in sources[:2]:
                                    st.markdown(f"**📄 {src['source']}**")
                                    st.caption(f'"{src["content"][:200]}..."')
                                    st.divider()

                        # Save to History
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "confidence": confidence,
                            "sources": sources
                        })
                        
                    else:
                        error_msg = f"Server Error: {response.text}"
                        message_placeholder.error(error_msg)
                        
                except Exception as e:
                    message_placeholder.error(f"Connection Failed: {str(e)}")
