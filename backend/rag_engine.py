import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS

from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.schema import Document
from langchain_classic.prompts import PromptTemplate


# Folder to save the database permanently
DB_PATH = "faiss_db_store"

class RAGManager:
    def __init__(self):
        """Initialize and try to load existing DB from disk."""
        self.vector_store = None
        self.embeddings = None
        # Hum baad mein embeddings init karein gy jab request ayegi,
        # lekin agar disk par DB hai to load karna padega.

    def _get_embeddings(self, provider, api_key):
        """Helper to get embedding object."""
        if provider == "openai":
            return OpenAIEmbeddings(openai_api_key=api_key)
        elif provider == "gemini":
            return GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=api_key)
        return None

    def load_existing_db(self, provider, api_key):
        """Loads the FAISS index from disk if it exists."""
        if os.path.exists(DB_PATH):
            embeddings = self._get_embeddings(provider, api_key)
            try:
                self.vector_store = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
                print("✅ Loaded existing database from disk.")
            except Exception as e:
                print(f"⚠️ Could not load existing DB: {e}")
                self.vector_store = None
        else:
            print("ℹ️ No existing database found. Starting fresh.")

    def process_files(self, file_paths, username, privacy, provider, api_key):
        """
        Reads files (PDF, TXT, DOCX), adds metadata, and saves to disk.
        """
        documents = []
        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            
            # --- NEW: Check File Type ---
            if file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith(".docx"):  # Word Files Support
                loader = Docx2txtLoader(file_path)
            else:
                loader = TextLoader(file_path)
            # ----------------------------
            
            try:
                docs = loader.load()
                # Intelligent Metadata Tagging
                for doc in docs:
                    doc.metadata["source"] = file_name
                    doc.metadata["owner"] = username  # "zain"
                    doc.metadata["privacy"] = privacy # "private" or "public"
                documents.extend(docs)
            except Exception as e:
                print(f"❌ Error loading file {file_name}: {e}")
                continue

        if not documents:
            return 0

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)

        embeddings = self._get_embeddings(provider, api_key)

        # Update or Create Vector Store
        if self.vector_store is None:
            # Try loading first
            if os.path.exists(DB_PATH):
                try:
                    self.vector_store = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
                    self.vector_store.add_documents(splits)
                except:
                    # Agar load fail hojaye to naya banao
                    self.vector_store = FAISS.from_documents(splits, embeddings)
            else:
                self.vector_store = FAISS.from_documents(splits, embeddings)
        else:
            self.vector_store.add_documents(splits)
        
        # SAVE TO DISK (Persistence Magic)
        self.vector_store.save_local(DB_PATH)
        
        return len(splits)

    def get_answer(self, query, history, username, provider, api_key):
        """
        Retrieves context using Intelligent Filtering (Privacy Check).
        """
        embeddings = self._get_embeddings(provider, api_key)
        
        # Load DB if not in memory
        if self.vector_store is None:
            self.load_existing_db(provider, api_key)

        if not self.vector_store:
            return {
                "answer": "No knowledge base found. Please upload documents first.", 
                "sources": [], 
                "confidence": 0.0
            }

        # 1. Retrieve MORE docs first (fetch 15), then filter manually
        docs_and_scores = self.vector_store.similarity_search_with_score(query, k=15)
        
        filtered_results = []
        
        # 2. Apply Privacy Logic
        for doc, score in docs_and_scores:
            doc_owner = doc.metadata.get("owner", "unknown")
            doc_privacy = doc.metadata.get("privacy", "private")
            
            # Logic: Show if (It's MY file) OR (It's a PUBLIC file)
            if doc_owner == username or doc_privacy == "public":
                filtered_results.append((doc, score))
            
            if len(filtered_results) >= 4: # Stop after finding 4 good matches
                break
        
        if not filtered_results:
             return {
                "answer": "I couldn't find any relevant information that you have access to.", 
                "sources": [], 
                "confidence": 0.0
            }

        context_text = ""
        sources = []
        min_distance = 100.0
        
        for doc, score in filtered_results:
            py_score = float(score)
            
            # Metadata for UI
            source_label = f"{doc.metadata.get('source')} ({doc.metadata.get('privacy').upper()})"
            
            context_text += f"\n---\nSource: {source_label}\nContent: {doc.page_content}"
            sources.append({
                "source": source_label, 
                "content": doc.page_content, 
                "score": py_score
            })
            
            if py_score < min_distance:
                min_distance = py_score

        # 3. Calculate Confidence
        min_distance = float(min_distance)
        
        # NOTE: Maine wahi purana formula rakha hai jo aapne diya tha.
        # Agar aapko Green Score chahiye to mera "Smart Formula" use karein,
        # lekin abhi aapne kaha "Logic same rakhna", to ye raha original:
        confidence = max(0.0, (1.0 - min_distance) * 100.0)

        # 4. Generate Answer
        if provider == "openai":
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, openai_api_key=api_key)
        else:
            # Gemini Model name standard kar diya hai
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, google_api_key=api_key)

        prompt_template = """
        You are an intelligent Corporate Assistant. Answer the question strictly based on the provided Context.
        
        Chat History:
        {history}

        Context:
        {context}

        Question: {question}

        Guidelines:
        - If the answer is not in the context, strictly say "I cannot find the answer in the provided documents."
        - Cite the specific document name if available in the context.

        Answer:
        """
        
        prompt = PromptTemplate(template=prompt_template, input_variables=["history", "context", "question"])
        final_prompt = prompt.format(history=history, context=context_text, question=query)
        
        response = llm.invoke(final_prompt)
        
        return {
            "answer": response.content,
            "sources": sources,
            "confidence": float(round(confidence, 2))
        }

