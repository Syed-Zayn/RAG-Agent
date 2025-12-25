import os
import math
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.prompts import PromptTemplate

# Folder to save the database permanently
DB_PATH = "faiss_db_store"

class RAGManager:
    def __init__(self):
        """Initialize and try to load existing DB from disk."""
        self.vector_store = None
        self.embeddings = None

    def _get_embeddings(self, provider, api_key):
        """Helper to get embedding object based on provider."""
        if provider == "openai":
            return OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
        elif provider == "gemini":
            return GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=api_key)
        return None

    def _calculate_advanced_score(self, distance):
        """
        ADVANCED SCORING MECHANISM: Exponential Decay.
        Converts L2 Distance (0 to infinity) into a Probability (0% to 100%).
        
        Formula: Score = e^(-1.5 * distance) * 100
        
        Why this logic?
        - Small distances (high similarity) get very high scores.
        - As distance increases, score drops rapidly (penalizing weak matches).
        """
        # Sensitivity factor (1.5 is strict, 1.0 is lenient)
        sensitivity = 0.7  
        score = math.exp(-sensitivity * distance) * 100
        return round(score, 2)

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
            
            # Detect File Type
            if file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            else:
                loader = TextLoader(file_path)
            
            try:
                docs = loader.load()
                # Intelligent Metadata Tagging (RBAC Core)
                for doc in docs:
                    doc.metadata["source"] = file_name
                    doc.metadata["owner"] = username
                    doc.metadata["privacy"] = privacy # "private" or "public"
                documents.extend(docs)
            except Exception as e:
                print(f"❌ Error loading file {file_name}: {e}")
                continue

        if not documents:
            return 0

        # Chunking Strategy: Optimized for Context Retention
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)

        embeddings = self._get_embeddings(provider, api_key)

        # Update or Create Vector Store
        if self.vector_store is None:
            if os.path.exists(DB_PATH):
                try:
                    self.vector_store = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
                    self.vector_store.add_documents(splits)
                except:
                    self.vector_store = FAISS.from_documents(splits, embeddings)
            else:
                self.vector_store = FAISS.from_documents(splits, embeddings)
        else:
            self.vector_store.add_documents(splits)
        
        self.vector_store.save_local(DB_PATH)
        return len(splits)

    def get_answer(self, query, history, username, provider, api_key):
        """
        Retrieves context with STRICT filtering and Advanced Scoring.
        """
        # 1. Init Logic
        embeddings = self._get_embeddings(provider, api_key)
        if self.vector_store is None:
            self.load_existing_db(provider, api_key)

        if not self.vector_store:
            return {
                "answer": "Knowledge base is empty. Please upload documents first.", 
                "sources": [], 
                "confidence": 0.0
            }

        # 2. Retrieve Broad Set (k=15) then Filter (RBAC)
        docs_and_scores = self.vector_store.similarity_search_with_score(query, k=15)
        
        filtered_results = []
        
        # 3. Privacy & Owner Filter
        for doc, distance in docs_and_scores:
            doc_owner = doc.metadata.get("owner", "unknown")
            doc_privacy = doc.metadata.get("privacy", "private")
            
            # RBAC Logic: Access if (File is Mine) OR (File is Public)
            if doc_owner == username or doc_privacy == "public":
                # Convert Distance to Relevance Score immediately
                relevance = self._calculate_advanced_score(float(distance))
                filtered_results.append((doc, relevance))
            
            if len(filtered_results) >= 4: # Only keep Top 4 relevant chunks
                break
        
        # 4. STRICT THRESHOLD CHECK (Safety Guardrail)
        # If no documents found OR the best match is too weak (< 45% confidence)
        if not filtered_results or filtered_results[0][1] < 45.0:
             return {
                "answer": "I could not find sufficiently relevant information in your documents to answer this question accurately.", 
                "sources": [], 
                "confidence": 0.0 
            }

        # 5. Prepare Context for LLM
        context_text = ""
        sources = []
        
        # Calculate Overall Confidence (Weighted Average of Top 2)
        top_scores = [score for _, score in filtered_results[:2]]
        avg_confidence = sum(top_scores) / len(top_scores) if top_scores else 0.0
        
        for doc, score in filtered_results:
            source_label = f"{doc.metadata.get('source')} ({doc.metadata.get('privacy').upper()})"
            context_text += f"\n---\nSource: {source_label}\nContent: {doc.page_content}"
            
            sources.append({
                "source": source_label, 
                "content": doc.page_content, 
                "score": score  # Precision@k metric
            })

        # 6. LLM Generation
        if provider == "openai":
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, openai_api_key=api_key)
        else:
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, google_api_key=api_key)

        prompt_template = """
        You are a specialized Corporate RAG Assistant. 
        Your task is to answer the user's question ONLY using the provided Context.
        
        STRICT RULES:
        1. If the answer is NOT in the Context, explicitly say "I cannot find the answer in the provided documents."
        2. Do not hallucinate or use outside knowledge.
        3. Keep the tone professional and concise.

        Chat History:
        {history}

        Context:
        {context}

        Question: {question}

        Answer:
        """
        
        prompt = PromptTemplate(template=prompt_template, input_variables=["history", "context", "question"])
        final_prompt = prompt.format(history=history, context=context_text, question=query)
        
        try:
            response = llm.invoke(final_prompt)
            answer_text = response.content
        except Exception as e:
            answer_text = f"Error generating response: {e}"

        return {
            "answer": answer_text,
            "sources": sources,
            "confidence": float(round(avg_confidence, 2))
        }

