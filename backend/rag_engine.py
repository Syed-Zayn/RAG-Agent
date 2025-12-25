import os
import numpy as np
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.prompts import PromptTemplate

from langchain_classic.schema import Document

# Folder to save the database permanently
DB_PATH = "faiss_db_store"

class RAGManager:
    def __init__(self):
        self.vector_store = None

    def _get_embeddings(self, provider, api_key):
        if provider == "openai":
            return OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
        elif provider == "gemini":
            return GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=api_key)
        return None

    def _calculate_confidence(self, distance):
        """
        Boosted Confidence Score Logic (0-100%)
        """
        dist = float(distance)
        if dist < 0: dist = 0.0
        # Formula: Closer to 0 is better. 
        # Map 0.4 distance -> ~71% confidence (Green)
        # Map 0.8 distance -> ~55% confidence (Orange)
        score = 1.0 / (1.0 + dist)
        return float(round(score * 100, 2))

    def _format_history(self, history_list):
        """
        Convert raw List of Dicts to Clean Dialogue String.
        Crucial for preventing LLM confusion.
        """
        history_text = ""
        for msg in history_list:
            role = "User" if msg['role'] == 'user' else "Assistant"
            content = msg['content']
            history_text += f"{role}: {content}\n"
        return history_text if history_text else "No previous chat history."

    def load_existing_db(self, provider, api_key):
        if os.path.exists(DB_PATH):
            embeddings = self._get_embeddings(provider, api_key)
            try:
                self.vector_store = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
                print("✅ Loaded existing database from disk.")
            except Exception as e:
                print(f"⚠️ Could not load existing DB: {e}")
                self.vector_store = None
        else:
            print("ℹ️ No existing database found.")

    def process_files(self, file_paths, username, privacy, provider, api_key):
        documents = []
        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            
            if file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            else:
                loader = TextLoader(file_path)
            
            try:
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = file_name
                    doc.metadata["owner"] = username
                    doc.metadata["privacy"] = privacy
                documents.extend(docs)
            except Exception as e:
                print(f"❌ Error loading file {file_name}: {e}")
                continue

        if not documents:
            return 0

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)

        embeddings = self._get_embeddings(provider, api_key)

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
        embeddings = self._get_embeddings(provider, api_key)
        
        if self.vector_store is None:
            self.load_existing_db(provider, api_key)

        if not self.vector_store:
            return {
                "answer": "⚠️ System is empty. Please upload documents first.", 
                "sources": [], 
                "confidence": 0.0,
                "retrieval_quality": 0.0
            }

        # 1. Retrieve Candidate Chunks
        docs_and_scores = self.vector_store.similarity_search_with_score(query, k=10)
        
        filtered_results = []
        
        # 2. Filter by Access
        for doc, distance in docs_and_scores:
            doc_owner = doc.metadata.get("owner", "unknown")
            doc_privacy = doc.metadata.get("privacy", "private")
            
            has_access = (doc_owner == username) or (doc_privacy == "public")
            
            if has_access:
                filtered_results.append((doc, distance))

        if not filtered_results:
             return {
                "answer": "I couldn't find any relevant information in your accessible documents.", 
                "sources": [], 
                "confidence": 0.0,
                "retrieval_quality": 0.0
            }

        # 3. Sort and Select Top 3
        filtered_results.sort(key=lambda x: x[1])
        top_results = filtered_results[:3]

        best_distance = float(top_results[0][1])
        confidence = self._calculate_confidence(best_distance)
        
        # NO BLOCKING: We trust the LLM to filter irrelevance now
        
        context_text = ""
        sources = []
        
        for doc, dist in top_results:
            score = self._calculate_confidence(dist)
            source_name = f"{doc.metadata.get('source')}"
            
            context_text += f"\n---\n[Source: {source_name}]\nContent: {doc.page_content}\n"
            
            sources.append({
                "source": source_name, 
                "content": doc.page_content, 
                "score": score
            })

        avg_precision = sum([s['score'] for s in sources]) / len(sources) if sources else 0.0

        # 4. LLM Generation
        # Temperature 0.3 allows slight flexibility to connect "Delivery" with "Time Limit"
        if provider == "openai":
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, openai_api_key=api_key)
        else:
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, google_api_key=api_key)

        # 5. FIXED PROMPT WITH FORMATTED HISTORY
        formatted_history = self._format_history(history)
        
        prompt_template = """
        You are an intelligent Corporate Assistant. 
        Your goal is to answer the user's question using the provided Context.

        INSTRUCTIONS:
        1. Read the Context carefully. If the answer is present, output it clearly.
        2. Don't be too strict. If the user asks "What is RAG" and the document mentions "Mini-RAG Assistant Objective...", that IS the answer. Use it.
        3. If the user asks about "Delivery" or "Time", look for "Time Limit" or "Deadlines".
        4. Cite sources like [Source Name].

        Chat History:
        {history_text}

        Context:
        {context}

        Question: {question}

        Answer:
        """
        
        prompt = PromptTemplate(template=prompt_template, input_variables=["history_text", "context", "question"])
        final_prompt = prompt.format(history_text=formatted_history, context=context_text, question=query)
        
        try:
            response = llm.invoke(final_prompt)
            answer_text = response.content
        except Exception as e:
            answer_text = f"Error generating response: {str(e)}"

        return {
            "answer": answer_text,
            "sources": sources,
            "confidence": float(confidence),
            "retrieval_quality": float(round(avg_precision, 2))
        }
