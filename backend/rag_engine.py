import os
import logging
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Folder to save the database
DB_PATH = "faiss_db_store"

class RAGManager:
    def __init__(self):
        self.vector_store = None

    def _get_embeddings(self, provider, api_key):
        if provider == "openai":
            return OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
        elif provider == "gemini":
            return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        return None
    
    def _get_llm(self, provider, api_key, temperature=0.1):
        if provider == "openai":
            return ChatOpenAI(model="gpt-4o-mini", temperature=temperature, openai_api_key=api_key)
        elif provider == "gemini":
            return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=temperature, google_api_key=api_key)
        return None

    def _calculate_confidence(self, distance):
        """
        Advanced Confidence Scoring.
        Converts Vector L2 Distance to Percentage.
        """
        try:
            dist = float(distance)
            if dist < 0: dist = 0.0
            # Formula: 1 / (1 + distance) maps distance to 0-100 scale gracefully
            score = 1.0 / (1.0 + dist)
            return float(round(score * 100, 2))
        except:
            return 0.0

    def _format_history(self, history_list):
        history_text = ""
        for msg in history_list:
            role = "User" if msg['role'] == 'user' else "Assistant"
            content = str(msg['content'])
            history_text += f"{role}: {content}\n"
        return history_text if history_text else "No previous chat history."

    def load_existing_db(self, provider, api_key):
        if os.path.exists(DB_PATH):
            embeddings = self._get_embeddings(provider, api_key)
            try:
                self.vector_store = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
                print("✅ Database loaded successfully.")
            except Exception as e:
                print(f"⚠️ Database load error: {e}")
                self.vector_store = None

    def process_files(self, file_paths, username, privacy, provider, api_key):
        documents = []
        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            try:
                if file_path.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                elif file_path.endswith(".docx"):
                    loader = Docx2txtLoader(file_path)
                else:
                    loader = TextLoader(file_path)
                
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = file_name
                    doc.metadata["owner"] = username
                    doc.metadata["privacy"] = privacy
                documents.extend(docs)
            except Exception as e:
                print(f"❌ Error processing {file_name}: {e}")
                continue

        if not documents:
            return 0

        # Advanced Splitting: Overlap is crucial for context continuity
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

    def _generate_query_variations(self, original_query, llm):
        """
        ADVANCED TECHNIQUE: Multi-Query Expansion
        Generates 3 different versions of the user's question to improve retrieval coverage.
        """
        prompt = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI language model assistant. Your task is to generate 3 different versions of the given user question to retrieve relevant documents from a vector database. 
            By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. 
            Provide these alternative questions separated by newlines.
            Original question: {question}"""
        )
        
        try:
            # We use a chain to get variations
            response = llm.invoke(prompt.format(question=original_query))
            variations = response.content.split('\n')
            # Clean up and keep non-empty lines
            cleaned_variations = [v.strip() for v in variations if v.strip()]
            return cleaned_variations[:3] # Limit to 3 variations
        except:
            return [original_query]

    def get_answer(self, query, history, username, provider, api_key):
        embeddings = self._get_embeddings(provider, api_key)
        llm = self._get_llm(provider, api_key, temperature=0.2)
        
        if self.vector_store is None:
            self.load_existing_db(provider, api_key)

        if not self.vector_store:
            return {"answer": "⚠️ Database is empty.", "sources": [], "confidence": 0.0}

        # --- STEP 1: MULTI-QUERY EXPANSION ---
        # Instead of searching once, we search for synonyms too.
        # e.g., "Delivery Time" -> searches for "Time Limit", "Deadline", "Schedule"
        queries_to_search = [query]
        
        # Only expand if query is short/ambiguous
        if len(query.split()) < 10:
            variations = self._generate_query_variations(query, llm)
            queries_to_search.extend(variations)

        # --- STEP 2: AGGREGATE SEARCH RESULTS ---
        unique_docs = {} # Deduplication map
        
        for q in queries_to_search:
            docs_and_scores = self.vector_store.similarity_search_with_score(q, k=4)
            for doc, distance in docs_and_scores:
                # Privacy Check
                doc_owner = doc.metadata.get("owner", "unknown")
                doc_privacy = doc.metadata.get("privacy", "private")
                has_access = (doc_owner == username) or (doc_privacy == "public")
                
                if has_access:
                    # Use content as key to prevent duplicates
                    content_hash = hash(doc.page_content)
                    if content_hash not in unique_docs:
                        unique_docs[content_hash] = (doc, distance)
                    else:
                        # Keep the one with better score (lower distance)
                        if distance < unique_docs[content_hash][1]:
                            unique_docs[content_hash] = (doc, distance)

        # Convert back to list and sort
        results = list(unique_docs.values())
        results.sort(key=lambda x: x[1]) # Sort by distance (asc)
        top_results = results[:4] # Take top 4 unique matches

        if not top_results:
             return {"answer": "I couldn't find relevant info.", "sources": [], "confidence": 0.0}

        # --- STEP 3: PREPARE CONTEXT & SCORES ---
        best_distance = float(top_results[0][1])
        confidence = self._calculate_confidence(best_distance)
        
        context_text = ""
        sources = []
        
        for doc, dist in top_results:
            score = self._calculate_confidence(dist)
            source_label = doc.metadata.get('source', 'Unknown')
            
            context_text += f"\n---\n[Source: {source_label}]\nContent: {doc.page_content}\n"
            
            sources.append({
                "source": source_label, 
                "content": doc.page_content, 
                "score": score
            })

        avg_precision = sum([s['score'] for s in sources]) / len(sources) if sources else 0.0

        # --- STEP 4: CHAIN OF THOUGHT PROMPT ---
        formatted_history = self._format_history(history)
        
        prompt_template = """
        You are an advanced Corporate RAG Assistant. 
        Your task is to answer the user's question using the retrieved Context.
        
        STRATEGY:
        1. **Analyze:** Look at the user's question and the Context chunks.
        2. **Connect:** Synonyms matter. If user asks "Delivery" and Context says "Time Limit", match them.
        3. **Answer:** Provide a direct answer based on the context.
        4. **Cite:** Always include the [Source Filename] for verification.
        
        STRICT RULES:
        - If the Context supports the answer, give it confidently.
        - If the Context is completely unrelated, say "I cannot find this information in the documents."
        
        Chat History:
        {history}

        Context from Documents:
        {context}

        User Question: {question}

        Answer:
        """
        
        prompt = PromptTemplate(template=prompt_template, input_variables=["history", "context", "question"])
        final_prompt = prompt.format(history=formatted_history, context=context_text, question=query)
        
        try:
            response = llm.invoke(final_prompt)
            answer_text = response.content
        except Exception as e:
            answer_text = f"Error from LLM: {str(e)}"

        return {
            "answer": answer_text,
            "sources": sources,
            "confidence": confidence,
            "retrieval_quality": float(round(avg_precision, 2))
        }
