import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Configurable Default Directory
DEFAULT_POLICY_DIR = os.path.join(os.path.dirname(__file__), "data", "policies")


# --- 1. Load Documents ---
def load_documents(folder_path: str = DEFAULT_POLICY_DIR) -> List[Document]:
    """
    Loads all PDF and TXT safety policy documents from the specified directory.
    Includes basic error handling for missing directories or empty folders.
    """
    if not os.path.exists(folder_path):
        print(f"[Error] Directory not found: {folder_path}")
        return []

    documents = []
    
    # Load PDF files
    try:
        pdf_loader = DirectoryLoader(folder_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents.extend(pdf_loader.load())
    except Exception as e:
        print(f"[Warning] Error loading PDF files: {e}")

    # Load TXT files
    try:
        txt_loader = DirectoryLoader(folder_path, glob="**/*.txt", loader_cls=TextLoader)
        documents.extend(txt_loader.load())
    except Exception as e:
        print(f"[Warning] Error loading TXT files: {e}")

    if not documents:
        print(f"[Warning] No documents found in {folder_path}")
    else:
        print(f"[Info] Successfully loaded {len(documents)} document page(s)/file(s).")

    return documents


# --- 2. Split Text into Chunks ---
def split_into_chunks(documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
    """
    Splits long documents into smaller readable chunks with overlap for embedding.
    """
    if not documents:
        print("[Warning] No documents to split.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"[Info] Created {len(chunks)} text chunks from documents.")
    return chunks


# --- 3. Create Embedding Model ---
def create_embeddings():
    """
    Initializes a lightweight, fast open-source Sentence Transformer embedding model.
    """
    print("[Info] Initializing HuggingFace Embedding model (all-MiniLM-L6-v2)...")
    # Uses a compact model that runs fast on CPU
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return embedding_model


# --- 4. Build FAISS Vector Database ---
def build_faiss_index(chunks: List[Document], embeddings) -> FAISS:
    """
    Converts text chunks into numerical vectors and builds an in-memory FAISS vector index.
    """
    if not chunks:
        raise ValueError("Cannot build FAISS index with an empty list of chunks.")

    print("[Info] Building FAISS vector database...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    print("[Info] FAISS index build successful.")
    return vector_store


# --- 5. Search Relevant Chunks ---
def search_relevant_chunks(query: str, vector_store: FAISS, top_k: int = 3) -> List[Document]:
    """
    Searches the FAISS vector database for the top_k most relevant chunks matching the query.
    """
    if not vector_store:
        print("[Error] Vector store is not initialized.")
        return []

    print(f"[Info] Searching vector DB for query: '{query}'")
    results = vector_store.similarity_search(query, k=top_k)
    return results


# --- Class-based Wrapper for easy integration with UI/NLP Engine ---
class RAGEngine:
    """
    High-level RAG Engine class orchestrating document loading, vector storage, and retrieval.
    """
    def __init__(self, policy_folder: str = DEFAULT_POLICY_DIR):
        self.policy_folder = policy_folder
        self.embeddings = create_embeddings()
        self.vector_store = None
        self.initialize_engine()

    def initialize_engine(self):
        """Loads documents and builds FAISS index."""
        docs = load_documents(self.policy_folder)
        if docs:
            chunks = split_into_chunks(docs)
            self.vector_store = build_faiss_index(chunks, self.embeddings)
        else:
            print("[Warning] RAG Engine initialized without documents.")

    def query_policies(self, user_query: str, top_k: int = 3) -> str:
        """Retrieves relevant chunks and returns them concatenated as context string."""
        if not self.vector_store:
            return "No policy context available."
        
        relevant_chunks = search_relevant_chunks(user_query, self.vector_store, top_k=top_k)
        context_texts = [f"--- Rule Snippet {i+1} ---\n" + doc.page_content for i, doc in enumerate(relevant_chunks)]
        return "\n\n".join(context_texts)


# --- 6 & 7. Test script if run directly ---
if __name__ == "__main__":
    print("\n================ RAG ENGINE TEST ================")
    
    # 1. Load documents
    docs = load_documents()
    
    if docs:
        # 2. Split chunks
        chunks = split_into_chunks(docs)
        
        # 3. Create embeddings
        embeddings = create_embeddings()
        
        # 4. Build FAISS index
        vector_db = build_faiss_index(chunks, embeddings)
        
        # 5. Test search query
        sample_query = "What is the penalty for not wearing a helmet?"
        results = search_relevant_chunks(sample_query, vector_db, top_k=2)
        
        print("\n--- SAMPLE QUERY RESULTS ---")
        for idx, res in enumerate(results):
            print(f"\n[Result {idx + 1}]:")
            print(res.page_content)
    else:
        print("[Error] Could not run test because no documents were loaded.")
    
    print("\n================ TEST COMPLETE ================")
