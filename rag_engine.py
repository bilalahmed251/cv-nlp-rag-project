import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

DEFAULT_POLICY_DIR = os.path.join(os.path.dirname(__file__), "data", "policies")

def load_documents(folder_path: str = DEFAULT_POLICY_DIR) -> List[Document]:
    """
    Loads all PDF and TXT safety policy documents from the specified directory.
    Includes basic error handling for missing directories or empty folders.
    """
    if not os.path.exists(folder_path):
        print(f"[Error] Directory not found: {folder_path}")
        return []

    documents = []
    
    try:
        pdf_loader = DirectoryLoader(folder_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents.extend(pdf_loader.load())
    except Exception as e:
        print(f"[Warning] Error loading PDF files: {e}")

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

def split_into_chunks(documents: List[Document], chunk_size: int = 800, chunk_overlap: int = 100) -> List[Document]:
    """
    Splits long documents into smaller readable chunks with overlap for embedding.
    """
    if not documents:
        print("[Warning] No documents to split.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"[Info] Created {len(chunks)} text chunks from documents.")
    return chunks

def create_embeddings():
    """
    Initializes a lightweight, fast open-source Sentence Transformer embedding model.
    """
    print("[Info] Initializing HuggingFace Embedding model (all-MiniLM-L6-v2)...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return embedding_model

def build_chroma_index(chunks: List[Document], embeddings) -> Chroma:
    """
    Converts text chunks into numerical vectors and builds a persistent Chroma vector index.
    """
    if not chunks:
        raise ValueError("Cannot build Chroma index with an empty list of chunks.")

    print("[Info] Building persistent Chroma vector database...")
    vector_store = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings,
        persist_directory="./chroma_data"
    )
    print("[Info] Chroma index build successful.")
    return vector_store

def search_relevant_chunks(query: str, vector_store: Chroma, top_k: int = 4) -> List[Document]:
    """
    Searches the Chroma vector database for the top_k most relevant chunks matching the query.
    """
    if not vector_store:
        print("[Error] Vector store is not initialized.")
        return []

    print(f"[Info] Searching vector DB for query: '{query}'")
    results = vector_store.similarity_search(query, k=top_k)
    return results

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
        """Loads documents and builds Chroma index if it does not exist or if config changed."""
        version_file = "./chroma_data/version.txt"
        current_version = "chunk_size=800,chunk_overlap=100,sep=\\n## "
        
        rebuild_needed = True
        if os.path.exists("./chroma_data"):
            if os.path.exists(version_file):
                with open(version_file, "r") as f:
                    if f.read().strip() == current_version:
                        rebuild_needed = False
                        
        if rebuild_needed:
            print("[Info] Building or rebuilding database due to config change or missing DB...")
            if os.path.exists("./chroma_data"):
                import shutil
                shutil.rmtree("./chroma_data", ignore_errors=True)
                
            docs = load_documents(self.policy_folder)
            if docs:
                chunks = split_into_chunks(docs)
                self.vector_store = build_chroma_index(chunks, self.embeddings)
                with open(version_file, "w") as f:
                    f.write(current_version)
            else:
                print("[Warning] RAG Engine initialized without documents.")
        else:
            print("[Info] Loading existing database from disk (version matched)...")
            self.vector_store = Chroma(
                persist_directory="./chroma_data", 
                embedding_function=self.embeddings
            )

    def add_document(self, file_path: str):
        """Loads a single document, chunks it, and adds it to the Chroma index dynamically."""
        print(f"[Info] Dynamically adding document: {file_path}")
        docs = []
        if file_path.lower().endswith(".pdf"):
            try:
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(file_path)
                docs.extend(loader.load())
            except Exception as e:
                print(f"[Error] Failed to load PDF: {e}")
        elif file_path.lower().endswith(".txt"):
            try:
                from langchain_community.document_loaders import TextLoader
                loader = TextLoader(file_path)
                docs.extend(loader.load())
            except Exception as e:
                print(f"[Error] Failed to load TXT: {e}")
        else:
            print("[Error] Unsupported file format.")
            return False

        if not docs:
            return False

        chunks = split_into_chunks(docs)
        if self.vector_store:
            self.vector_store.add_documents(chunks)
            print("[Info] Document added to existing Chroma index.")
        else:
            self.vector_store = build_chroma_index(chunks, self.embeddings)
            print("[Info] Created new Chroma index with uploaded document.")
        return True

    def query_policies(self, user_query: str, top_k: int = 4) -> str:
        """Retrieves relevant chunks and returns them concatenated as context string."""
        if not self.vector_store:
            return "No policy context available."
        
        relevant_chunks = search_relevant_chunks(user_query, self.vector_store, top_k=top_k)
        context_texts = []
        for i, doc in enumerate(relevant_chunks):
            source = doc.metadata.get('source', 'Unknown Source')
            source_file = os.path.basename(source)
            page = doc.metadata.get('page', 'N/A')
            context_texts.append(f"--- Rule Snippet {i+1} (Source: {source_file}, Page: {page}) ---\n{doc.page_content}")
            
        return "\n\n".join(context_texts)

if __name__ == "__main__":
    print("\n================ RAG ENGINE TEST ================")
    docs = load_documents()
    
    if docs:
        chunks = split_into_chunks(docs)
        embeddings = create_embeddings()
        vector_db = build_chroma_index(chunks, embeddings)
        
        sample_query = "What is the penalty for not wearing a helmet?"
        results = search_relevant_chunks(sample_query, vector_db, top_k=2)
        
        print("\n--- SAMPLE QUERY RESULTS ---")
        for idx, res in enumerate(results):
            print(f"\n[Result {idx + 1}]:")
            print(res.page_content)
    else:
        print("[Error] Could not run test because no documents were loaded.")
    
    print("\n================ TEST COMPLETE ================")
