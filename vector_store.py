import glob
import os

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
TAVILY_API_KEY = os.environ['TAVILY_API_KEY']

def get_all_pdfs(directory: str) -> list[str]:
    """
    Recursively find all PDF files in the specified directory.
    
    Args:
        directory (str): The base directory to search for PDFs.
    
    Returns:
        list[str]: A list of paths to PDF files.
    """
    return glob.glob(f"{directory}/**/*.pdf", recursive=True)

def load_clark_vectorstore(db_path: str, base_dir: str) -> FAISS:
    """
    Load or create a FAISS vector store for the CLARK library from PDF files.
    
    Args:
        db_path (str): Path to the FAISS vector store database.
        base_dir (str): Base directory containing CLARK library PDFs.
    
    Returns:
        FAISS: The loaded or newly created vector store.
    """
    if os.path.exists(db_path):
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(
            db_path, embeddings, allow_dangerous_deserialization=True
        )
        print(f"CLARK Vector DB loaded from {db_path}")
    else:
        pdf_files = get_all_pdfs(base_dir)
        if not pdf_files:
            print(f"Warning: No PDF files found in {base_dir} or its subdirectories")
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(
                [Document(page_content="Empty document")], embeddings
            )
            vectorstore.save_local(db_path)
            return vectorstore
        
        print(f"Found {len(pdf_files)} PDF files in {base_dir}")
        documents = []
        for pdf_path in pdf_files:
            try:
                # Extract metadata from directory structure
                parts = pdf_path.split(os.sep)
                if len(parts) >= 4:
                    collection_name = parts[-4]
                    course_name = parts[-3]
                    module_name = parts[-2]
                    file_name = parts[-1]
                else:
                    collection_name = "unknown_collection"
                    course_name = "unknown_course"
                    module_name = "unknown_module"
                    file_name = os.path.basename(pdf_path)

                loader = PyPDFLoader(pdf_path)
                pages = loader.load()
                for page in pages:
                    page.metadata.update({
                        "collection_name": collection_name,
                        "course_name": course_name,
                        "module_name": module_name,
                        "file_path": pdf_path,
                        "file_name": file_name
                    })
                documents.extend(pages)
                print(f"Loaded {pdf_path} with metadata")
            except Exception as e:
                print(f"Error loading {pdf_path}: {e}")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(db_path)
        print(f"CLARK Vector DB created at {db_path} with {vectorstore.index.ntotal} documents")
    
    return vectorstore

# Define paths and initialize the vector store
clark_db_path = "db/clark_db/clark_library_db"
clark_base_dir = "rag/clark_doc"
clark_vectorstore = load_clark_vectorstore(clark_db_path, clark_base_dir)