import glob
import os
from typing import List

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel

load_dotenv()
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

def get_all_pdfs(directory: str) -> List[str]:
    """Recursively find all PDF files in the given directory."""
    return glob.glob(f"{directory}/**/*.pdf", recursive=True)

def load_clark_vectorstore(db_path: str, base_dir: str):
    """
    Load an existing FAISS vector store from db_path or create a new one from PDFs in base_dir.
    
    Args:
        db_path (str): Path to the vector store database.
        base_dir (str): Directory containing PDF files.
    
    Returns:
        FAISS: The loaded or newly created vector store.
    """
    if os.path.exists(db_path):
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(
            db_path, embeddings, allow_dangerous_deserialization=True
        )
        print(f"CLARK Vector DB loaded at {db_path}")
    else:
        pdf_files = get_all_pdfs(base_dir)
        if not pdf_files:
            print(f"Warning: No PDF files found in {base_dir} or its subdirectories")
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(
                [Document(page_content="Empty document")], embedding=embeddings
            )
            vectorstore.save_local(db_path)
            return vectorstore
        print(f"Found {len(pdf_files)} PDF files in {base_dir}")
        documents = []
        for pdf_path in pdf_files:
            try:
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
        vectorstore = FAISS.from_documents(docs, embedding=embeddings)
        vectorstore.save_local(db_path)
        print(f"CLARK Vector DB created at {db_path} with {vectorstore.index.ntotal} documents")
    return vectorstore

clark_db_path = "db/clark_db/clark_library_db"
clark_base_dir = "rag/clark_doc"
clark_vectorstore = load_clark_vectorstore(clark_db_path, clark_base_dir)

# RAG retrieval tool with docstring
class RagToolSchema(BaseModel):
    question: str

@tool(args_schema=RagToolSchema)
def clark_retriever_tool(question: str) -> str:
    """
    Retrieve cybersecurity educational resources from the CLARK library.
    
    This tool provides access to modular content including lecture notes, labs, slides, quizzes, and more
    for teaching and learning cybersecurity materials from the largest compilation of high-value,
    high-impact cybersecurity curriculum.
    
    Args:
        question (str): The user's query.
    
    Returns:
        str: Formatted response with retrieved content and metadata.
    """
    retriever = clark_vectorstore.as_retriever(search_kwargs={"k": 3})
    retriever_result = retriever.invoke(question)
    if not retriever_result:
        return "No relevant cybersecurity information found in the CLARK library."
    formatted_response = []
    for doc in retriever_result:
        metadata = doc.metadata
        entry = (
            f"**Content:**\n{doc.page_content}\n\n"
            f"**Metadata:**\n"
            f"- Collection: {metadata['collection_name']}\n"
            f"- Course: {metadata['course_name']}\n"
            f"- Module: {metadata['module_name']}\n"
            f"- File: {metadata['file_name']}\n"
            f"- Path: {metadata['file_path']}"
        )
        formatted_response.append(entry)
    return "\n\n---\n\n".join(formatted_response)

# Exercise retrieval tool with docstring
class ExerciseToolSchema(BaseModel):
    topic: str

@tool(args_schema=ExerciseToolSchema)
def exercise_retriever_tool(topic: str) -> str:
    """
    Retrieve cybersecurity exercises from the CLARK library based on the given topic.
    
    This tool searches the CLARK library for content related to exercises, such as quizzes, labs, or practice questions,
    and returns them with metadata. If no exercises are found, it indicates so for further processing.
    
    Args:
        topic (str): The cybersecurity topic for which to find exercises.
    
    Returns:
        str: Formatted response with retrieved exercises and metadata, or a message if none are found.
    """
    retriever = clark_vectorstore.as_retriever(search_kwargs={"k": 5})
    retriever_result = retriever.invoke(f"exercises on {topic}")
    exercise_docs = [
        doc for doc in retriever_result
        if any(keyword in doc.page_content.lower() for keyword in ["exercise", "question", "quiz", "lab"])
        or any(keyword in doc.metadata.get("file_name", "").lower() for keyword in ["exercise", "quiz", "lab"])
    ]
    if not exercise_docs:
        return "No exercises found in the CLARK library for this topic."
    formatted_response = []
    for doc in exercise_docs:
        metadata = doc.metadata
        entry = (
            f"**Exercise Content:**\n{doc.page_content}\n\n"
            f"**Source:**\n"
            f"- Collection: {metadata['collection_name']}\n"
            f"- Course: {metadata['course_name']}\n"
            f"- Module: {metadata['module_name']}\n"
            f"- File: {metadata['file_name']}\n"
            f"- Path: {metadata['file_path']}"
        )
        formatted_response.append(entry)
    return "\n\n---\n\n".join(formatted_response)