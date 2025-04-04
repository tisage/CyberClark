import datetime
import glob
import logging
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

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

# Create log directory if it doesn't exist
log_dir = 'log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Set up logging with a timestamped filename
log_filename = f"pdf_errors_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    filename=os.path.join(log_dir, log_filename),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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
        logging.info(f"CLARK Vector DB loaded at {db_path}")
        print(f"CLARK Vector DB loaded at {db_path}")
    else:
        pdf_files = get_all_pdfs(base_dir)
        if not pdf_files:
            logging.warning(f"No PDF files found in {base_dir} or its subdirectories")
            print(f"Warning: No PDF files found in {base_dir} or its subdirectories")
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(
                [Document(page_content="Empty document")], embedding=embeddings
            )
            vectorstore.save_local(db_path)
            return vectorstore
        
        logging.info(f"Found {len(pdf_files)} PDF files in {base_dir}")
        print(f"Found {len(pdf_files)} PDF files in {base_dir}")
        
        documents = []
        success_count = 0
        error_count = 0
        
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
                    
                # Detect if this is likely an exercise document based on filename
                is_exercise_file = any(keyword in file_name.lower() 
                                    for keyword in ["exercise", "quiz", "lab", "assessment", "practice"])
                
                loader = PyPDFLoader(pdf_path)
                pages = loader.load()
                
                for page in pages:
                    page.metadata.update({
                        "collection_name": collection_name,
                        "course_name": course_name,
                        "module_name": module_name,
                        "file_path": pdf_path,
                        "file_name": file_name,
                        "is_exercise_file": is_exercise_file
                    })
                    
                    # Check if page content appears to be an exercise
                    exercise_keywords = ["exercise", "question", "quiz", "lab", "assessment", 
                                        "practice", "problem", "scenario", "challenge"]
                    is_exercise_content = any(keyword in page.page_content.lower() 
                                            for keyword in exercise_keywords)
                    
                    page.metadata["is_exercise_content"] = is_exercise_content
                    
                documents.extend(pages)
                logging.info(f"Successfully loaded {pdf_path}")
                print(f"Loaded {pdf_path} with metadata")
                success_count += 1
            except Exception as e:
                logging.error(f"Error loading {pdf_path}: {e}", exc_info=True)
                print(f"Error loading {pdf_path}: {e}")
                error_count += 1
        
        # Log and print summary
        summary_msg = f"Finished loading PDFs. Successfully loaded {success_count} PDFs, {error_count} PDFs had errors."
        logging.info(summary_msg)
        print(summary_msg)
        
        if error_count > 0:
            logging.error(f"Encountered errors in {error_count} PDFs. Check the log for details.")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(docs, embedding=embeddings)
        vectorstore.save_local(db_path)
        logging.info(f"CLARK Vector DB created at {db_path} with {vectorstore.index.ntotal} documents")
        print(f"CLARK Vector DB created at {db_path} with {vectorstore.index.ntotal} documents")
    
    return vectorstore

# Main execution
clark_db_path = "db/clark_db/clark_library_db"
clark_base_dir = "rag/clark_doc"
clark_vectorstore = load_clark_vectorstore(clark_db_path, clark_base_dir)

# RAG retrieval tool
class RagToolSchema(BaseModel):
    question: str

@tool(args_schema=RagToolSchema)
def clark_retriever_tool(question: str) -> str:
    """
    Retrieve cybersecurity educational resources from the CLARK library.
    
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

# Exercise retrieval tool
class ExerciseToolSchema(BaseModel):
    topic: str

@tool(args_schema=ExerciseToolSchema)
def exercise_retriever_tool(topic: str) -> str:
    """
    Retrieve cybersecurity exercises from the CLARK library based on the given topic.
    
    Args:
        topic (str): The cybersecurity topic for which to find exercises.
    
    Returns:
        str: Formatted response with retrieved exercises and metadata, or a message if none are found.
    """
    retriever = clark_vectorstore.as_retriever(search_kwargs={"k": 7})
    
    search_queries = [
        f"quiz on {topic}",
        f"exercises for {topic}",
        f"practice questions about {topic}",
        f"lab assignment {topic}",
        f"assessment {topic}",
        f"{topic} exercise",
    ]
    
    all_results = []
    for query in search_queries:
        results = retriever.invoke(query)
        all_results.extend(results)
    
    exercise_keywords = ["exercise", "question", "quiz", "lab", "assessment", "practice", 
                         "problem", "scenario", "challenge", "test your knowledge"]
    
    exercise_docs = []
    seen_content = set()
    
    for doc in all_results:
        content_hash = hash(doc.page_content[:100])
        if content_hash in seen_content:
            continue
        seen_content.add(content_hash)
        
        is_exercise = any(keyword in doc.page_content.lower() for keyword in exercise_keywords)
        has_exercise_filename = any(keyword in doc.metadata.get("file_name", "").lower() 
                                   for keyword in exercise_keywords)
        
        if is_exercise or has_exercise_filename:
            score = sum(1 for keyword in exercise_keywords 
                        if keyword in doc.page_content.lower())
            doc.metadata["relevance_score"] = score
            exercise_docs.append(doc)
    
    exercise_docs.sort(key=lambda x: x.metadata.get("relevance_score", 0), reverse=True)
    exercise_docs = exercise_docs[:3]
    
    if not exercise_docs:
        return "No existing exercises found in the CLARK library for this topic."
    
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