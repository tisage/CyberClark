import datetime
import glob
import logging
import os
from typing import List, Dict, Any

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

# Optimized chunk settings for better retrieval
CHUNK_SIZE = 1000  # Smaller chunks for more precise retrieval
CHUNK_OVERLAP = 150  # Maintain context between chunks

# Create cached embeddings to avoid recomputing
def get_embeddings():
    # Use dimensions=1536 for ada-002 model (OpenAI's default)
    # Use dimensions=768 for smaller models if you switch
    return OpenAIEmbeddings(
        model="text-embedding-ada-002",  # Explicitly set model
        chunk_size=1000,  # Process 1000 texts at once when embedding
        embedding_ctx_length=8191,  # Maximum context length
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
    # Create directory structure if it doesn't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    if os.path.exists(db_path):
        embeddings = get_embeddings()
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
            embeddings = get_embeddings()
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
        
        # Process files in smaller batches to manage memory
        batch_size = 10
        for i in range(0, len(pdf_files), batch_size):
            batch = pdf_files[i:i+batch_size]
            batch_docs = []
            
            for pdf_path in batch:
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
                            "file_name": file_name,
                            "is_exercise_file": is_exercise_file,
                            # Add keywords field for better filtering during retrieval
                            "keywords": extract_keywords(page.page_content),
                            # Add more semantic search-friendly fields
                            "document_type": "exercise" if is_exercise_file else "content",
                        })
                        
                        # Check if page content appears to be an exercise
                        exercise_keywords = ["exercise", "question", "quiz", "lab", "assessment", 
                                            "practice", "problem", "scenario", "challenge"]
                        is_exercise_content = any(keyword in page.page_content.lower() 
                                                for keyword in exercise_keywords)
                        
                        page.metadata["is_exercise_content"] = is_exercise_content
                        
                    batch_docs.extend(pages)
                    logging.info(f"Successfully loaded {pdf_path}")
                    print(f"Loaded {pdf_path} with metadata")
                    success_count += 1
                except Exception as e:
                    logging.error(f"Error loading {pdf_path}: {e}", exc_info=True)
                    print(f"Error loading {pdf_path}: {e}")
                    error_count += 1
            
            # Process this batch
            documents.extend(batch_docs)
        
        # Log and print summary
        summary_msg = f"Finished loading PDFs. Successfully loaded {success_count} PDFs, {error_count} PDFs had errors."
        logging.info(summary_msg)
        print(summary_msg)
        
        if error_count > 0:
            logging.error(f"Encountered errors in {error_count} PDFs. Check the log for details.")
        
        # Optimize text splitting for better retrieval performance
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        docs = text_splitter.split_documents(documents)
        
        # Add chunk index metadata to help with context retrieval
        for i, doc in enumerate(docs):
            doc.metadata["chunk_id"] = i
        
        embeddings = get_embeddings()
        
        # Configure FAISS with optimized parameters for cybersecurity content
        vectorstore = FAISS.from_documents(
            docs, 
            embedding=embeddings,
            # Use 'l2' for more precise similarity (Euclidean distance)
            # Use 'cosine' for better handling of different length documents
            distance_strategy="cosine"  
        )
        
        vectorstore.save_local(db_path)
        logging.info(f"CLARK Vector DB created at {db_path} with {vectorstore.index.ntotal} documents")
        print(f"CLARK Vector DB created at {db_path} with {vectorstore.index.ntotal} documents")
    
    return vectorstore

def extract_keywords(text: str) -> List[str]:
    """Extract key cybersecurity terms from text for better indexing."""
    # A simple keyword extraction approach - in production you might use NLP
    cybersecurity_terms = [
        "vulnerability", "exploit", "malware", "firewall", "encryption", 
        "authentication", "authorization", "risk", "threat", "attack", 
        "network", "security", "data", "protection", "privacy", "compliance",
        "penetration", "testing", "incident", "response", "phishing", "ransomware",
        "cryptography", "hash", "cipher", "protocol", "certificate", "access control"
    ]
    
    text_lower = text.lower()
    found_keywords = [term for term in cybersecurity_terms if term in text_lower]
    return found_keywords

# Main execution with default database paths
clark_db_path = "db/clark_db/clark_library_db"
clark_base_dir = "rag/clark_doc"
clark_vectorstore = load_clark_vectorstore(clark_db_path, clark_base_dir)

# Precompute common query embeddings to speed up future queries
def precompute_common_queries():
    """Precompute embeddings for common cybersecurity terms to improve query speed."""
    common_queries = [
        "network security", "cybersecurity basics", "encryption", "authentication", 
        "access control", "risk management", "penetration testing", "malware analysis",
        "incident response", "security policies", "vulnerability assessment"
    ]
    
    embeddings = get_embeddings()
    # Store these embeddings for faster lookup (not using them yet, but framework is in place)
    query_embeddings = {query: embeddings.embed_query(query) for query in common_queries}
    return query_embeddings

# Try to precompute common query embeddings if possible
try:
    precomputed_embeddings = precompute_common_queries()
except Exception as e:
    logging.warning(f"Could not precompute embeddings: {e}")

# Optimized RAG retrieval tool with better search parameters
class RagToolSchema(BaseModel):
    question: str

@tool(args_schema=RagToolSchema)
def clark_retriever_tool(question: str) -> str:
    """
    Retrieve cybersecurity educational resources from the CLARK library with optimized search.
    
    Args:
        question (str): The user's query.
    
    Returns:
        str: Formatted response with retrieved content and metadata.
    """
    # Increase k slightly to get more candidates
    raw_retriever = clark_vectorstore.as_retriever(
        search_type="similarity",  # Use similarity search for better precision
        search_kwargs={
            "k": 5,  # Retrieve more candidates
            "score_threshold": 0.6  # Only return reasonably relevant results
        }
    )
    
    # Add query expansion for better recall
    expanded_query = generate_expanded_query(question)
    retriever_result = raw_retriever.invoke(expanded_query)
    
    if not retriever_result:
        return "No relevant cybersecurity information found in the CLARK library."
    
    # Sort by relevance and take top 3
    sorted_results = sorted(
        retriever_result, 
        key=lambda x: similarity_score(question, x.page_content), 
        reverse=True
    )[:3]
    
    formatted_response = []
    for doc in sorted_results:
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

def generate_expanded_query(question: str) -> str:
    """Generate an expanded query to improve recall."""
    # Simple expansion - in production you might use a model for this
    # For now, just add common cybersecurity context words
    cybersecurity_prefixes = ["cybersecurity", "information security", "infosec"]
    
    # Don't expand if question is already long enough
    if len(question.split()) > 8:
        return question
        
    for prefix in cybersecurity_prefixes:
        if prefix in question.lower():
            return question
            
    return f"cybersecurity {question}"

def similarity_score(query: str, text: str) -> float:
    """Calculate a simple text similarity score - could be replaced with more sophisticated scoring."""
    query_terms = set(query.lower().split())
    text_terms = set(text.lower().split())
    
    # Jaccard similarity as a simple metric
    if not query_terms or not text_terms:
        return 0.0
        
    intersection = len(query_terms.intersection(text_terms))
    union = len(query_terms.union(text_terms))
    
    return intersection / union

# Optimized Exercise retrieval tool with better filtering
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
    # Create a more focused search for exercises
    retriever = clark_vectorstore.as_retriever(
        search_type="mmr",  # Maximum Marginal Relevance - better diversity in results
        search_kwargs={
            "k": 10,  # Get more candidate documents
            "fetch_k": 15,  # Consider an even larger initial set
            "lambda_mult": 0.7,  # Balance between relevance and diversity
        }
    )
    
    # Create a more targeted set of search queries
    search_queries = [
        f"quiz on {topic}",
        f"exercises for {topic}",
        f"practice questions about {topic}",
        f"lab assignment {topic}",
        f"assessment {topic}",
        f"{topic} exercise",
    ]
    
    all_results = []
    for query in search_queries[:3]:  # Limit to top 3 queries for speed
        results = retriever.invoke(query)
        all_results.extend(results)
    
    exercise_keywords = ["exercise", "question", "quiz", "lab", "assessment", "practice", 
                         "problem", "scenario", "challenge", "test your knowledge"]
    
    exercise_docs = []
    seen_content = set()
    
    # More efficient deduplication and scoring
    for doc in all_results:
        # Create a more robust content fingerprint
        content_hash = hash(doc.page_content[:200])
        if content_hash in seen_content:
            continue
        seen_content.add(content_hash)
        
        # Only process documents that are likely exercises
        is_exercise = (
            doc.metadata.get("is_exercise_file", False) or 
            doc.metadata.get("is_exercise_content", False) or
            any(keyword in doc.page_content.lower() for keyword in exercise_keywords[:5])
        )
        
        if is_exercise:
            # Calculate a more nuanced relevance score
            keyword_score = sum(1 for keyword in exercise_keywords 
                             if keyword in doc.page_content.lower())
            topic_score = 3 if topic.lower() in doc.page_content.lower() else 0
            
            doc.metadata["relevance_score"] = keyword_score + topic_score
            exercise_docs.append(doc)
    
    # Sort by our custom relevance score
    exercise_docs.sort(key=lambda x: x.metadata.get("relevance_score", 0), reverse=True)
    exercise_docs = exercise_docs[:3]  # Take top 3
    
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