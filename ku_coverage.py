import glob
import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_environment():
    """Load environment variables and initialize LLM."""
    load_dotenv()
    openai_api_key = os.environ.get('OPENAI_API_KEY', '')
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)  # Lower temperature for more consistent analysis
    embeddings = OpenAIEmbeddings()
    
    return llm, embeddings

def get_course_structure(directory_path):
    """
    Analyzes the directory structure to identify courses and their modules.
    
    Args:
        directory_path (str): Path to the directory containing course materials
        
    Returns:
        dict: Dictionary mapping course names to their PDF files
    """
    course_structure = {}
    
    # Get all PDF files
    pdf_files = glob.glob(f"{directory_path}/**/*.pdf", recursive=True)
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {directory_path}")
        return course_structure
    
    # Group files by course
    for pdf_path in pdf_files:
        # Extract course name from path (assuming directory structure)
        path_parts = Path(pdf_path).parts
        
        # Find the index of 'ncaec' in the path
        try:
            course_index = path_parts.index("ncaec") + 1
        except ValueError:
            logger.warning(f"Could not find 'ncaec' in path: {pdf_path}")
            continue
            
        if course_index < len(path_parts):
            course_name = path_parts[course_index]
            if course_name not in course_structure:
                course_structure[course_name] = []
            course_structure[course_name].append(pdf_path)
    
    # Log some info about what we found
    logger.info(f"Found {len(course_structure)} courses in {directory_path}")
    for course, files in course_structure.items():
        logger.info(f"Course '{course}' has {len(files)} PDF files")
    
    return course_structure

def load_knowledge_units(ku_directory):
    """
    Load knowledge units from FAISS database.
    
    Args:
        ku_directory (str): Path to the directory containing the FAISS index
        
    Returns:
        tuple: (FAISS index, list of knowledge unit metadata)
    """
    logger.info(f"Loading knowledge units from {ku_directory}")
    
    try:
        # Load the FAISS index with allow_dangerous_deserialization=True
        # Note: Only use this with trusted data sources you control
        vectorstore = FAISS.load_local(
            ku_directory, 
            OpenAIEmbeddings(),
            allow_dangerous_deserialization=True
        )
        
        # Extract knowledge unit metadata from the docstore
        ku_metadata = []
        for doc_id, doc in vectorstore.docstore._dict.items():
            if hasattr(doc, 'metadata'):
                # Add the doc_id to metadata for easier reference
                metadata = doc.metadata.copy()
                if 'id' not in metadata:
                    metadata['id'] = doc_id
                ku_metadata.append(metadata)
        
        logger.info(f"Loaded {len(ku_metadata)} knowledge units")
        return vectorstore, ku_metadata
    except Exception as e:
        logger.error(f"Error loading knowledge units: {e}")
        raise

def extract_text_from_pdfs(pdf_files):
    """
    Extract text from a list of PDF files.
    
    Args:
        pdf_files (list): List of paths to PDF files
        
    Returns:
        list: List of document chunks
    """
    text_chunks = []
    
    # Configure text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    # Process each PDF file
    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        try:
            # Extract filename for better logging and metadata
            filename = os.path.basename(pdf_path)
            logger.debug(f"Processing PDF: {filename}")
            
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            
            if not pages:
                logger.warning(f"No content extracted from {filename}")
                continue
                
            # Add source information to metadata
            for page in pages:
                page.metadata["source"] = pdf_path
                page.metadata["filename"] = filename
                page.metadata["page"] = page.metadata.get("page", 0)
            
            # Split the document into chunks
            chunks = text_splitter.split_documents(pages)
            logger.debug(f"Extracted {len(chunks)} chunks from {filename}")
            text_chunks.extend(chunks)
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
    
    logger.info(f"Total chunks extracted from all PDFs: {len(text_chunks)}")
    return text_chunks

def identify_knowledge_units(course_chunks, vectorstore, ku_metadata, llm):
    """
    Identify knowledge units covered in the course materials.
    
    Args:
        course_chunks (list): List of document chunks from the course
        vectorstore (FAISS): FAISS vector store containing knowledge units
        ku_metadata (list): List of knowledge unit metadata
        llm (ChatOpenAI): Language model instance
        
    Returns:
        dict: Dictionary of identified knowledge units with evidence
    """
    identified_kus = {}
    
    # Create a dictionary to map KU IDs to their metadata for quick lookup
    ku_id_to_metadata = {meta.get('id', i): meta for i, meta in enumerate(ku_metadata)}
    
    # Define a prompt for analyzing text chunks
    analysis_prompt = ChatPromptTemplate.from_template("""
    You are a cybersecurity curriculum expert. Analyze the following course material and identify any cybersecurity Knowledge Units (KUs) that are covered.
    
    Course material:
    {context}
    
    If this content covers any specific cybersecurity Knowledge Units, extract them. For each KU, provide:
    1. The name of the Knowledge Unit
    2. A brief explanation of how the content covers this KU
    3. Any specific concepts, tools, or techniques mentioned that relate to this KU
    
    If no specific Knowledge Units are covered, simply state "No specific cybersecurity KUs identified in this section."
    
    Provide your analysis in a structured format.
    """)
    
    # Create a chain to analyze each chunk
    analysis_chain = LLMChain(llm=llm, prompt=analysis_prompt, output_parser=StrOutputParser())
    
    # Use batch processing to analyze chunks more efficiently
    batch_size = 5  # Process 5 chunks at a time
    
    for i in tqdm(range(0, len(course_chunks), batch_size), desc="Analyzing content batches"):
        batch = course_chunks[i:i+batch_size]
        
        # Process each chunk in the batch
        for chunk in batch:
            try:
                # Get chunk text
                chunk_text = chunk.page_content
                
                # Skip very short chunks
                if len(chunk_text.strip()) < 100:
                    continue
                
                # Use semantic search to find relevant KUs
                search_results = vectorstore.similarity_search_with_score(
                    chunk_text, 
                    k=5,  # Return top 5 matches
                    fetch_k=10  # Consider top 10 matches internally
                )
                
                # Filter results to keep only good matches (lower score is better in FAISS)
                relevant_kus = [(doc, score) for doc, score in search_results if score < 0.8]
                
                if relevant_kus:
                    # Run LLM analysis to confirm the match
                    analysis_result = analysis_chain.invoke({"context": chunk_text})
                    
                    # Process each potentially relevant KU
                    for doc, score in relevant_kus:
                        ku_id = doc.metadata.get('id')
                        
                        # Skip if we don't have an ID
                        if not ku_id:
                            continue
                        
                        # Get complete metadata
                        ku_info = ku_id_to_metadata.get(ku_id, {})
                        ku_name = ku_info.get('name', "Unknown KU")
                        ku_abbr = ku_info.get('abbreviation', "")
                        ku_desc = ku_info.get('description', "No description available")
                        
                        # Add to identified KUs if not already present
                        if ku_id not in identified_kus:
                            identified_kus[ku_id] = {
                                'name': ku_name,
                                'abbreviation': ku_abbr,
                                'description': ku_desc,
                                'confidence_score': 1.0 - score,  # Convert to confidence (0-1)
                                'evidence': []
                            }
                        
                        # Add this chunk as evidence
                        identified_kus[ku_id]['evidence'].append({
                            'text': chunk_text[:300] + "..." if len(chunk_text) > 300 else chunk_text,
                            'source': chunk.metadata.get('source', 'Unknown'),
                            'page': chunk.metadata.get('page', 0),
                            'analysis': analysis_result
                        })
                        
                        # Update confidence score (use max confidence found)
                        new_confidence = 1.0 - score
                        if new_confidence > identified_kus[ku_id]['confidence_score']:
                            identified_kus[ku_id]['confidence_score'] = new_confidence
                        
            except Exception as e:
                logger.error(f"Error analyzing chunk: {e}")
                continue
    
    logger.info(f"Identified {len(identified_kus)} knowledge units")
    return identified_kus

def generate_course_report(course_name, identified_kus, output_dir):
    """
    Generate a comprehensive report for the course.
    
    Args:
        course_name (str): Name of the course
        identified_kus (dict): Dictionary of identified knowledge units
        output_dir (str): Directory to save the output
        
    Returns:
        str: Path to the generated report file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort KUs by confidence score (descending)
    sorted_kus = sorted(
        identified_kus.items(), 
        key=lambda x: x[1]['confidence_score'], 
        reverse=True
    )
    
    # Prepare report data
    report_data = {
        'course_name': course_name,
        'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
        'knowledge_units': [ku_data for _, ku_data in sorted_kus]
    }
    
    # Create a table for human-readable output
    ku_table = []
    for ku_id, ku_data in sorted_kus:
        confidence = ku_data['confidence_score']
        confidence_str = f"{confidence:.2f}" if isinstance(confidence, float) else str(confidence)
        
        ku_table.append({
            'Knowledge Unit': ku_data['name'],
            'Abbreviation': ku_data['abbreviation'],
            'Confidence': confidence_str,
            'Description': ku_data['description'][:100] + '...' if len(ku_data['description']) > 100 else ku_data['description'],
            'Evidence Count': len(ku_data['evidence'])
        })
    
    # Create a DataFrame for easier CSV export
    if ku_table:
        df = pd.DataFrame(ku_table)
        
        # Save reports in different formats
        json_path = os.path.join(output_dir, f"{course_name}_ku_coverage.json")
        csv_path = os.path.join(output_dir, f"{course_name}_ku_coverage.csv")
        txt_path = os.path.join(output_dir, f"{course_name}_ku_coverage.txt")
        
        # Save JSON (full detailed report)
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=4)
        
        # Save CSV (summary table)
        df.to_csv(csv_path, index=False)
        
        # Generate text report with more details
        with open(txt_path, 'w') as f:
            f.write(f"Knowledge Unit Coverage Report for: {course_name}\n")
            f.write(f"Analysis Date: {report_data['analysis_date']}\n")
            f.write(f"Total Knowledge Units Identified: {len(sorted_kus)}\n\n")
            
            f.write("KNOWLEDGE UNITS COVERAGE SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            for i, (ku_id, ku_data) in enumerate(sorted_kus, 1):
                f.write(f"{i}. {ku_data['name']}")
                if ku_data['abbreviation']:
                    f.write(f" ({ku_data['abbreviation']})")
                f.write(f" - Confidence: {ku_data['confidence_score']:.2f}\n")
                
                f.write(f"   Description: {ku_data['description']}\n\n")
                
                # Show top evidence (limit to 3 for readability)
                f.write("   Evidence (top examples):\n")
                for j, evidence in enumerate(ku_data['evidence'][:3], 1):
                    source = evidence['source'].split('/')[-1]  # Just the filename
                    f.write(f"   {j}. Source: {source}, Page: {evidence['page']}\n")
                    f.write(f"      Preview: {evidence['text'][:150]}...\n\n")
                
                f.write("-" * 80 + "\n\n")
        
        return json_path
    else:
        # No KUs identified, create a simple report
        no_ku_path = os.path.join(output_dir, f"{course_name}_no_ku_found.txt")
        with open(no_ku_path, 'w') as f:
            f.write(f"No Knowledge Units identified for course: {course_name}\n")
        
        return no_ku_path

def main():
    """Main function to run the knowledge unit coverage analysis."""
    try:
        # Load environment and initialize LLM
        llm, embeddings = load_environment()
        
        # Define directories
        course_dir = "rag/clark_doc/ncaec/"
        ku_dir = "db/cyber_ku/"
        output_dir = "output/"
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Load knowledge units
        vectorstore, ku_metadata = load_knowledge_units(ku_dir)
        
        # Get course structure
        course_structure = get_course_structure(course_dir)
        
        if not course_structure:
            logger.error("No courses found in the specified directory")
            return
        
        # Process each course
        for course_name, pdf_files in course_structure.items():
            logger.info(f"Processing course: {course_name} ({len(pdf_files)} PDF files)")
            
            # Extract text from PDFs
            course_chunks = extract_text_from_pdfs(pdf_files)
            
            if not course_chunks:
                logger.warning(f"No text chunks extracted for course: {course_name}")
                continue
                
            logger.info(f"Extracted {len(course_chunks)} text chunks from PDFs")
            
            # Identify knowledge units
            identified_kus = identify_knowledge_units(course_chunks, vectorstore, ku_metadata, llm)
            logger.info(f"Identified {len(identified_kus)} knowledge units for {course_name}")
            
            # Generate report
            report_path = generate_course_report(course_name, identified_kus, output_dir)
            logger.info(f"Generated report for {course_name}: {report_path}")
        
        logger.info("Analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main function: {e}", exc_info=True)

if __name__ == "__main__":
    main()