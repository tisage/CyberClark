"""
Vector Store Knowledge Unit (KU) Creator

This script reads PDF files from the rag/ku_doc/ directory, processes them, and 
creates a FAISS vector store with the extracted content to enable semantic search
and knowledge mapping for cybersecurity course analysis.
"""

import os
import glob
from typing import List, Dict, Optional
import logging
from dotenv import load_dotenv
from pathlib import Path

# PDF processing libraries
import fitz  # PyMuPDF
import re

# OpenAI and vector store libraries
from openai import OpenAI
import numpy as np
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
if not OPENAI_API_KEY:
    raise ValueError("No OpenAI API key found. Please set the OPENAI_API_KEY environment variable.")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

class KnowledgeUnitProcessor:
    """Process knowledge unit PDFs and create vector store."""
    
    def __init__(self, 
                 input_dir: str = "rag/ku_doc/", 
                 output_dir: str = "db/cyber_ku/",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Initialize the KnowledgeUnitProcessor.
        
        Args:
            input_dir: Directory containing knowledge unit PDFs
            output_dir: Directory to save the FAISS vector store
            chunk_size: Size of text chunks for vectorization
            chunk_overlap: Overlap between text chunks
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file while preserving structure.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text from the PDF
        """
        logger.info(f"Extracting text from {pdf_path}")
        
        try:
            doc = fitz.open(pdf_path)
            text_content = []
            
            for page_num, page in enumerate(doc):
                # Extract text with preservation of layout
                text = page.get_text("text")
                
                # Add page number reference
                page_header = f"\n--- Page {page_num + 1} ---\n"
                text_content.append(page_header + text)
            
            # Join all text with spacing
            full_text = "\n".join(text_content)
            
            # Clean up text
            # Remove excessive whitespace but preserve paragraph breaks
            full_text = re.sub(r'\n{3,}', '\n\n', full_text)
            full_text = re.sub(r' {2,}', ' ', full_text)
            
            return full_text
            
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
            return ""
    
    def get_pdf_metadata(self, pdf_path: str) -> Dict:
        """
        Extract metadata from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing metadata
        """
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            
            # Add filename and path to metadata
            filename = os.path.basename(pdf_path)
            metadata["filename"] = filename
            metadata["filepath"] = pdf_path
            
            # Try to extract knowledge unit ID from filename
            ku_id_match = re.search(r'KU(\d+)', filename)
            if ku_id_match:
                metadata["ku_id"] = ku_id_match.group(0)
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata from {pdf_path}: {str(e)}")
            return {"filename": os.path.basename(pdf_path), "filepath": pdf_path}
    
    def process_knowledge_unit(self, pdf_path: str) -> List[Document]:
        """
        Process a single knowledge unit PDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of Document objects ready for vectorization
        """
        # Extract text from PDF
        text_content = self.extract_text_from_pdf(pdf_path)
        if not text_content:
            logger.warning(f"No text content extracted from {pdf_path}")
            return []
        
        # Get metadata
        metadata = self.get_pdf_metadata(pdf_path)
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(text_content)
        
        # Create Document objects
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    **metadata,
                    "chunk_id": i,
                    "source": pdf_path,
                }
            )
            documents.append(doc)
        
        logger.info(f"Created {len(documents)} document chunks from {pdf_path}")
        return documents
    
    def process_all_knowledge_units(self) -> List[Document]:
        """
        Process all knowledge unit PDFs in the input directory.
        
        Returns:
            List of all Document objects
        """
        pdf_files = glob.glob(os.path.join(self.input_dir, "*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.input_dir}")
            return []
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        all_documents = []
        for pdf_file in pdf_files:
            documents = self.process_knowledge_unit(pdf_file)
            all_documents.extend(documents)
        
        logger.info(f"Created a total of {len(all_documents)} document chunks")
        return all_documents
    
    def create_vector_store(self, documents: List[Document]) -> Optional[FAISS]:
        """
        Create a FAISS vector store from the processed documents.
        
        Args:
            documents: List of Document objects
            
        Returns:
            FAISS vector store or None if creation fails
        """
        if not documents:
            logger.error("No documents to create vector store")
            return None
        
        try:
            logger.info("Creating FAISS vector store...")
            vector_store = FAISS.from_documents(documents, self.embeddings)
            return vector_store
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            return None
    
    def save_vector_store(self, vector_store: FAISS) -> bool:
        """
        Save the FAISS vector store to disk.
        
        Args:
            vector_store: FAISS vector store to save
            
        Returns:
            True if successful, False otherwise
        """
        if not vector_store:
            logger.error("No vector store to save")
            return False
        
        try:
            save_path = Path(self.output_dir)
            logger.info(f"Saving vector store to {save_path}")
            vector_store.save_local(str(save_path))
            return True
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            return False
    
    def run(self) -> bool:
        """
        Run the full knowledge unit processing pipeline.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("Starting knowledge unit processing pipeline")
        
        # Process all knowledge units
        documents = self.process_all_knowledge_units()
        if not documents:
            logger.error("No documents were created. Aborting.")
            return False
        
        # Create vector store
        vector_store = self.create_vector_store(documents)
        if not vector_store:
            logger.error("Failed to create vector store. Aborting.")
            return False
        
        # Save vector store
        success = self.save_vector_store(vector_store)
        if not success:
            logger.error("Failed to save vector store. Aborting.")
            return False
        
        logger.info("Knowledge unit processing pipeline completed successfully")
        return True

def main():
    """Main function to run the knowledge unit processor."""
    try:
        processor = KnowledgeUnitProcessor()
        success = processor.run()
        
        if success:
            print("✅ Successfully created and saved knowledge unit vector store!")
            print(f"Vector store location: {os.path.abspath(processor.output_dir)}")
        else:
            print("❌ Failed to create knowledge unit vector store.")
    except Exception as e:
        logger.exception("Unexpected error in main function")
        print(f"❌ An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()