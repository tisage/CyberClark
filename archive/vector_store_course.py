"""
Vector Store Course Material Creator with LLM-based Analysis

This script reads PDF files from the course materials directory, processes them using
GPT-4o-mini for content analysis, and creates a FAISS vector store with the extracted 
content to enable semantic search and knowledge mapping for cybersecurity course analysis.
"""

import os
import glob
import json
import time
from typing import List, Dict, Optional, Set, Tuple, Any
import logging
from dotenv import load_dotenv
from pathlib import Path
import re

# PDF processing libraries
import fitz  # PyMuPDF

# OpenAI and vector store libraries
from openai import OpenAI
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

class CourseContentProcessor:
    """Process course material PDFs and create vector store."""
    
    def __init__(self, 
                 input_dir: str = "rag/clark_doc/Computer and NW Security - Undergrad/", 
                 output_dir: str = "db/courses/",
                 chunk_size: int = 1500,
                 chunk_overlap: int = 300,
                 llm_model: str = "gpt-4o-mini"):
        """
        Initialize the CourseContentProcessor.
        
        Args:
            input_dir: Directory containing course material PDFs
            output_dir: Directory to save the FAISS vector store
            chunk_size: Size of text chunks for vectorization
            chunk_overlap: Overlap between text chunks
            llm_model: OpenAI model to use for analysis
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.llm_model = llm_model
        self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Track processed files for reporting
        self.processed_files = []
        self.failed_files = []
        self.total_chunks = 0
        
        # Initialize a set to store detected topics and concepts
        self.detected_topics = set()
        
        # Initialize analysis cache to avoid duplicate LLM calls
        self.analysis_cache_dir = os.path.join(self.output_dir, "analysis_cache")
        os.makedirs(self.analysis_cache_dir, exist_ok=True)
        self.analysis_cache = {}
        self.load_analysis_cache()
    
    def load_analysis_cache(self):
        """Load existing analysis cache if available."""
        cache_file = os.path.join(self.analysis_cache_dir, "analysis_cache.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    self.analysis_cache = json.load(f)
                logger.info(f"Loaded analysis cache with {len(self.analysis_cache)} entries")
            except Exception as e:
                logger.error(f"Error loading analysis cache: {str(e)}")
                self.analysis_cache = {}
    
    def save_analysis_cache(self):
        """Save analysis cache to disk."""
        cache_file = os.path.join(self.analysis_cache_dir, "analysis_cache.json")
        try:
            with open(cache_file, 'w') as f:
                json.dump(self.analysis_cache, f)
            logger.info(f"Saved analysis cache with {len(self.analysis_cache)} entries")
        except Exception as e:
            logger.error(f"Error saving analysis cache: {str(e)}")
    
    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, Dict]:
        """
        Extract text and structural information from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Tuple of (extracted text, structure metadata)
        """
        logger.info(f"Extracting text from {pdf_path}")
        
        try:
            doc = fitz.open(pdf_path)
            text_content = []
            toc = doc.get_toc()
            
            # Extract document structure metadata
            structure_metadata = {
                "total_pages": len(doc),
                "toc": toc if toc else [],
                "has_toc": bool(toc),
                "pdf_info": doc.metadata if doc.metadata else {}
            }
            
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
            
            return full_text, structure_metadata
            
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
            return "", {}
    
    def analyze_content_with_llm(self, text_sample: str, filename: str) -> Dict:
        """
        Analyze the content using GPT-4o-mini to identify content type and topics.
        
        Args:
            text_sample: Sample text from the document to analyze
            filename: Filename of the document (for caching)
            
        Returns:
            Dictionary with analysis results
        """
        # Create a cache key based on the filename and first 100 chars of text
        # (to handle cases where the filename is reused with different content)
        cache_key = f"{filename}_{hash(text_sample[:100])}"
        
        # Check if we have this analysis cached
        if cache_key in self.analysis_cache:
            logger.info(f"Using cached analysis for {filename}")
            return self.analysis_cache[cache_key]
        
        logger.info(f"Analyzing content with LLM for {filename}")
        
        try:
            # Prepare a prompt for the LLM
            prompt = f"""
            Analyze the following text from a cybersecurity course material document.
            
            Filename: {filename}
            
            Text sample from the document:
            ```
            {text_sample[:3000]}
            ```
            
            Please provide a detailed analysis in JSON format with the following structure:
            
            1. content_type: Determine the primary type of this document (syllabus, lecture, assignment, lab, assessment, readings, other)
            2. content_properties: Identify characteristics like "contains_code", "has_diagrams", "has_exercises", etc.
            3. cybersecurity_topics: List all cybersecurity topics covered in this document
            4. learning_objectives: Extract or infer learning objectives from this document
            5. document_title: Extract a meaningful title from the document
            6. technologies_mentioned: List any specific technologies, tools, or software mentioned
            7. confidence_score: Your confidence in this analysis (0-1)
            
            Return ONLY valid JSON and absolutely nothing else. Format your response as:
            {{
              "content_type": "string",
              "content_properties": {{
                "contains_code": boolean,
                "has_exercises": boolean,
                ...
              }},
              "cybersecurity_topics": ["topic1", "topic2", ...],
              "learning_objectives": ["objective1", "objective2", ...],
              "document_title": "string",
              "technologies_mentioned": ["tech1", "tech2", ...],
              "confidence_score": float
            }}
            """
            
            # Call the OpenAI API
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            # Extract and parse the JSON response
            try:
                analysis_result = json.loads(response.choices[0].message.content)
                
                # Add to cache
                self.analysis_cache[cache_key] = analysis_result
                
                # Add topics to the global set
                if "cybersecurity_topics" in analysis_result:
                    self.detected_topics.update(analysis_result["cybersecurity_topics"])
                
                return analysis_result
                
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing LLM response as JSON: {str(e)}")
                logger.error(f"Response content: {response.choices[0].message.content}")
                return self._get_default_analysis()
                
        except Exception as e:
            logger.error(f"Error analyzing content with LLM: {str(e)}")
            return self._get_default_analysis()
    
    def _get_default_analysis(self) -> Dict:
        """Get a default analysis result when LLM analysis fails."""
        return {
            "content_type": "unknown",
            "content_properties": {
                "contains_code": False,
                "has_exercises": False
            },
            "cybersecurity_topics": [],
            "learning_objectives": [],
            "document_title": "Untitled Document",
            "technologies_mentioned": [],
            "confidence_score": 0.0
        }
    
    def get_course_metadata(self, pdf_path: str, extracted_text: str, structure_metadata: Dict) -> Dict:
        """
        Gather comprehensive metadata about a course document.
        
        Args:
            pdf_path: Path to the PDF file
            extracted_text: Extracted text content
            structure_metadata: Structure information from the PDF
            
        Returns:
            Dictionary containing metadata
        """
        # Get basic file information
        filename = os.path.basename(pdf_path)
        relative_path = os.path.relpath(pdf_path, self.input_dir)
        file_size = os.path.getsize(pdf_path)
        
        # Sample the text for LLM analysis (first part, middle part, and last part)
        text_length = len(extracted_text)
        sample_size = min(1000, text_length // 3)
        
        start_sample = extracted_text[:sample_size]
        middle_start = max(0, text_length // 2 - sample_size // 2)
        middle_sample = extracted_text[middle_start:middle_start + sample_size]
        end_sample = extracted_text[max(0, text_length - sample_size):]
        
        text_sample = start_sample + "\n\n[...]\n\n" + middle_sample + "\n\n[...]\n\n" + end_sample
        
        # Analyze content with LLM
        llm_analysis = self.analyze_content_with_llm(text_sample, filename)
        
        # Get document title from various sources (prioritizing LLM-extracted title)
        title = llm_analysis.get("document_title", None)
        if not title or title == "Untitled Document":
            title = structure_metadata.get("pdf_info", {}).get("title", None)
        if not title or title.strip() == "":
            title = filename
        
        metadata = {
            "filename": filename,
            "filepath": relative_path,
            "full_path": pdf_path,
            "title": title,
            "file_size_bytes": file_size,
            "content_type": llm_analysis.get("content_type", "unknown"),
            "content_properties": llm_analysis.get("content_properties", {}),
            "cybersecurity_topics": llm_analysis.get("cybersecurity_topics", []),
            "learning_objectives": llm_analysis.get("learning_objectives", []),
            "technologies_mentioned": llm_analysis.get("technologies_mentioned", []),
            "analysis_confidence": llm_analysis.get("confidence_score", 0.0),
            "structure": structure_metadata
        }
        
        return metadata
    
    def process_course_document(self, pdf_path: str) -> List[Document]:
        """
        Process a single course document PDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of Document objects ready for vectorization
        """
        try:
            # Extract text and structure from PDF
            text_content, structure_metadata = self.extract_text_from_pdf(pdf_path)
            if not text_content:
                logger.warning(f"No text content extracted from {pdf_path}")
                self.failed_files.append(pdf_path)
                return []
            
            # Get comprehensive metadata
            metadata = self.get_course_metadata(pdf_path, text_content, structure_metadata)
            
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
                        "total_chunks": len(chunks),
                    }
                )
                documents.append(doc)
            
            logger.info(f"Created {len(documents)} document chunks from {pdf_path}")
            self.total_chunks += len(documents)
            self.processed_files.append(pdf_path)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            self.failed_files.append(pdf_path)
            return []
    
    def process_all_course_documents(self) -> List[Document]:
        """
        Process all course document PDFs in the input directory.
        
        Returns:
            List of all Document objects
        """
        pdf_files = glob.glob(os.path.join(self.input_dir, "**/*.pdf"), recursive=True)
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.input_dir}")
            return []
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        all_documents = []
        for i, pdf_file in enumerate(pdf_files):
            logger.info(f"Processing file {i+1}/{len(pdf_files)}: {pdf_file}")
            documents = self.process_course_document(pdf_file)
            all_documents.extend(documents)
            
            # Save the analysis cache every 5 files to prevent data loss
            if (i + 1) % 5 == 0:
                self.save_analysis_cache()
                
            # Add a small delay to prevent API rate limiting
            time.sleep(0.5)
        
        # Save the final analysis cache
        self.save_analysis_cache()
        
        logger.info(f"Created a total of {len(all_documents)} document chunks from {len(self.processed_files)} files")
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
    
    def save_processing_report(self) -> None:
        """Save a report of the processing results."""
        report = {
            "input_directory": self.input_dir,
            "output_directory": self.output_dir,
            "processed_files_count": len(self.processed_files),
            "failed_files_count": len(self.failed_files),
            "total_chunks": self.total_chunks,
            "detected_topics": list(self.detected_topics),
            "processed_files": [os.path.relpath(f, self.input_dir) for f in self.processed_files],
            "failed_files": [os.path.relpath(f, self.input_dir) for f in self.failed_files],
            "llm_model_used": self.llm_model
        }
        
        report_path = os.path.join(self.output_dir, "processing_report.json")
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Processing report saved to {report_path}")
        except Exception as e:
            logger.error(f"Error saving processing report: {str(e)}")
    
    def run(self) -> bool:
        """
        Run the full course content processing pipeline.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("Starting course content processing pipeline")
        
        # Process all course documents
        documents = self.process_all_course_documents()
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
        
        # Save processing report
        self.save_processing_report()
        
        logger.info("Course content processing pipeline completed successfully")
        return True

def main():
    """Main function to run the course content processor."""
    try:
        processor = CourseContentProcessor()
        success = processor.run()
        
        if success:
            print("✅ Successfully created and saved course content vector store!")
            print(f"Vector store location: {os.path.abspath(processor.output_dir)}")
            print(f"Processed {len(processor.processed_files)} files with {processor.total_chunks} total chunks")
            print(f"Detected {len(processor.detected_topics)} cybersecurity topics")
        else:
            print("❌ Failed to create course content vector store.")
    except Exception as e:
        logger.exception("Unexpected error in main function")
        print(f"❌ An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()