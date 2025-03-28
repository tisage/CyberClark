"""
Knowledge Mapping Engine

This module maps course content to knowledge units using vector embeddings and semantic similarity.
It identifies gaps between KU requirements and course materials and quantifies coverage levels.
Results are saved to the /output/ folder for later visualization and reporting.
"""

import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Vector stores and embeddings
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

# OpenAI LLM
from openai import OpenAI

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
if not OPENAI_API_KEY:
    raise ValueError("No OpenAI API key found. Please set the OPENAI_API_KEY environment variable.")

class KnowledgeMappingEngine:
    """Maps course content to knowledge units and analyzes coverage."""
    
    def __init__(self, 
                 ku_vectorstore_path: str = "db/cyber_ku/",
                 course_vectorstore_path: str = "db/courses/",
                 output_dir: str = "output/",
                 llm_model: str = "gpt-4o-mini",
                 similarity_threshold: float = 0.7):
        """
        Initialize the Knowledge Mapping Engine.
        
        Args:
            ku_vectorstore_path: Path to knowledge unit vector store
            course_vectorstore_path: Path to course content vector store
            output_dir: Directory to save mapping results
            llm_model: OpenAI model to use for analysis
            similarity_threshold: Minimum similarity score for considering a match
        """
        self.ku_vectorstore_path = ku_vectorstore_path
        self.course_vectorstore_path = course_vectorstore_path
        self.output_dir = output_dir
        self.llm_model = llm_model
        self.similarity_threshold = similarity_threshold
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load vector stores
        self.ku_vectorstore = None
        self.course_vectorstore = None
        
        # Store mapping results
        self.mapping_results = {}
        
        # Store KU metadata
        self.ku_metadata = {}
        
        # Store course metadata
        self.course_metadata = {}
    
    def load_vector_stores(self) -> bool:
        """
        Load the knowledge unit and course content vector stores.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Loading knowledge unit vector store from {self.ku_vectorstore_path}")
            self.ku_vectorstore = FAISS.load_local(
                self.ku_vectorstore_path, 
                self.embeddings,
                allow_dangerous_deserialization=True  # Required for newer LangChain versions
            )
            
            logger.info(f"Loading course content vector store from {self.course_vectorstore_path}")
            self.course_vectorstore = FAISS.load_local(
                self.course_vectorstore_path, 
                self.embeddings,
                allow_dangerous_deserialization=True  # Required for newer LangChain versions
            )
            
            return True
        except Exception as e:
            logger.error(f"Error loading vector stores: {str(e)}")
            return False
    
    def extract_ku_metadata(self) -> Dict[str, Dict]:
        """
        Extract metadata about knowledge units from the vector store.
        
        Returns:
            Dictionary mapping KU IDs to metadata
        """
        logger.info("Extracting knowledge unit metadata")
        
        ku_metadata = {}
        try:
            # Get all documents from the KU vector store
            ku_docs = self.ku_vectorstore.docstore._dict.values()
            
            # Group documents by KU ID
            for doc in ku_docs:
                ku_id = doc.metadata.get("ku_id")
                if not ku_id:
                    continue
                
                if ku_id not in ku_metadata:
                    ku_metadata[ku_id] = {
                        "title": doc.metadata.get("title", "Unknown"),
                        "filepath": doc.metadata.get("filepath", ""),
                        "filename": doc.metadata.get("filename", ""),
                        "chunks": [],
                        "total_chunks": 0
                    }
                
                # Add this chunk
                ku_metadata[ku_id]["chunks"].append({
                    "chunk_id": doc.metadata.get("chunk_id", -1),
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                })
                ku_metadata[ku_id]["total_chunks"] += 1
            
            logger.info(f"Extracted metadata for {len(ku_metadata)} knowledge units")
            
            # Save extracted metadata
            self.ku_metadata = ku_metadata
            
            return ku_metadata
            
        except Exception as e:
            logger.error(f"Error extracting KU metadata: {str(e)}")
            return {}
    
    def extract_course_metadata(self) -> Dict[str, Dict]:
        """
        Extract metadata about course content from the vector store.
        
        Returns:
            Dictionary mapping course document filenames to metadata
        """
        logger.info("Extracting course metadata")
        
        course_metadata = {}
        try:
            # Get all documents from the course vector store
            course_docs = self.course_vectorstore.docstore._dict.values()
            
            # Group documents by filename
            for doc in course_docs:
                filename = doc.metadata.get("filename")
                if not filename:
                    continue
                
                if filename not in course_metadata:
                    course_metadata[filename] = {
                        "title": doc.metadata.get("title", "Unknown"),
                        "filepath": doc.metadata.get("filepath", ""),
                        "content_type": doc.metadata.get("content_type", "unknown"),
                        "cybersecurity_topics": doc.metadata.get("cybersecurity_topics", []),
                        "learning_objectives": doc.metadata.get("learning_objectives", []),
                        "technologies_mentioned": doc.metadata.get("technologies_mentioned", []),
                        "chunks": [],
                        "total_chunks": 0
                    }
                
                # Add this chunk
                course_metadata[filename]["chunks"].append({
                    "chunk_id": doc.metadata.get("chunk_id", -1),
                    "page_content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                    "metadata": doc.metadata
                })
                course_metadata[filename]["total_chunks"] += 1
            
            logger.info(f"Extracted metadata for {len(course_metadata)} course documents")
            
            # Save extracted metadata
            self.course_metadata = course_metadata
            
            return course_metadata
            
        except Exception as e:
            logger.error(f"Error extracting course metadata: {str(e)}")
            return {}
    
    def map_ku_to_course_content(self, ku_id: str, ku_data: Dict) -> Dict:
        """
        Map a knowledge unit to relevant course content using semantic search.
        
        Args:
            ku_id: Knowledge unit ID
            ku_data: Knowledge unit metadata
            
        Returns:
            Mapping results for this knowledge unit
        """
        logger.info(f"Mapping knowledge unit {ku_id} to course content")
        
        results = {
            "ku_id": ku_id,
            "ku_title": ku_data.get("title", "Unknown"),
            "mapping_timestamp": datetime.now().isoformat(),
            "course_matches": [],
            "coverage_score": 0.0,
            "gap_analysis": {},
            "relevant_topics": []
        }
        
        try:
            # Combine all chunks for this KU into a single query
            all_ku_content = "\n\n".join([chunk["page_content"] for chunk in ku_data["chunks"]])
            
            # Perform similarity search against course content
            matches = self.course_vectorstore.similarity_search_with_score(
                all_ku_content,
                k=20  # Get top 20 matches
            )
            
            # Process matches
            if not matches:
                logger.warning(f"No course content matches found for {ku_id}")
                results["gap_analysis"] = self.generate_gap_analysis(ku_id, all_ku_content, [])
                return results
            
            # Group matches by document
            document_matches = {}
            for doc, score in matches:
                similarity_score = float(1.0 - score)  # Convert distance to similarity
                
                # Only consider matches above threshold
                if similarity_score < self.similarity_threshold:
                    continue
                
                filename = doc.metadata.get("filename")
                if not filename:
                    continue
                
                if filename not in document_matches:
                    document_matches[filename] = {
                        "filename": filename,
                        "title": doc.metadata.get("title", "Unknown"),
                        "filepath": doc.metadata.get("filepath", ""),
                        "content_type": doc.metadata.get("content_type", "unknown"),
                        "best_score": similarity_score,
                        "avg_score": similarity_score,
                        "total_scores": 1,
                        "matching_chunks": [{
                            "chunk_id": doc.metadata.get("chunk_id", -1),
                            "similarity": similarity_score,
                            "snippet": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                        }]
                    }
                else:
                    # Update existing entry
                    document_matches[filename]["total_scores"] += 1
                    document_matches[filename]["avg_score"] = (
                        (document_matches[filename]["avg_score"] * (document_matches[filename]["total_scores"] - 1)) + 
                        similarity_score
                    ) / document_matches[filename]["total_scores"]
                    
                    if similarity_score > document_matches[filename]["best_score"]:
                        document_matches[filename]["best_score"] = similarity_score
                    
                    document_matches[filename]["matching_chunks"].append({
                        "chunk_id": doc.metadata.get("chunk_id", -1),
                        "similarity": similarity_score,
                        "snippet": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                    })
            
            # Convert to list and sort by best score
            sorted_matches = sorted(
                document_matches.values(), 
                key=lambda x: x["best_score"], 
                reverse=True
            )
            
            # Add to results
            results["course_matches"] = sorted_matches
            
            # Calculate overall coverage score
            if sorted_matches:
                # Average of best scores from top 3 matches (or fewer if less are available)
                top_n = min(3, len(sorted_matches))
                results["coverage_score"] = sum([m["best_score"] for m in sorted_matches[:top_n]]) / top_n
            
            # Generate gap analysis
            results["gap_analysis"] = self.generate_gap_analysis(ku_id, all_ku_content, sorted_matches)
            
            # Extract relevant topics
            all_topics = set()
            for match in sorted_matches:
                filename = match["filename"]
                if filename in self.course_metadata:
                    all_topics.update(self.course_metadata[filename].get("cybersecurity_topics", []))
            
            results["relevant_topics"] = list(all_topics)
            
            return results
            
        except Exception as e:
            logger.error(f"Error mapping KU {ku_id} to course content: {str(e)}")
            return results
    
    def generate_gap_analysis(self, ku_id: str, ku_content: str, matches: List[Dict]) -> Dict:
        """
        Generate a gap analysis using the LLM to analyze the coverage of a knowledge unit.
        
        Args:
            ku_id: Knowledge unit ID
            ku_content: Combined content of the knowledge unit
            matches: List of matching course content
            
        Returns:
            Gap analysis information
        """
        logger.info(f"Generating gap analysis for {ku_id}")
        
        try:
            # Create a summary of matches
            matches_summary = ""
            for i, match in enumerate(matches[:5]):  # Use top 5 matches for analysis
                match_summary = (
                    f"Match {i+1}: {match['title']} (Score: {match['best_score']:.2f})\n"
                    f"Document type: {match['content_type']}\n"
                    f"Sample content: {match['matching_chunks'][0]['snippet'][:200]}...\n\n"
                )
                matches_summary += match_summary
            
            if not matches_summary:
                matches_summary = "No matching course content found."
            
            # Prompt for the LLM
            prompt = f"""
            Analyze the coverage of a cybersecurity knowledge unit in course materials:
            
            KNOWLEDGE UNIT {ku_id}:
            ```
            {ku_content[:1500]}
            ```
            
            MATCHING COURSE MATERIALS:
            ```
            {matches_summary}
            ```
            
            Please provide a structured gap analysis in JSON format with the following:
            
            1. coverage_summary: A qualitative assessment of how well the course materials cover this knowledge unit
            2. coverage_level: Numeric estimate (0-100) of coverage percentage
            3. identified_gaps: List of specific topics, concepts or skills from the KU that are missing or undercovered
            4. recommendations: Suggestions for improving coverage
            
            Return ONLY valid JSON and absolutely nothing else. Format your response as:
            {{
              "coverage_summary": "string",
              "coverage_level": int,
              "identified_gaps": ["gap1", "gap2", ...],
              "recommendations": ["rec1", "rec2", ...]
            }}
            """
            
            # Call the OpenAI API
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            # Extract and parse the JSON response
            try:
                analysis_result = json.loads(response.choices[0].message.content)
                return analysis_result
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing LLM response as JSON: {str(e)}")
                logger.error(f"Response content: {response.choices[0].message.content}")
                return self._get_default_gap_analysis()
                
        except Exception as e:
            logger.error(f"Error generating gap analysis: {str(e)}")
            return self._get_default_gap_analysis()
    
    def _get_default_gap_analysis(self) -> Dict:
        """Get a default gap analysis when LLM analysis fails."""
        return {
            "coverage_summary": "Error generating analysis",
            "coverage_level": 0,
            "identified_gaps": ["Error: Unable to identify gaps"],
            "recommendations": ["Error: Unable to generate recommendations"]
        }
    
    def generate_course_ku_coverage_report(self) -> Dict:
        """
        Generate a comprehensive report of KU coverage across the entire course.
        
        Returns:
            Dictionary with course-wide coverage information
        """
        logger.info("Generating course-wide KU coverage report")
        
        # Prepare the report structure
        report = {
            "course_name": "Computer and Network Security - Undergrad",
            "analysis_timestamp": datetime.now().isoformat(),
            "ku_coverage_summary": {},
            "overall_coverage_metrics": {
                "average_coverage_score": 0.0,
                "well_covered_ku_count": 0,
                "partially_covered_ku_count": 0,
                "poorly_covered_ku_count": 0,
                "total_ku_count": len(self.mapping_results)
            },
            "course_content_summary": {
                "total_documents": len(self.course_metadata),
                "document_types": {}
            },
            "topic_coverage": {},
            "recommended_improvements": []
        }
        
        try:
            # Summarize KU coverage
            total_coverage = 0.0
            
            for ku_id, results in self.mapping_results.items():
                coverage_score = results.get("coverage_score", 0.0)
                gap_analysis = results.get("gap_analysis", {})
                
                # Determine coverage level
                coverage_level = "poor"
                if coverage_score >= 0.85:
                    coverage_level = "excellent"
                    report["overall_coverage_metrics"]["well_covered_ku_count"] += 1
                elif coverage_score >= 0.7:
                    coverage_level = "good"
                    report["overall_coverage_metrics"]["well_covered_ku_count"] += 1
                elif coverage_score >= 0.5:
                    coverage_level = "partial"
                    report["overall_coverage_metrics"]["partially_covered_ku_count"] += 1
                else:
                    report["overall_coverage_metrics"]["poorly_covered_ku_count"] += 1
                
                # Add to total coverage
                total_coverage += coverage_score
                
                # Create a summary for this KU
                report["ku_coverage_summary"][ku_id] = {
                    "title": results.get("ku_title", "Unknown"),
                    "coverage_score": coverage_score,
                    "coverage_level": coverage_level,
                    "matching_documents": [match["title"] for match in results.get("course_matches", [])[:3]],
                    "identified_gaps": gap_analysis.get("identified_gaps", []),
                    "llm_coverage_level": gap_analysis.get("coverage_level", 0)
                }
                
                # Add recommendations to the overall list if poorly or partially covered
                if coverage_level in ["poor", "partial"]:
                    for rec in gap_analysis.get("recommendations", []):
                        if rec not in report["recommended_improvements"]:
                            report["recommended_improvements"].append(rec)
            
            # Calculate average coverage
            if self.mapping_results:
                report["overall_coverage_metrics"]["average_coverage_score"] = total_coverage / len(self.mapping_results)
            
            # Summarize document types
            for doc_data in self.course_metadata.values():
                content_type = doc_data.get("content_type", "unknown")
                if content_type not in report["course_content_summary"]["document_types"]:
                    report["course_content_summary"]["document_types"][content_type] = 0
                report["course_content_summary"]["document_types"][content_type] += 1
            
            # Summarize topic coverage
            all_topics = set()
            for doc_data in self.course_metadata.values():
                topics = doc_data.get("cybersecurity_topics", [])
                all_topics.update(topics)
            
            for topic in all_topics:
                report["topic_coverage"][topic] = {
                    "document_count": sum(1 for doc in self.course_metadata.values() 
                                        if topic in doc.get("cybersecurity_topics", [])),
                    "related_kus": [ku_id for ku_id, results in self.mapping_results.items() 
                                  if topic in results.get("relevant_topics", [])]
                }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating course-wide KU coverage report: {str(e)}")
            return report
    
    def save_results(self) -> None:
        """Save all mapping results to the output directory."""
        logger.info(f"Saving mapping results to {self.output_dir}")
        
        try:
            # Create a timestamped directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = os.path.join(self.output_dir, f"mapping_results_{timestamp}")
            os.makedirs(results_dir, exist_ok=True)
            
            # Save individual KU mapping results
            ku_results_dir = os.path.join(results_dir, "ku_mappings")
            os.makedirs(ku_results_dir, exist_ok=True)
            
            for ku_id, results in self.mapping_results.items():
                file_path = os.path.join(ku_results_dir, f"{ku_id}_mapping.json")
                with open(file_path, 'w') as f:
                    json.dump(results, f, indent=2)
            
            # Save KU metadata
            ku_metadata_path = os.path.join(results_dir, "ku_metadata.json")
            with open(ku_metadata_path, 'w') as f:
                json.dump(self.ku_metadata, f, indent=2)
            
            # Save course metadata (truncated to avoid huge files)
            truncated_course_metadata = {}
            for filename, data in self.course_metadata.items():
                truncated_course_metadata[filename] = {
                    "title": data.get("title", "Unknown"),
                    "filepath": data.get("filepath", ""),
                    "content_type": data.get("content_type", "unknown"),
                    "cybersecurity_topics": data.get("cybersecurity_topics", []),
                    "learning_objectives": data.get("learning_objectives", []),
                    "technologies_mentioned": data.get("technologies_mentioned", []),
                    "total_chunks": data.get("total_chunks", 0)
                }
            
            course_metadata_path = os.path.join(results_dir, "course_metadata.json")
            with open(course_metadata_path, 'w') as f:
                json.dump(truncated_course_metadata, f, indent=2)
            
            # Save the course-wide coverage report
            coverage_report = self.generate_course_ku_coverage_report()
            report_path = os.path.join(results_dir, "course_coverage_report.json")
            with open(report_path, 'w') as f:
                json.dump(coverage_report, f, indent=2)
            
            logger.info(f"All results saved to {results_dir}")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
    
    def run(self) -> bool:
        """
        Run the full knowledge mapping pipeline.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("Starting knowledge mapping pipeline")
        
        # Load vector stores
        if not self.load_vector_stores():
            logger.error("Failed to load vector stores. Aborting.")
            return False
        
        # Extract metadata
        ku_metadata = self.extract_ku_metadata()
        if not ku_metadata:
            logger.error("Failed to extract KU metadata. Aborting.")
            return False
        
        course_metadata = self.extract_course_metadata()
        if not course_metadata:
            logger.error("Failed to extract course metadata. Aborting.")
            return False
        
        # Map each KU to course content
        logger.info(f"Mapping {len(ku_metadata)} knowledge units to course content")
        
        for i, (ku_id, ku_data) in enumerate(ku_metadata.items()):
            logger.info(f"Processing KU {i+1}/{len(ku_metadata)}: {ku_id}")
            mapping_results = self.map_ku_to_course_content(ku_id, ku_data)
            self.mapping_results[ku_id] = mapping_results
        
        # Save results
        self.save_results()
        
        logger.info("Knowledge mapping pipeline completed successfully")
        return True

def main():
    """Main function to run the knowledge mapping engine."""
    try:
        engine = KnowledgeMappingEngine()
        success = engine.run()
        
        if success:
            print("✅ Successfully mapped knowledge units to course content!")
            print(f"Results saved to {os.path.abspath(engine.output_dir)}")
        else:
            print("❌ Failed to complete knowledge mapping.")
    except Exception as e:
        logger.exception("Unexpected error in main function")
        print(f"❌ An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()