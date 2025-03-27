import os

from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']


import json
import os
import pickle
import tempfile

import faiss
import langchain
import numpy as np
import pandas as pd
import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Set page configuration
st.set_page_config(
    page_title="Cybersecurity Knowledge Unit Analysis",
    page_icon="🔒",
    layout="wide",
)

# Title and description
st.title("Cybersecurity Course Knowledge Unit Analyzer")
st.markdown("""
Upload cybersecurity course materials (PDFs) to analyze what knowledge units are covered.
This app will compare your course material with the cybersecurity Knowledge Units database and provide coverage details.
""")

# Function to load the knowledge units from FAISS database
@st.cache_resource
def load_knowledge_units():
    try:
        # Path to the FAISS database
        db_path = "db/cyber_ku"
        
        # Load the FAISS index
        index_file = os.path.join(db_path, "index.faiss")
        index = faiss.read_index(index_file)
        
        # Load the docstore (contains document metadata)
        docstore_file = os.path.join(db_path, "index.pkl")
        with open(docstore_file, "rb") as f:
            docstore = pickle.load(f)
            
        # Load the embedding function
        embeddings = OpenAIEmbeddings()
        
        # Reconstruct the FAISS vector store
        vectorstore = FAISS(embeddings.embed_query, index, docstore, {})
        
        # Extract knowledge units info from docstore
        ku_info = []
        for doc_id, doc in docstore.items():
            metadata = doc.metadata
            ku_info.append({
                "id": doc_id,
                "category": metadata.get("category", "Unknown"),
                "abbreviation": metadata.get("abbreviation", ""),
                "name": metadata.get("name", "Unknown"),
                "description": metadata.get("description", "No description available")
            })
        
        return vectorstore, ku_info
    except Exception as e:
        st.error(f"Error loading knowledge units database: {e}")
        return None, []

# Load the database
vectorstore, ku_info = load_knowledge_units()

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        
        # Extract basic course info
        course_info = {
            "filename": pdf_file.name,
            "page_count": len(documents),
            "title": extract_title(documents)
        }
        
        # Text splitter for processing
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        # Clean up the temporary file
        os.unlink(tmp_path)
        
        return chunks, course_info
    except Exception as e:
        os.unlink(tmp_path)
        raise e

# Function to extract course title from documents
def extract_title(documents):
    if not documents:
        return "Unknown"
    
    # Try to extract title from first page
    first_page = documents[0].page_content
    lines = first_page.split('\n')
    
    # Simple heuristic: first non-empty line that's not too long
    for line in lines:
        line = line.strip()
        if line and len(line) < 100 and len(line) > 3:
            return line
    
    return "Unknown Title"

# Function to analyze knowledge unit coverage
def analyze_ku_coverage(chunks, vectorstore):
    # Set up OpenAI LLM
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    
    # Create embeddings for the chunks
    embeddings = OpenAIEmbeddings()
    
    # Matched KUs dict to track coverage
    matched_kus = {}
    
    # Process each chunk
    for i, chunk in enumerate(chunks):
        # Get chunk embedding
        chunk_embedding = embeddings.embed_documents([chunk.page_content])[0]
        
        # Search for similar KUs in the database
        results = vectorstore.similarity_search_by_vector_with_relevance_scores(
            chunk_embedding, 
            k=3  # Get top 3 matches
        )
        
        # Filter results with relevance score > 0.75
        relevant_results = [(doc, score) for doc, score in results if score > 0.75]
        
        if relevant_results:
            # Process each relevant result
            for doc, score in relevant_results:
                ku_id = doc.metadata.get("id", "unknown")
                
                if ku_id not in matched_kus:
                    # Add new matched KU
                    matched_kus[ku_id] = {
                        "category": doc.metadata.get("category", "Unknown"),
                        "abbreviation": doc.metadata.get("abbreviation", ""),
                        "name": doc.metadata.get("name", "Unknown"),
                        "description": doc.metadata.get("description", "No description available"),
                        "confidence": score,
                        "evidence": [{"chunk_index": i, "page": chunk.metadata.get("page", 0), "text": chunk.page_content[:200] + "..."}]
                    }
                else:
                    # Update existing match with better confidence and add evidence
                    if score > matched_kus[ku_id]["confidence"]:
                        matched_kus[ku_id]["confidence"] = score
                    
                    # Add evidence if not too many already
                    if len(matched_kus[ku_id]["evidence"]) < 3:
                        matched_kus[ku_id]["evidence"].append({
                            "chunk_index": i, 
                            "page": chunk.metadata.get("page", 0), 
                            "text": chunk.page_content[:200] + "..."
                        })
    
    # Convert to list and sort by confidence
    ku_coverage = list(matched_kus.values())
    ku_coverage.sort(key=lambda x: x["confidence"], reverse=True)
    
    return ku_coverage

# Function to generate a summary of KU coverage using LLM
def generate_coverage_summary(course_info, ku_coverage):
    if not ku_coverage:
        return "No Knowledge Units were detected in this course material."
    
    llm = ChatOpenAI(temperature=0, model="gpt-4o")
    
    prompt_template = """
    You are analyzing a cybersecurity course titled "{title}" with {page_count} pages.
    
    Based on the analysis, the course covers the following Knowledge Units:
    
    {ku_list}
    
    Please provide a concise summary of the Knowledge Unit coverage of this course.
    Include information about which areas of cybersecurity are well-covered and which areas may be missing.
    
    Your summary should be 2-3 paragraphs and focus on the educational value of this course.
    """
    
    # Create a formatted list of KUs
    ku_text = ""
    for ku in ku_coverage[:10]:  # Limit to top 10 for summary
        ku_text += f"- {ku['name']} ({ku['category']}): {ku['description'][:100]}...\n"
    
    # Fill prompt template
    prompt = PromptTemplate.from_template(prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Generate summary
    summary = chain.run(
        title=course_info["title"],
        page_count=course_info["page_count"],
        ku_list=ku_text
    )
    
    return summary

# File uploader
uploaded_file = st.file_uploader("Upload Cybersecurity Course PDF", type="pdf")

if uploaded_file is not None:
    with st.spinner("Analyzing PDF content..."):
        try:
            # Extract text from PDF
            chunks, course_info = extract_text_from_pdf(uploaded_file)
            
            # Display basic course info
            st.header("Course Information")
            st.write(f"**Title:** {course_info['title']}")
            st.write(f"**Filename:** {course_info['filename']}")
            st.write(f"**Pages:** {course_info['page_count']}")
            st.write(f"**Content Chunks:** {len(chunks)}")
            
            # Analyze KU coverage
            with st.spinner("Analyzing Knowledge Unit coverage..."):
                ku_coverage = analyze_ku_coverage(chunks, vectorstore)
            
            # Generate and display summary
            with st.spinner("Generating summary..."):
                summary = generate_coverage_summary(course_info, ku_coverage)
                
            st.header("Coverage Summary")
            st.write(summary)
            
            # Display detailed results
            st.header("Knowledge Unit Coverage Details")
            
            if not ku_coverage:
                st.warning("No Knowledge Units were detected in this course material.")
            else:
                # Create tabs for different views
                tab1, tab2 = st.tabs(["Table View", "Detailed View"])
                
                with tab1:
                    # Create DataFrame for table view
                    df_data = []
                    for ku in ku_coverage:
                        df_data.append({
                            "Category": ku["category"],
                            "Name": ku["name"],
                            "Abbreviation": ku["abbreviation"],
                            "Confidence": f"{ku['confidence']:.2f}",
                            "Evidence Count": len(ku["evidence"])
                        })
                    
                    df = pd.DataFrame(df_data)
                    st.dataframe(df, use_container_width=True)
                
                with tab2:
                    # Detailed expandable view
                    for i, ku in enumerate(ku_coverage):
                        with st.expander(f"{i+1}. {ku['name']} ({ku['category']}) - Confidence: {ku['confidence']:.2f}"):
                            st.write(f"**Abbreviation:** {ku['abbreviation'] if ku['abbreviation'] else 'N/A'}")
                            st.write(f"**Description:** {ku['description']}")
                            
                            st.write("**Evidence:**")
                            for evidence in ku["evidence"]:
                                st.info(f"Page {evidence['page']+1}: {evidence['text']}")
            
            # Download options
            st.header("Download Results")
            
            # Create JSON output for download
            output_data = {
                "course_info": course_info,
                "summary": summary,
                "ku_coverage": ku_coverage
            }
            
            json_output = json.dumps(output_data, indent=2)
            
            st.download_button(
                label="Download Analysis (JSON)",
                data=json_output,
                file_name=f"ku_analysis_{course_info['filename'].replace('.pdf', '')}.json",
                mime="application/json"
            )
            
            # Create CSV for basic results
            csv_data = []
            for ku in ku_coverage:
                csv_data.append({
                    "Category": ku["category"],
                    "Name": ku["name"],
                    "Abbreviation": ku["abbreviation"],
                    "Description": ku["description"],
                    "Confidence": f"{ku['confidence']:.2f}"
                })
            
            csv_df = pd.DataFrame(csv_data)
            csv_output = csv_df.to_csv(index=False)
            
            st.download_button(
                label="Download Coverage Table (CSV)",
                data=csv_output,
                file_name=f"ku_coverage_{course_info['filename'].replace('.pdf', '')}.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Error processing PDF: {e}")

else:
    # Display some instructions when no file is uploaded
    st.info("Please upload a cybersecurity course PDF to analyze its knowledge unit coverage.")
    
    # Display info about the KU database
    if ku_info:
        st.header("Knowledge Unit Database Information")
        st.write(f"The database contains {len(ku_info)} knowledge units across various cybersecurity categories.")
        
        # Show category distribution
        categories = [ku["category"] for ku in ku_info]
        category_counts = pd.Series(categories).value_counts()
        
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.subheader("Categories")
            st.dataframe(pd.DataFrame({"Category": category_counts.index, "Count": category_counts.values}))
        
        with col2:
            st.subheader("Category Distribution")
            st.bar_chart(category_counts)
    else:
        st.warning("Knowledge Unit database not loaded properly. Check database path and structure.")