import glob
import json
import logging
import os
from typing import List, Optional

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from neo4j import GraphDatabase
from pydantic import BaseModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()
NEO4J_URI = os.environ.get("NEO4J_URI")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Check environment variables
assert NEO4J_URI, "NEO4J_URI environment variable is not set"
assert NEO4J_USERNAME, "NEO4J_USERNAME environment variable is not set"
assert NEO4J_PASSWORD, "NEO4J_PASSWORD environment variable is not set"
assert OPENAI_API_KEY, "OPENAI_API_KEY environment variable is not set"

# Base directory containing the CLARK library
CLARK_BASE_DIR = "rag/clark_doc"

# Pydantic model for CAE and NICE units
class Unit(BaseModel):
    name: str
    description: str
    version: str

# Pydantic model for course data
class CourseData(BaseModel):
    course_name: str
    collection_name: str
    updated_time: Optional[str] = None
    contributors: Optional[List[str]] = None
    academic_levels: Optional[List[str]] = None
    topic: Optional[str] = None
    url_link: Optional[str] = None
    description: Optional[str] = None
    outcomes: Optional[List[str]] = None
    cae_units: Optional[List[Unit]] = None
    nice_units: Optional[List[Unit]] = None

# Neo4j driver setup
class Neo4jDriver:
    def __init__(self, uri: str, username: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self):
        self.driver.close()

    def create_course(self, course_data: CourseData):
        with self.driver.session() as session:
            # Create Course and Collection with BELONGS_TO relationship
            session.run(
                """
                MERGE (c:Course {name: $course_name})
                SET c.updated_time = $updated_time,
                    c.url_link = $url_link,
                    c.description = $description
                WITH c
                MERGE (col:Collection {name: $collection_name})
                MERGE (c)-[:BELONGS_TO]->(col)
                """,
                course_name=course_data.course_name,
                collection_name=course_data.collection_name,
                updated_time=course_data.updated_time,
                url_link=course_data.url_link,
                description=course_data.description,
            )
            # Link Contributors
            if course_data.contributors:
                for contributor in course_data.contributors:
                    session.run(
                        """
                        MERGE (p:Contributor {name: $contributor})
                        MERGE (c:Course {name: $course_name})
                        MERGE (p)-[:CONTRIBUTED_TO]->(c)
                        """,
                        contributor=contributor,
                        course_name=course_data.course_name,
                    )
            # Link Academic Levels
            if course_data.academic_levels:
                for level in course_data.academic_levels:
                    session.run(
                        """
                        MERGE (l:AcademicLevel {name: $level})
                        MERGE (c:Course {name: $course_name})
                        MERGE (c)-[:TARGETS]->(l)
                        """,
                        level=level,
                        course_name=course_data.course_name,
                    )
            # Link Topic
            if course_data.topic:
                session.run(
                    """
                    MERGE (t:Topic {name: $topic})
                    MERGE (c:Course {name: $course_name})
                    MERGE (c)-[:COVERS]->(t)
                    """,
                    topic=course_data.topic,
                    course_name=course_data.course_name,
                )
            # Link Outcomes
            if course_data.outcomes:
                for outcome in course_data.outcomes:
                    session.run(
                        """
                        MERGE (o:Outcome {description: $outcome})
                        MERGE (c:Course {name: $course_name})
                        MERGE (c)-[:ACHIEVES]->(o)
                        """,
                        outcome=outcome,
                        course_name=course_data.course_name,
                    )
            # Link CAE Units
            if course_data.cae_units:
                for unit in course_data.cae_units:
                    session.run(
                        """
                        MERGE (u:CAEUnit {name: $name, version: $version})
                        SET u.description = $description
                        WITH u
                        MATCH (c:Course {name: $course_name})
                        MERGE (c)-[:ALIGNS_WITH]->(u)
                        """,
                        name=unit.name,
                        description=unit.description,
                        version=unit.version,
                        course_name=course_data.course_name,
                    )
            # Link NICE Units
            if course_data.nice_units:
                for unit in course_data.nice_units:
                    session.run(
                        """
                        MERGE (u:NICEUnit {name: $name, version: $version})
                        SET u.description = $description
                        WITH u
                        MATCH (c:Course {name: $course_name})
                        MERGE (c)-[:ALIGNS_WITH]->(u)
                        """,
                        name=unit.name,
                        description=unit.description,
                        version=unit.version,
                        course_name=course_data.course_name,
                    )

# LLM setup
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

def clean_llm_response(response: str) -> str:
    """Remove Markdown code block markers from LLM response."""
    cleaned = response.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json"):].strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-len("```")].strip()
    if cleaned.startswith("```"):
        cleaned = cleaned[len("```"):].strip()
    return cleaned

def extract_data_from_readme(pdf_path: str) -> CourseData:
    """Extract structured data from a README.pdf using LLM."""
    logging.info(f"Extracting data from {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    text = "\n".join(page.page_content for page in pages)

    # Fallback metadata from path
    parts = pdf_path.split(os.sep)
    if len(parts) >= 3:
        collection_name = parts[-3]
        course_name = parts[-2]
    else:
        collection_name = "unknown_collection"
        course_name = "unknown_course"

    # Updated LLM prompt
    prompt = f"""
    You are an expert data extractor. From the following README text, extract these fields into a valid JSON object:
    - course_name (string)
    - collection_name (string)
    - updated_time (string or null)
    - contributors (list of strings)
    - academic_levels (list of strings)
    - topic (string or null)
    - url_link (string or null)
    - description (string or null)
    - outcomes (list of strings)
    - cae_units (list of objects with name, description, version)
    - nice_units (list of objects with name, description, version)

    Rules:
    - Return ONLY a valid JSON string, no additional text or explanations.
    - Use null for missing single-value fields (e.g., updated_time).
    - Use empty lists for missing list fields (e.g., contributors).
    - For description, concatenate paragraphs into a single string.
    - For outcomes, list each outcome separately.
    - For cae_units and nice_units, extract from alignment or relevant sections:
      - CAE units start with patterns like "CAE Cyber Defense (version)" or "CAE CDE (version)" followed by "- name: description".
      - NICE units start with "NICE Components vX.X.X (year)" followed by "- name: description".
      - Extract name, description, and version for each unit.
      - If the description spans multiple lines, include the entire description.
      - Ignore page numbers like "1 CLARK", "2 CLARK".
    - If no CAE or NICE units are found, return empty lists.

    Example:
    If the text contains:
    "Alignment:
    CAE Cyber Defense (2014) - Databases: Students will be able to apply security principles to the design and development of database systems and database structures
    NICE Components v1.0.0 (2024) - K0707: Knowledge of database systems and software"
    Then:
    "cae_units": [{{"name": "Databases", "description": "Students will be able to apply security principles to the design and development of database systems and database structures", "version": "2014"}}],
    "nice_units": [{{"name": "K0707", "description": "Knowledge of database systems and software", "version": "2024"}}]

    Text:
    {text}
    """

    response = llm.invoke(prompt)
    cleaned_response = clean_llm_response(response.content)
    try:
        extracted_data = json.loads(cleaned_response)
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing LLM response for {pdf_path}: {e}")
        logging.error(f"Cleaned LLM response: {cleaned_response}")
        # Fallback to minimal data
        extracted_data = {
            "course_name": course_name,
            "collection_name": collection_name,
            "updated_time": None,
            "contributors": [],
            "academic_levels": [],
            "topic": None,
            "url_link": None,
            "description": None,
            "outcomes": [],
            "cae_units": [],
            "nice_units": [],
        }

    # Ensure all fields are present
    data = {
        "course_name": extracted_data.get("course_name", course_name),
        "collection_name": extracted_data.get("collection_name", collection_name),
        "updated_time": extracted_data.get("updated_time"),
        "contributors": extracted_data.get("contributors", []) or [],
        "academic_levels": extracted_data.get("academic_levels", []) or [],
        "topic": extracted_data.get("topic"),
        "url_link": extracted_data.get("url_link"),
        "description": extracted_data.get("description"),
        "outcomes": extracted_data.get("outcomes", []) or [],
        "cae_units": extracted_data.get("cae_units", []) or [],
        "nice_units": extracted_data.get("nice_units", []) or [],
    }

    return CourseData(**data)

def load_readme_files_into_neo4j(base_dir: str, neo4j_driver: Neo4jDriver):
    """Load all README.pdf files into Neo4j."""
    readme_files = glob.glob(f"{base_dir}/*/*/README.pdf")
    if not readme_files:
        logging.warning(f"No README.pdf files found in {base_dir}/*/*/README.pdf")
        return

    logging.info(f"Found {len(readme_files)} README.pdf files")
    for readme_path in readme_files:
        try:
            course_data = extract_data_from_readme(readme_path)
            neo4j_driver.create_course(course_data)
            logging.info(f"Loaded data from {readme_path} into Neo4j")
        except Exception as e:
            logging.error(f"Error processing {readme_path}: {e}")

def main():
    neo4j_driver = Neo4jDriver(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
    load_readme_files_into_neo4j(CLARK_BASE_DIR, neo4j_driver)
    print("Complete. Check Neo4j for the results.")
    neo4j_driver.close()

if __name__ == "__main__":
    main()