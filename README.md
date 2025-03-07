# CyberClark
A multi-agent system to study cybersecurity course with Clark library.

- Supervisor Agent
    - `general_conversation`: small talk with the user
    - `rag`: load Clark vectorstore with metadata info (`ncae-c` course materials)
    - `exercise`: load Clark vectorestore and find and generate exercise questions
    - `web_researcher`: search for information that not in the vectorstore
    - CLI-Chatbot (with memory support)

## Installation
```bash
pip install python-dotenv langchain langchain_community langgraph langchain-openai faiss-cpu pypdf neo4j
```
## Graph DB Neo4j
```
docker run --name neo4j -p 7474:7474 -p 7687:7687 -d -e NEO4J_AUTH=neo4j/password -e NEO4J_PLUGINS='["apoc"]'  neo4j:latest
```

## Run
```bash
python app.py
```

Since Feb. 2025

v. 0.8