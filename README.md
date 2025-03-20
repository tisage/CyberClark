# CyberClark
A multi-agent system to study cybersecurity course with Clark library.

- Supervisor Agent
    - `general_conversation`: small talk with the user
    - `rag`: load Clark vectorstore with metadata info (`ncae-c` 10 course materials)
    - `exercise`: load Clark vectorestore and find existing exercises and also generate exercise questions
    - `web_researcher`: search for information that not in the vectorstore
    - With Memory Support
    - CLI-Chatbot: with memory support
    - GUI-Chatbot: Use Streamlit to create a GUI interface

## Installation
```bash
pip install python-dotenv langchain langchain_community langgraph langchain-openai faiss-cpu pypdf neo4j streamlit
```
## Set up Graph DB Neo4j

Not Mount Local Folder
```bash
docker run --name neo4j -p 7474:7474 -p 7687:7687 -d -e NEO4J_AUTH=neo4j/password -e NEO4J_PLUGINS='["apoc"]' -e apoc.export.file.enabled=true neo4j:latest
```

Windows OS Mount Local Folder (Documents\Notebooks\neo4j)
```bash
docker run --name neo4j -p 7474:7474 -p 7687:7687 -d -e NEO4J_AUTH=neo4j/password -e NEO4J_PLUGINS='["apoc"]' -e apoc.export.file.enabled=true -v $HOME\Documents\Notebooks\neo4j:/data neo4j:latest
```

Mac/Linux OS Mount Local Folder (~/Notebooks/neo4j)
```bash
docker run --name neo4j -p 7474:7474 -p 7687:7687 -d -e NEO4J_AUTH=neo4j/password -e NEO4J_PLUGINS='["apoc"]' -e apoc.export.file.enabled=true -v ~/Notebooks/neo4j:/data neo4j:latest
```

## Run App GUI
```bash
streamlit run app_gui.py 
```

## Run App Command-line
```bash
python app_cli.py
```

---

Since Feb. 2025

v. 0.9.0