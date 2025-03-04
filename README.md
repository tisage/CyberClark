# CyberClark
A multi-agent system to study cybersecurity course with Clark library.

- Supervisor Agent
    - general_conversation:
    - rag: load Clark vectorstore with metadata info (`ncae-c` course materials)
    - web_researcher: search for information that not in the vectorstore
    - CLI-Chatbot (with memory support)

## Installation
```bash
pip install python-dotenv langchain langchain_community langgraph langchain-openai faiss-cpu
```


Since Feb. 2025

v. 0.7