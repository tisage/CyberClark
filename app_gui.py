# streamlit_app.py
import os
import uuid
import streamlit as st
from typing import Dict, List, Optional, Union
from langchain_core.messages import HumanMessage, AIMessage
from agents import build_graph
import time
import re

# Configure the Streamlit page
st.set_page_config(
    page_title="CyberGuide - Cybersecurity Education Assistant",
    page_icon="🔒",
    layout="wide",
)

# Custom CSS for better formatting
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #f5f7f9;
    }
    
    /* Message container styling */
    .msg-container {
        padding: 10px 15px;
        border-radius: 15px;
        margin-bottom: 10px;
        position: relative;
        max-width: 90%;
    }
    
    /* User message styling */
    .user-msg {
        background-color: #e6f7ff;
        border: 1px solid #91d5ff;
        margin-left: auto;
        margin-right: 0;
        text-align: right;
    }
    
    /* Bot message styling */
    .bot-msg {
        background-color: #ffffff;
        border: 1px solid #d9d9d9;
        margin-left: 0;
        margin-right: auto;
    }
    
    /* Agent label styling */
    .agent-label {
        font-size: 12px;
        color: #888;
        margin-bottom: 5px;
        font-weight: bold;
    }
    
    /* Code block styling */
    pre {
        background-color: #f0f2f5;
        padding: 10px;
        border-radius: 5px;
        overflow-x: auto;
    }
    
    /* Table formatting */
    table {
        width: 100%;
        border-collapse: collapse;
    }
    
    th, td {
        border: 1px solid #ddd;
        padding: 8px;
    }
    
    th {
        background-color: #f2f2f2;
    }
    
    /* Metadata section */
    .metadata {
        background-color: #f9f9f9;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
        border-left: 3px solid #1890ff;
    }
    
    /* Divider */
    hr {
        margin: 15px 0;
        border: 0;
        border-top: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)

def create_session_id() -> str:
    """Create a unique session ID."""
    return str(uuid.uuid4())

def initialize_session():
    """Initialize the session state variables."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = create_session_id()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "langchain_messages" not in st.session_state:
        st.session_state.langchain_messages = []
    
    if "graph" not in st.session_state:
        st.session_state.graph = build_graph()

def format_message_content(content: str) -> str:
    """Format message content with proper markdown and styling."""
    # Handle Markdown code blocks
    content = re.sub(r'```(\w+)?\n(.*?)\n```', r'<pre><code>\2</code></pre>', content, flags=re.DOTALL)
    
    # Handle bold text
    content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', content)
    
    # Format metadata sections
    content = re.sub(
        r'\*\*Metadata:\*\*\n((?:- .*\n)+)',
        r'<div class="metadata"><strong>Metadata:</strong><br>\1</div>',
        content
    )
    
    # Convert bullet points to HTML lists
    content = re.sub(
        r'(?:^|\n)((?:- .*(?:\n|$))+)',
        lambda m: f'<ul>{"".join(f"<li>{line[2:]}</li>" for line in m.group(1).split("\n") if line.strip())}</ul>',
        content
    )
    
    # Format sections with headings
    content = re.sub(
        r'\*\*(.*?):\*\*\n',
        r'<h4>\1:</h4>',
        content
    )
    
    # Handle horizontal rules
    content = content.replace("\n\n---\n\n", '<hr>')
    
    return content

def display_messages():
    """Display all messages in the chat."""
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="msg-container user-msg">
                <div>{message["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            agent = message.get("agent", "CyberGuide")
            formatted_content = format_message_content(message["content"])
            
            st.markdown(f"""
            <div class="msg-container bot-msg">
                <div class="agent-label">{agent}</div>
                <div>{formatted_content}</div>
            </div>
            """, unsafe_allow_html=True)

def chat_with_cybersecurity_agent(user_input: str, thread_id: str) -> Dict:
    """
    Process the user input through the LangGraph workflow.
    
    Args:
        user_input: The text input from the user
        thread_id: The session ID for conversation continuity
        
    Returns:
        Dict containing the final conversation state
    """
    config = {"configurable": {"thread_id": thread_id}}
    
    # Create a state with the current message history
    state = {"messages": st.session_state.langchain_messages + [HumanMessage(content=user_input)]}
    
    # Placeholder for step information
    steps = []
    
    # Invoke the graph
    with st.status("CyberGuide is thinking...", expanded=False) as status:
        status.update(label="Processing your question...")
        result = st.session_state.graph.invoke(state, config=config)
        
        # Extract messages from result
        messages = result["messages"]
        
        # Get the new messages only (skip previously processed messages)
        new_messages = messages[len(st.session_state.langchain_messages):]
        
        # Process the messages to extract steps and final answer
        final_answer = None
        for msg in new_messages:
            sender = getattr(msg, "name", None) or (
                "User" if isinstance(msg, HumanMessage) else "CyberGuide"
            )
            
            if sender not in ["User", "supervisor"]:
                final_answer = msg.content
                steps.append({"agent": sender, "content": msg.content})
        
        # Update session state with new messages
        st.session_state.langchain_messages = messages
        
        status.update(label="Ready!", state="complete")
    
    return {
        "final_answer": final_answer or "I couldn't process your request. Please try again.",
        "steps": steps,
    }

def process_user_input():
    """Process the user input and update the chat."""
    user_input = st.session_state.user_input
    
    if user_input:
        # Add user message to the chat
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Show typing indicator
        with st.chat_message("assistant"):
            typing_placeholder = st.empty()
            typing_placeholder.markdown("...")
        
        # Process with the agent
        result = chat_with_cybersecurity_agent(user_input, st.session_state.session_id)
        
        # Remove typing indicator
        typing_placeholder.empty()
        
        # Add agent steps to the chat history
        for step in result["steps"]:
            st.session_state.messages.append({
                "role": "assistant", 
                "agent": step["agent"], 
                "content": step["content"]
            })
        
        # Clear the input box
        st.session_state.user_input = ""

def sidebar():
    """Create the sidebar with session information and controls."""
    with st.sidebar:
        st.title("CyberGuide")
        st.subheader("Your Cybersecurity Education Assistant")
        
        st.markdown("---")
        
        st.subheader("Session Information")
        st.code(f"Session ID: {st.session_state.session_id}")
        
        if st.button("Start New Session"):
            # Reset the session
            st.session_state.session_id = create_session_id()
            st.session_state.messages = []
            st.session_state.langchain_messages = []
            st.rerun()
        
        st.markdown("---")
        
        st.markdown("""
        ### About CyberGuide
        
        CyberGuide is an AI assistant that helps you learn about cybersecurity. It can:
        
        - Answer questions about cybersecurity concepts
        - Search the CLARK library for educational resources
        - Provide exercises and practice questions
        - Find additional information from the web
        
        All content is sourced from the CLARK library, with alignments to NICE and CAE frameworks.
        """)

def main():
    """Main function to run the Streamlit app."""
    # Initialize session state
    initialize_session()
    
    # Setup sidebar
    sidebar()
    
    # Main content area
    st.title("CyberGuide: Cybersecurity Education Assistant")
    
    # Display the chat messages
    display_messages()
    
    # Chat input
    st.text_input(
        "Ask a question about cybersecurity:",
        key="user_input",
        on_change=process_user_input,
        placeholder="e.g., What is cross-site scripting?",
    )
    
    # Add some helpful example questions
    st.markdown("---")
    st.markdown("### Example Questions")
    example_cols = st.columns(3)
    
    examples = [
        "What is the OWASP Top 10?",
        "Can you explain public key cryptography?",
        "I want to practice network security concepts",
        "What are common password vulnerabilities?",
        "How do I defend against SQL injection?",
        "Give me exercises on penetration testing"
    ]
    
    for i, col in enumerate(example_cols):
        for j in range(2):
            idx = i * 2 + j
            if idx < len(examples):
                if col.button(examples[idx], key=f"example_{idx}"):
                    st.session_state.user_input = examples[idx]
                    process_user_input()

if __name__ == "__main__":
    main()