import streamlit as st
import os
from agents import build_graph, OPENAI_API_KEY, TAVILY_API_KEY
import uuid
from langchain_core.messages import HumanMessage
import re
import sys

# Set page config
st.set_page_config(
    page_title="CyberClark: Cybersecurity Education Chatbot",
    page_icon="🔐",
    layout="wide"
)

# Sidebar configuration
with st.sidebar:
    st.title("CyberClark Settings")
    
    # Display LLM model info
    st.subheader("Model Information")
    st.info("Using: GPT-4o-mini")
    
    # Display API key status
    st.subheader("API Keys")
    openai_status = "✅ Loaded" if OPENAI_API_KEY else "❌ Missing"
    tavily_status = "✅ Loaded" if TAVILY_API_KEY else "❌ Missing"
    
    st.code(f"OpenAI API: {openai_status}\nTavily API: {tavily_status}")
    
    # About section
    st.subheader("About CyberClark")
    st.markdown("""
    CyberClark is an intelligent chatbot designed for cybersecurity education. 
    It leverages multiple specialized agents:
    
    - **General Conversation**: Handles greetings and small talk
    - **RAG Agent**: Searches through the CLARK library
    - **Web Researcher**: Finds information on the internet
    - **Exercise Agent**: Provides practice questions and quizzes
    
    Ask anything about cybersecurity concepts, best practices, or request exercises!
    """)

# Build the graph once (not on every query)
@st.cache_resource
def get_graph():
    return build_graph(generate_graph_image=False)

graph = get_graph()

# Initialize session state
if 'thread_id' not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

# Set up the app
st.title("CyberClark: Cybersecurity Education Chatbot")
st.markdown("Ask me anything about cybersecurity, from general topics to exercises!")

# Format the response with proper markdown
def format_message(message):
    # Replace ** with proper markdown bold
    message = re.sub(r'\*\*(.*?)\*\*', r'**\1**', message)
    
    # Handle code blocks
    message = re.sub(r'```(\w+)(.*?)```', r'```\1\2```', message, flags=re.DOTALL)
    
    # Handle quizzes and exercises with proper formatting
    message = re.sub(r'(Question \d+:)', r'**\1**', message)
    
    return message

# Debug print function that outputs to terminal
def debug_print(message, separator="-"):
    print(message)
    if separator:
        print(separator * 50)
    sys.stdout.flush()  # Ensure output is flushed immediately

# Capture user query
query = st.chat_input("Type your question here (e.g., 'What is risk management?')")

# Process query if submitted
if query:
    with st.spinner("Processing your request..."):
        # Add query to conversation with placeholder
        st.session_state.conversation.append({'user': query, 'bot': 'Thinking...'})
        
        # Debug header
        debug_print(f"{'='*50}\nProcessing Query: '{query}'\nSession ID: {st.session_state.thread_id}\n{'='*50}")
        
        # Process the query with the graph
        user_message = HumanMessage(content=query)
        state = {"messages": [user_message]}
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        
        # Track previous message count to identify new messages
        prev_msg_count = 1  # Starting with 1 user message
        
        result = graph.invoke(state, config=config)
        
        # Extract and display debug info for each step
        messages = result["messages"]
        
        for i, msg in enumerate(messages[prev_msg_count:], start=prev_msg_count):
            sender = msg.name if hasattr(msg, "name") and msg.name else ("User" if isinstance(msg, HumanMessage) else "Unknown")
            
            if sender == "supervisor":
                step_info = f"Supervisor: Routing decision based on prior message"
            elif hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_name = msg.tool_calls[0]["name"] if msg.tool_calls else "Unknown Tool"
                step_info = f"Agent: {sender}\nTool: {tool_name}\nOutput: {msg.content}"
            elif sender not in ["User", "supervisor"]:
                step_info = f"Agent: {sender}\nOutput: {msg.content}"
            else:
                step_info = f"{sender}: {msg.content}"
                
            debug_print(step_info)
        
        # Extract the final answer
        final_answer = next(
            (msg.content for msg in reversed(messages) if hasattr(msg, "name") and msg.name and msg.name not in ["supervisor", None]),
            "No answer provided."
        )
        
        # Format the response
        formatted_answer = format_message(final_answer)
        
        # Debug print the final answer
        debug_print(f"{'='*50}\nFinal Answer to User:\n{'-'*20}\n{final_answer}\n{'='*50}")
        
        # Update the conversation with the actual response
        st.session_state.conversation[-1]['bot'] = formatted_answer

# Display the conversation history
for turn in st.session_state.conversation:
    with st.chat_message("user"):
        st.markdown(turn['user'])
    with st.chat_message("assistant", avatar="🔐"):
        st.markdown(turn['bot'], unsafe_allow_html=True)

# Clear chat button
if st.session_state.conversation:
    if st.button("Clear Chat"):
        st.session_state.conversation = []
        st.session_state.thread_id = str(uuid.uuid4())
        debug_print(f"{'='*50}\nChat session cleared. New session ID: {st.session_state.thread_id}\n{'='*50}")
        st.rerun()