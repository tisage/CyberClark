import streamlit as st
from agents import build_graph
import uuid
from langchain_core.messages import HumanMessage

# Build the graph with agents and tools
graph = build_graph()

# Initialize session state
if 'thread_id' not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

# Set up the app
st.title("CyberClark: Cybersecurity Education Chatbot")
st.markdown("Ask me anything about cybersecurity, from general topics to exercises!")

# Capture user query
query = st.chat_input("Type your question here (e.g., 'What is risk management?')")

# Process query if submitted
if query:
    with st.spinner("Processing your request..."):
        # Add query to conversation with placeholder
        st.session_state.conversation.append({'user': query, 'bot': 'Thinking...'})
        
        # Process the query with the graph
        user_message = HumanMessage(content=query)
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        result = graph.invoke({"messages": [user_message]}, config=config)
        
        # Extract the AI's response
        messages = result["messages"]
        final_answer = next(
            (msg.content for msg in reversed(messages) if hasattr(msg, "name") and msg.name and msg.name not in ["supervisor", None]),
            "No answer provided."
        )
        
        # Update the conversation with the actual response
        st.session_state.conversation[-1]['bot'] = final_answer

# Display the conversation history
for turn in st.session_state.conversation:
    with st.chat_message("user"):
        st.markdown(turn['user'])
    with st.chat_message("assistant"):
        st.markdown(turn['bot'], unsafe_allow_html=True)