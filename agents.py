import os
from dotenv import load_dotenv
from typing import Annotated, Dict, Literal, Sequence
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command
from pydantic import BaseModel
from typing_extensions import TypedDict
from vector_store import clark_vectorstore  # Import vector store from vector_store.py

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
TAVILY_API_KEY = os.environ['TAVILY_API_KEY']

# Initialize LLM
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.8)

# Define available agents
members = ["general_conversation", "web_researcher", "rag"]
options = members + ["FINISH"]

# Supervisor system prompt
system_prompt = """
You are a supervisor managing a conversation between a user and AI agents: a general conversation agent, a RAG agent, and a web researcher agent. 

- The general conversation agent handles small talk and guides users toward cybersecurity topics.
- The RAG agent searches through the CLARK library, a compilation of high-value cybersecurity curriculum.
- The web researcher agent searches the internet for additional information.

Your task is to route the conversation to the appropriate agent or to finish the conversation based on the following rules:

1. For a new user query:
   - If it's small talk, a greeting, or a casual conversation, route to 'general_conversation'.
   - If it's a cybersecurity question, route to 'rag' to search the CLARK library.

2. After the 'general_conversation' agent responds:
   - If the conversation should continue as general discussion, route to 'FINISH'.
   - If the user asks a cybersecurity question, route to 'rag'.

3. After the 'rag' agent responds:
   - If the response adequately answers the user's query using CLARK resources, route to 'FINISH'.
   - If the response indicates that no relevant information was found in the CLARK library or does not adequately answer the query, route to 'web_researcher' agent.

4. After the 'web_researcher' agent responds, route to 'FINISH'.

To make this decision, consider:
- Is the user's message small talk or a casual greeting?
- Is the user asking about a specific cybersecurity topic?
- Did the previous agent provide a satisfactory response?

Respond with a JSON object containing the key 'next' and the value being one of: 'general_conversation', 'rag', 'web_researcher', or 'FINISH'. For example: {"next": "general_conversation"}
"""


class Router(TypedDict):
    next: Literal["general_conversation", "rag", "web_researcher", "FINISH"]

def supervisor_node(state: MessagesState) -> Command[Literal["general_conversation", "rag", "web_researcher", "__end__"]]:
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = llm.with_structured_output(Router).invoke(messages)
    goto = response["next"]
    print(f"Next Worker: {goto}")
    if goto == "FINISH":
        goto = END
    return Command(goto=goto)

# Define AgentState
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Function to create an agent
def create_agent(llm, tools: list, system_prompt: str = ""):
    """Create a LangGraph agent with specified tools and system prompt."""
    llm_with_tools = llm.bind_tools(tools)
    def chatbot(state: AgentState):
        messages = [{"role": "system", "content": system_prompt}] + state["messages"] if system_prompt else state["messages"]
        return {"messages": [llm_with_tools.invoke(messages)]}
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("agent", chatbot)
    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_conditional_edges("agent", tools_condition)
    graph_builder.add_edge("tools", "agent")
    graph_builder.set_entry_point("agent")
    return graph_builder.compile()

# General Conversation Agent
general_conversation_system_prompt = """
You are a friendly cybersecurity teaching assistant named CyberGuide. Your primary goal is to help students learn about cybersecurity concepts.

For general conversation or small talk:
1. Respond in a friendly, conversational manner
2. Keep responses brief and engaging
3. When appropriate, gently guide the conversation toward cybersecurity topics with suggestions like:
   "Would you like to learn about network security basics?"
   "Have you considered exploring topics like encryption or penetration testing?"
   "I can help you understand cybersecurity concepts from the CLARK library. What topic interests you?"

Remember that you're designed to be both approachable for beginners and helpful for more advanced cybersecurity students.
"""

general_conversation_agent = create_agent(llm, [], general_conversation_system_prompt)

def general_conversation_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    result = general_conversation_agent.invoke(state)
    return Command(
        update={"messages": [HumanMessage(content=result["messages"][-1].content, name="general_conversation")]},
        goto="supervisor",
    )

# Web Researcher Agent
web_search_tool = TavilySearchResults(max_results=2)
web_researcher_system_prompt = """
You are an AI assistant searching the internet for cybersecurity info. Use web_search_tool and include sources in your response (e.g., URLs).
"""
websearch_agent = create_agent(llm, [web_search_tool], web_researcher_system_prompt)

def web_research_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    result = websearch_agent.invoke(state)
    return Command(
        update={"messages": [HumanMessage(content=result["messages"][-1].content, name="web_researcher")]},
        goto="supervisor"
    )

# RAG Agent
rag_system_prompt = """
You are CyberGuide, retrieving cybersecurity resources from the CLARK library using clark_retriever_tool.

Include:
1. Retrieved content, clearly organized.
2. Source metadata (collection, course, module).
3. Educational explanations for students.
4. Extra clarification for complex content.

Maintain a helpful, encouraging tone.
"""
class RagToolSchema(BaseModel):
    question: str

@tool(args_schema=RagToolSchema)
def clark_retriever_tool(question: str) -> str:
    """
    Retrieve cybersecurity resources from the CLARK library.
    Returns content and metadata (collection, course, module, etc.).
    """
    print("INSIDE CLARK RETRIEVER NODE")
    retriever = clark_vectorstore.as_retriever(search_kwargs={"k": 3})
    retriever_result = retriever.invoke(question)
    if not retriever_result:
        return "No relevant cybersecurity information found in the CLARK library."
    
    formatted_response = []
    for doc in retriever_result:
        metadata = doc.metadata
        entry = (
            f"**Content:**\n{doc.page_content}\n\n"
            f"**Metadata:**\n"
            f"- Collection: {metadata['collection_name']}\n"
            f"- Course: {metadata['course_name']}\n"
            f"- Module: {metadata['module_name']}\n"
            f"- File: {metadata['file_name']}\n"
            f"- Path: {metadata['file_path']}"
        )
        formatted_response.append(entry)
    return "\n\n---\n\n".join(formatted_response)

rag_agent = create_agent(llm, [clark_retriever_tool], rag_system_prompt)

def rag_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    result = rag_agent.invoke(state)
    return Command(
        update={"messages": [HumanMessage(content=result["messages"][-1].content, name="rag")]},
        goto="supervisor"
    )

# Build and compile the graph
builder = StateGraph(MessagesState)
builder.add_edge(START, "supervisor")
builder.add_node("supervisor", supervisor_node)
builder.add_node("general_conversation", general_conversation_node)
builder.add_node("web_researcher", web_research_node)
builder.add_node("rag", rag_node)
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

try:
    graph_image = graph.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(graph_image)
    print("Graph saved as 'graph.png' in the current directory.")
except Exception as e:
    print(f"Error saving graph: {e}")

def chat_with_graph(graph, thread_id: str) -> Dict[str, str]:
    """
    Run an interactive chatbot session with the LangGraph workflow.
    
    Args:
        graph: Compiled LangGraph instance.
        thread_id: Unique identifier for the conversation thread.
    
    Returns:
        Dict with last query, final answer, and conversation steps.
    """
    config = {"configurable": {"thread_id": thread_id}}
    state = {"messages": []}
    result_info = {"last_query": "", "final_answer": "", "steps": []}
    
    print(f"{'='*50}\nWelcome to the Cybersecurity Chatbot!\nType your question or 'exit' to quit.\n{'='*50}")
    
    while True:
        query = input("You: ").strip()
        if query.lower() in ["exit", "quit"]:
            print(f"{'='*50}\nChat session ended.\n{'='*50}")
            break
        
        state["messages"].append(HumanMessage(content=query))
        result_info["last_query"] = query
        print(f"{'='*50}\nProcessing Query: '{query}'\n{'='*50}")
        
        result = graph.invoke(state, config=config)
        messages = result["messages"]
        
        for msg in messages[len(state["messages"]) - 1:]:
            sender = msg.name if hasattr(msg, "name") and msg.name else ("User" if isinstance(msg, HumanMessage) else "Unknown")
            formatted_msg = f"{sender}: {msg.content}"
            if sender == "supervisor":
                step_info = "Supervisor: Routing decision based on prior message"
            elif hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_name = msg.tool_calls[0]["name"] if msg.tool_calls else "Unknown Tool"
                step_info = f"Agent: {sender}\nTool: {tool_name}\nOutput: {msg.content}"
            elif sender not in ["User", "supervisor"]:
                step_info = f"Agent: {sender}\nOutput: {msg.content}"
            else:
                step_info = formatted_msg
            print(step_info)
            print("-" * 50)
            result_info["steps"].append(step_info)
        
        final_answer = next(
            (msg.content for msg in reversed(messages) if hasattr(msg, "name") and msg.name and msg.name not in ["supervisor", None]),
            "No answer provided."
        )
        result_info["final_answer"] = final_answer
        print(f"{'='*50}\nFinal Answer to You:\n{'-'*20}\n{final_answer}\n{'='*50}")
        
        state = result
    
    return result_info

# Run the chatbot
if __name__ == "__main__":
    result = chat_with_graph(graph, "test_12345")