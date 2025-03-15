# agents.py
import os
from typing import Annotated, Dict, Literal, Sequence

from dotenv import load_dotenv
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

from vector_store import clark_retriever_tool, exercise_retriever_tool

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
TAVILY_API_KEY = os.environ.get('TAVILY_API_KEY', '')

# Initialize LLM
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.8)

# Define available agents
members = ["general_conversation", "web_researcher", "rag", "exercise"]
options = members + ["FINISH"]

# Supervisor system prompt
system_prompt = """
You are a supervisor managing a conversation between a user and AI agents: a general conversation agent, a RAG agent, a web researcher agent, and an exercise agent.

- The general conversation agent handles small talk and guides users toward cybersecurity topics.
- The RAG agent searches through the CLARK library for cybersecurity information.
- The web researcher agent searches the internet for additional information.
- The exercise agent provides exercises or generates questions for students to practice cybersecurity topics.

Your task is to route the conversation to the appropriate agent or to finish the conversation based on the following rules:

1. For any user query:
   - If it's small talk, a greeting, or a casual conversation, route to 'general_conversation'.
   - If it's a cybersecurity question, route to 'rag' to search the CLARK library.
   - If it's a request for exercises or practice questions on a cybersecurity topic, route to 'exercise'.
   - If it's a question that cannot be answered by the other agents, route to 'web_researcher'.
   - The conversation can freely move between agents as needed.
2. After the 'general_conversation' agent responds, route to 'FINISH'.
3. After the 'rag' agent responds, route to 'FINISH'.
4. After the 'web_researcher' agent responds, route to 'FINISH'.
5. After the 'exercise' agent responds, route to 'FINISH'.

Your routing should be flexible based on each new user message, not bound by a predetermined sequence.

Respond with a JSON object containing the key 'next' and the value being one of: 'general_conversation', 'rag', 'web_researcher', 'exercise', or 'FINISH'. For example: {"next": "exercise"}
"""

class Router(TypedDict):
    next: Literal["general_conversation", "rag", "web_researcher", "exercise", "FINISH"]

def supervisor_node(state: MessagesState) -> Command[Literal["general_conversation", "rag", "web_researcher", "exercise", "__end__"]]:
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
def create_agent(llm, tools, system_prompt: str = ""):
    """Create an agent with the specified LLM, tools, and system prompt."""
    llm_with_tools = llm.bind_tools(tools)
    def chatbot(state: AgentState):
        messages = (
            [{"role": "system", "content": system_prompt}] + state["messages"]
            if system_prompt
            else state["messages"]
        )
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
You are an AI assistant tasked with searching the internet for information to answer the user's query about cybersecurity.
Use the web_search_tool to find relevant information.
In your response, include the information found and cite the sources (e.g., URLs or references provided by the tool).
"""
websearch_agent = create_agent(llm, [web_search_tool], web_researcher_system_prompt)

def web_research_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    result = websearch_agent.invoke(state)
    return Command(
        update={"messages": [HumanMessage(content=result["messages"][-1].content, name="web_researcher")]},
        goto="supervisor",
    )

# RAG Agent
rag_system_prompt = """
You are CyberGuide, an AI assistant tasked with retrieving and presenting cybersecurity educational resources from the CLARK library.

When you receive a query, use the clark_retriever_tool to search the library.
The tool will return relevant content along with metadata such as collection name, course name, and module name.

In your response:
1. Include the retrieved content in a clear, organized way
2. You MUST ALWAYS include the source metadata for EACH piece of information in this format:
   - Source: [Collection Name] - [Course Name] - [Module Name]
3. Explain cybersecurity concepts in an educational manner appropriate for students
4. If the content is complex, provide additional explanations to make it more accessible
5. After presenting the information, suggest to the user: "If you'd like to practice with some exercises on this topic, just let me know!"

IMPORTANT: Never omit the source metadata. This is critical for academic integrity and allowing users to explore more content from the same source.

Always maintain a helpful, encouraging tone to support students in their cybersecurity learning journey.

If you cannot find relevant information in the CLARK library, clearly state this so the supervisor can route to the web researcher agent.
"""
rag_agent = create_agent(llm, [clark_retriever_tool], rag_system_prompt)

def rag_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    result = rag_agent.invoke(state)
    return Command(
        update={"messages": [HumanMessage(content=result["messages"][-1].content, name="rag")]},
        goto="supervisor",
    )

# Exercise Agent
exercise_system_prompt = """
You are CyberGuide, an AI assistant tasked with providing cybersecurity exercises or generating questions for students.

When a user asks for exercises on a topic, follow these steps:

1. Use the exercise_retriever_tool to search for existing exercises in the CLARK library.

2. If the tool returns exercises (content that doesn't start with "No existing exercises found"):
   - Present them clearly to the user with this prefix: "**EXISTING EXERCISES FROM CLARK LIBRARY:**"
   - Include all the source metadata to give proper attribution
   - Format the exercises in a clear, structured way

3. If the tool returns "No existing exercises found in the CLARK library for this topic.":
   - Clearly state: "**I couldn't find existing exercises in our database for this topic, so I'll generate some practice questions for you.**"
   - Generate 3-5 high-quality questions related to the topic
   - Explicitly label these as "**AI-GENERATED PRACTICE QUESTIONS:**"
   - Make the questions relevant to the topic and suitable for a student's learning level
   - Create a mix of multiple-choice, true/false, and short answer questions

4. If appropriate, you can provide both: first show the retrieved exercises, then offer additional AI-generated questions with clear labeling of each section.

Always maintain an encouraging tone to support students in their learning. Ensure that any content you present or generate is accurate and educationally valuable.
"""
exercise_agent = create_agent(llm, [exercise_retriever_tool], exercise_system_prompt)

def exercise_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    result = exercise_agent.invoke(state)
    return Command(
        update={"messages": [HumanMessage(content=result["messages"][-1].content, name="exercise")]},
        goto="supervisor",
    )

# Build and compile the graph
def build_graph(generate_graph_image=False):
    builder = StateGraph(MessagesState)
    builder.add_edge(START, "supervisor")
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("general_conversation", general_conversation_node)
    builder.add_node("web_researcher", web_research_node)
    builder.add_node("rag", rag_node)
    builder.add_node("exercise", exercise_node)

    # Add memory to persist conversation state
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)

    # Save graph visualization (optional, can be commented out if not needed)
    if generate_graph_image:
        try:
            graph_image = graph.get_graph().draw_mermaid_png()
            with open("graph.png", "wb") as f:
                f.write(graph_image)
            print("Graph saved as 'graph.png' in the current directory.")
        except Exception as e:
            print(f"Error saving graph: {e}")

    return graph