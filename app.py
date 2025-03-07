# app.py
import uuid
from typing import Dict

from langchain_core.messages import HumanMessage
from agents import build_graph  # Import the graph builder from agents.py

def chat_with_graph(graph, thread_id: str = None) -> Dict[str, str]:
    """
    Run an interactive chatbot session with the LangGraph workflow, handling multiple questions.

    Args:
        graph: The compiled LangGraph instance.
        thread_id: A unique identifier for the conversation thread. If None, a new UUID is generated.

    Returns:
        Dict containing the final conversation state (last query, final answer, and all steps).
    """
    if thread_id is None:
        thread_id = str(uuid.uuid4())
    
    config = {"configurable": {"thread_id": thread_id}}
    state = {"messages": []}
    result_info = {
        "last_query": "",
        "final_answer": "",
        "steps": [],
        "session_id": thread_id
    }

    print(f"{'='*50}\nWelcome to the Cybersecurity Chatbot!\nSession ID: {thread_id}\nType your question or 'exit' to quit.\n{'='*50}")

    while True:
        query = input("You: ").strip()
        if query.lower() in ["exit", "quit"]:
            print(f"{'='*50}\nChat session ended.\nSession ID: {thread_id}\n{'='*50}")
            break

        state["messages"].append(HumanMessage(content=query))
        result_info["last_query"] = query
        # print(f"{'='*50}\nProcessing Query: '{query}'\n{'='*50}")
        print('='*50)
        
        result = graph.invoke(state, config=config)
        messages = result["messages"]

        for msg in messages[len(state["messages"]) - 1:]:
            sender = msg.name if hasattr(msg, "name") and msg.name else ("User" if isinstance(msg, HumanMessage) else "Unknown")
            formatted_msg = f"{sender}: {msg.content}"
            if sender == "supervisor":
                step_info = f"Supervisor: Routing decision based on prior message"
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

if __name__ == "__main__":
    # Build the graph from agents.py
    graph = build_graph()
    # Run the chat interface
    result = chat_with_graph(graph)
    print(f"Session completed. Session ID was: {result['session_id']}")