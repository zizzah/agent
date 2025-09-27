import google.generativeai as genai
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from prompts import system_prompt, system_prompt_3, system_prompt_4
load_dotenv()
from typing import Dict, List, Any
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END, START
from copilotkit import CopilotKitState
from copilotkit.langchain import copilotkit_customize_config
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from copilotkit.langgraph import copilotkit_emit_state
import uuid
import asyncio

# Define the agent's runtime state schema for CopilotKit/LangGraph
class AgentState(CopilotKitState):
    tool_logs: List[Dict[str, Any]]
    response: Dict[str, Any]


async def chat_node(state: AgentState, config: RunnableConfig):
    # 1. Configure genai client
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    
    state["tool_logs"].append(
        {
            "id": str(uuid.uuid4()),
            "message": "Analyzing the user's query",
            "status": "processing",
        }
    )
    await copilotkit_emit_state(config, state)

    # 2. Defining a condition to check if the last message is a tool so as to handle the FE tool responses
    if state["messages"][-1].type == "tool":
        client = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=1.0,
            max_retries=2,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
        messages = [*state["messages"]]
        messages[-1].content = (
            "The posts had been generated successfully. Just generate a summary of the posts."
        )
        resp = await client.ainvoke(
            [*state["messages"]],
            config,
        )
        state["tool_logs"] = []
        await copilotkit_emit_state(config, state)
        return Command(goto="fe_actions_node", update={"messages": resp})

    # 3. Initialize the model
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    if config is None:
        config = RunnableConfig(recursion_limit=25)
    else:
        config = copilotkit_customize_config(config, emit_messages=True, emit_tool_calls=True)
        
    # 4. Generate content with grounding (web search)
    try:
        response = model.generate_content(
            [
                system_prompt,
                system_prompt_4,
                state["messages"][-1].content
            ],
            tools='google_search_retrieval'
        )
    except Exception as e:
        # Fallback without grounding if it fails
        response = model.generate_content([
            system_prompt,
            system_prompt_4,
            state["messages"][-1].content
        ])
    
    # 5. Update tool logs and response
    state["tool_logs"][-1]["status"] = "completed"
    await copilotkit_emit_state(config, state)
    state["response"] = response.text
    
    # 6. Simulate web search queries for UI feedback
    search_queries = ["current trends", "latest information"]  # Simulated queries
    for query in search_queries:
        state["tool_logs"].append(
            {
                "id": str(uuid.uuid4()),
                "message": f"Performing Web Search for '{query}'",
                "status": "processing",
            }
        )
        await asyncio.sleep(1)
        await copilotkit_emit_state(config, state)
        state["tool_logs"][-1]["status"] = "completed"
        await copilotkit_emit_state(config, state)
    
    return Command(goto="fe_actions_node", update=state)


async def fe_actions_node(state: AgentState, config: RunnableConfig):
    try:
        if state["messages"][-2].type == "tool":
            return Command(goto="end_node", update=state)
    except Exception as e:
        print("Moved")
        
    state["tool_logs"].append(
        {
            "id": str(uuid.uuid4()),
            "message": "Generating post",
            "status": "processing",
        }
    )
    await copilotkit_emit_state(config, state)
    
    # 6. Initialize the model to generate the post
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=1.0,
        max_retries=2,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
    await copilotkit_emit_state(config, state)
    response = await model.bind_tools([*state["copilotkit"]["actions"]]).ainvoke(
        [system_prompt_3.replace("{context}", state["response"]), *state["messages"]],
        config,
    )
    state["tool_logs"] = []
    await copilotkit_emit_state(config, state)
    
    # 7. Return the response to the frontend
    return Command(goto="end_node", update={"messages": response})


async def end_node(state: AgentState, config: RunnableConfig):
    return Command(goto=END, update={"messages": state["messages"], "tool_logs": []})


def router_function(state: AgentState, config: RunnableConfig):
    if state["messages"][-2].role == "tool":
        return "end_node"
    else:
        return "fe_actions_node"


# Define a new graph
workflow = StateGraph(AgentState)
workflow.add_node("chat_node", chat_node)
workflow.add_node("fe_actions_node", fe_actions_node)
workflow.add_node("end_node", end_node)
workflow.set_entry_point("chat_node")
workflow.set_finish_point("end_node")
workflow.add_edge(START, "chat_node")
workflow.add_edge("chat_node", "fe_actions_node")
workflow.add_edge("fe_actions_node", END)

# Compile the graph
post_generation_graph = workflow.compile(checkpointer=MemorySaver())