from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from prompts import system_prompt, system_prompt_copywriting
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
import re
import json


# Define the agent's runtime state schema for CopilotKit/LangGraph
class CopywriterAgentState(CopilotKitState):
    tool_logs: List[Dict[str, Any]]
    response: str
    copywriting_brief: Dict[str, Any]
    copy_variations: Dict[str, str]
    parsed_copy: Dict[str, Any]


def parse_copy_sections(text: str) -> Dict[str, Any]:
    """
    Parse the generated copy into structured sections
    """
    result = {
        "headlines": [],
        "body": "",
        "cta": "",
        "rationale": ""
    }
    
    # Extract headlines
    headline_pattern = r'\[HEADLINE OPTIONS?\](.*?)(?:\[|$)'
    headline_match = re.search(headline_pattern, text, re.DOTALL | re.IGNORECASE)
    if headline_match:
        headlines_text = headline_match.group(1).strip()
        # Split by numbered list or newlines
        headlines = re.findall(r'(?:\d+[.):]\s*|\n-\s*)(.+?)(?=\d+[.):]\s*|\n-\s*|$)', headlines_text, re.MULTILINE)
        if not headlines:
            # Try splitting by newlines
            headlines = [h.strip() for h in headlines_text.split('\n') if h.strip() and len(h.strip()) > 10]
        result["headlines"] = [h.strip() for h in headlines if h.strip()][:5]
    
    # Extract body copy
    body_pattern = r'\[BODY(?:\s+COPY)?\](.*?)(?:\[|$)'
    body_match = re.search(body_pattern, text, re.DOTALL | re.IGNORECASE)
    if body_match:
        result["body"] = body_match.group(1).strip()
    
    # Extract CTA
    cta_pattern = r'\[CTA\](.*?)(?:\[|$)'
    cta_match = re.search(cta_pattern, text, re.DOTALL | re.IGNORECASE)
    if cta_match:
        result["cta"] = cta_match.group(1).strip()
    
    # Extract rationale
    rationale_pattern = r'\[RATIONALE\](.*?)(?:\[|$)'
    rationale_match = re.search(rationale_pattern, text, re.DOTALL | re.IGNORECASE)
    if rationale_match:
        result["rationale"] = rationale_match.group(1).strip()
    
    # Fallback: if no structured sections found, try to extract from plain text
    if not result["headlines"]:
        # Look for lines that might be headlines (short, impactful lines at the start)
        lines = text.split('\n')
        potential_headlines = [line.strip() for line in lines[:10] 
                              if line.strip() and len(line.strip()) < 100 and len(line.strip()) > 15]
        result["headlines"] = potential_headlines[:5]
    
    if not result["body"]:
        # Take the main content as body
        result["body"] = text[:500] if len(text) > 500 else text
    
    if not result["cta"]:
        # Look for common CTA phrases
        cta_keywords = ["get started", "buy now", "sign up", "learn more", "try free", "start today"]
        for line in text.split('\n'):
            if any(keyword in line.lower() for keyword in cta_keywords):
                result["cta"] = line.strip()
                break
        if not result["cta"]:
            result["cta"] = "Get Started Today"
    
    return result


async def research_node(state: CopywriterAgentState, config: RunnableConfig):
    """Phase 1: Research & Analysis"""
    model = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    
    state["tool_logs"].append({
        "id": str(uuid.uuid4()),
        "message": "🔍 Analyzing copywriting requirements and gathering market intelligence",
        "status": "processing",
    })
    await copilotkit_emit_state(config, state)

    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    model_config = types.GenerateContentConfig(tools=[grounding_tool])
    
    if config is None:
        config = RunnableConfig(recursion_limit=25)
    else:
        config = copilotkit_customize_config(config, emit_messages=True, emit_tool_calls=True)

    research_prompt = f"""
    {system_prompt}
    
    RESEARCH PHASE: Analyze the user's copywriting request and search for:
    1. Current market trends related to the topic
    2. Competitor copy examples
    3. Target audience pain points and desires
    4. Best-performing copy styles for this niche
    5. Recent successful campaigns
    
    User Request: {state["messages"][-1].content}
    """
    
    response = model.models.generate_content(
        model="gemini-2.5-pro",
        contents=[types.Content(role="user", parts=[types.Part(text=research_prompt)])],
        config=model_config,
    )

    state["tool_logs"][-1]["status"] = "completed"
    await copilotkit_emit_state(config, state)
    state["response"] = response.text

    # Log web search queries
    if hasattr(response.candidates[0], 'grounding_metadata') and response.candidates[0].grounding_metadata:
        for query in response.candidates[0].grounding_metadata.web_search_queries:
            state["tool_logs"].append({
                "id": str(uuid.uuid4()),
                "message": f"🌐 Researching: '{query}'",
                "status": "processing",
            })
            await asyncio.sleep(0.5)
            await copilotkit_emit_state(config, state)
            state["tool_logs"][-1]["status"] = "completed"
            await copilotkit_emit_state(config, state)

    return Command(goto="strategy_node", update=state)


async def strategy_node(state: CopywriterAgentState, config: RunnableConfig):
    """Phase 2: Strategic Framework"""
    state["tool_logs"].append({
        "id": str(uuid.uuid4()),
        "message": "🎯 Developing strategic copywriting framework",
        "status": "processing",
    })
    await copilotkit_emit_state(config, state)

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=0.7,
        max_retries=2,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    strategy_prompt = f"""
    Based on research findings, create a strategic copywriting brief:

    Research Context: {state.get("response", "")}
    User Request: {state["messages"][-1].content}

    Provide:
    1. Recommended persuasion framework (AIDA, PAS, FAB, BAB)
    2. Target audience profile and psychographics
    3. Key emotional triggers to leverage
    4. Tone and voice recommendations
    5. Primary and secondary conversion goals
    6. Unique value propositions to emphasize
    7. Potential objections to address
    """

    response = await model.ainvoke([strategy_prompt], config)
    
    state["copywriting_brief"] = {
        "strategy": response.content,
        "timestamp": str(uuid.uuid4())
    }
    
    state["tool_logs"][-1]["status"] = "completed"
    await copilotkit_emit_state(config, state)

    return Command(goto="copywriting_node", update=state)


async def copywriting_node(state: CopywriterAgentState, config: RunnableConfig):
    """Phase 3: Crafting the Copy"""
    state["tool_logs"].append({
        "id": str(uuid.uuid4()),
        "message": "✍️ Crafting compelling copy with conversion-focused techniques",
        "status": "processing",
    })
    await copilotkit_emit_state(config, state)

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=0.9,
        max_retries=2,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    copywriting_prompt = f"""
    {system_prompt_copywriting}

    STRATEGIC BRIEF: {state["copywriting_brief"].get("strategy", "")}
    RESEARCH INSIGHTS: {state.get("response", "")}
    USER REQUEST: {state["messages"][-1].content}

    Create high-converting copy with these sections:

    [HEADLINE OPTIONS]
    - Provide 3-5 compelling headline variations
    - Each headline should be attention-grabbing and benefit-focused

    [BODY COPY]
    - Write persuasive body copy (200-400 words)
    - Use the recommended persuasion framework
    - Include benefits, proof points, and emotional triggers
    - Address potential objections naturally

    [CTA]
    - Create a powerful call-to-action
    - Make it clear, urgent, and friction-free

    [RATIONALE]
    - Explain your strategic choices (50-100 words)
    - Mention the framework used and why

    Use exactly these section markers: [HEADLINE OPTIONS], [BODY COPY], [CTA], [RATIONALE]
    """

    response = await model.ainvoke([copywriting_prompt], config)
    
    state["response"] = response.content
    state["parsed_copy"] = parse_copy_sections(response.content)
    
    state["tool_logs"][-1]["status"] = "completed"
    await copilotkit_emit_state(config, state)

    return Command(goto="variation_node", update=state)


async def variation_node(state: CopywriterAgentState, config: RunnableConfig):
    """Phase 4: Generate Variations"""
    state["tool_logs"].append({
        "id": str(uuid.uuid4()),
        "message": "🔄 Generating A/B testing variations",
        "status": "processing",
    })
    await copilotkit_emit_state(config, state)

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=1.0,
        max_retries=2,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    variation_prompt = f"""
    Based on this primary copy:
    {state.get("response", "")}

    Generate 2 complete alternative versions (150-200 words each):

    VARIATION A - EMOTIONAL/STORY-DRIVEN:
    [Write a version that emphasizes emotional storytelling, personal transformation, and aspirational language]

    VARIATION B - DATA/LOGIC-DRIVEN:
    [Write a version that emphasizes statistics, logical benefits, and rational decision-making]

    Keep the core value proposition but vary the angle and tone significantly.
    Make each variation complete and ready to use.
    """

    response = await model.ainvoke([variation_prompt], config)
    
    # Parse variations
    variations_text = response.content
    emotional_match = re.search(r'VARIATION A.*?:(.*?)(?=VARIATION B|$)', variations_text, re.DOTALL | re.IGNORECASE)
    logical_match = re.search(r'VARIATION B.*?:(.*?)$', variations_text, re.DOTALL | re.IGNORECASE)
    
    state["copy_variations"] = {
        "emotional": emotional_match.group(1).strip() if emotional_match else "Emotional variation coming soon...",
        "logical": logical_match.group(1).strip() if logical_match else "Logical variation coming soon..."
    }
    
    state["tool_logs"][-1]["status"] = "completed"
    await copilotkit_emit_state(config, state)

    return Command(goto="fe_actions_node", update=state)


async def fe_actions_node(state: CopywriterAgentState, config: RunnableConfig):
    """Frontend Actions: Prepare copy for delivery"""
    try:
        if state["messages"][-2].type == "tool":
            return Command(goto="end_node", update=state)
    except (IndexError, AttributeError):
        pass

    state["tool_logs"].append({
        "id": str(uuid.uuid4()),
        "message": "📋 Finalizing copy for delivery",
        "status": "processing",
    })
    await copilotkit_emit_state(config, state)

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=0.8,
        max_retries=2,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    # Prepare structured copy data
    parsed = state.get("parsed_copy", {})
    variations = state.get("copy_variations", {})
    
    # Create the structured response for frontend
    copy_data = {
        "primary": {
            "headlines": parsed.get("headlines", ["Compelling Headline Here"]),
            "body": parsed.get("body", ""),
            "cta": parsed.get("cta", "Get Started Today"),
            "rationale": parsed.get("rationale", "")
        },
        "variations": {
            "emotional": variations.get("emotional", ""),
            "logical": variations.get("logical", "")
        }
    }

    # Format message to invoke frontend action
    formatted_message = f"""
    Generate the copy using the generate_copy action with this data:
    
    Primary Headlines: {json.dumps(copy_data['primary']['headlines'])}
    Body Copy: {copy_data['primary']['body'][:200]}...
    CTA: {copy_data['primary']['cta']}
    
    Variations:
    - Emotional: {copy_data['variations']['emotional'][:100]}...
    - Logical: {copy_data['variations']['logical'][:100]}...
    
    Call the generate_copy action with the complete copy data.
    """

    response = await model.bind_tools([*state["copilotkit"]["actions"]]).ainvoke(
        [formatted_message, *state["messages"]],
        config,
    )

    state["tool_logs"] = []
    await copilotkit_emit_state(config, state)

    return Command(goto="end_node", update={"messages": response})


async def end_node(state: CopywriterAgentState, config: RunnableConfig):
    """Terminal node"""
    return Command(goto=END, update={"messages": state["messages"], "tool_logs": []})


# Build the workflow graph
workflow = StateGraph(CopywriterAgentState)

workflow.add_node("research_node", research_node)
workflow.add_node("strategy_node", strategy_node)
workflow.add_node("copywriting_node", copywriting_node)
workflow.add_node("variation_node", variation_node)
workflow.add_node("fe_actions_node", fe_actions_node)
workflow.add_node("end_node", end_node)

workflow.set_entry_point("research_node")
workflow.set_finish_point("end_node")
workflow.add_edge(START, "research_node")
workflow.add_edge("research_node", "strategy_node")
workflow.add_edge("strategy_node", "copywriting_node")
workflow.add_edge("copywriting_node", "variation_node")
workflow.add_edge("variation_node", "fe_actions_node")
workflow.add_edge("fe_actions_node", "end_node")

copywriter_agent_graph = workflow.compile(checkpointer=MemorySaver())