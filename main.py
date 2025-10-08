"""
This serves our integrated multi-agent system through a FastAPI server.
Includes: Post Generator, GitHub Stack Analyzer, and Expert Copywriter agents.
"""

import os
from dotenv import load_dotenv

load_dotenv()  

from fastapi import FastAPI
import uvicorn
from copilotkit.integrations.fastapi import add_fastapi_endpoint
from copilotkit import CopilotKitSDK, LangGraphAgent
from posts_generator_agent import post_generation_graph
from stack_agent import stack_analysis_graph
from copywriter_agent import copywriter_agent_graph

app = FastAPI(
    title="Multi-Agent AI System",
    description="Integrated system with Post Generation, GitHub Analysis, and Expert Copywriting agents",
    version="2.0.0"
)


# Initialize CopilotKit SDK with all three agents
sdk = CopilotKitSDK(
    agents=[
        LangGraphAgent(
            name="post_generation_agent",
            description="An expert agent for generating engaging LinkedIn posts and X (Twitter) posts. Performs web research to gather trending topics and creates platform-optimized social media content with proper formatting, emojis, and hashtags.",
            graph=post_generation_graph,
        ),
        LangGraphAgent(
            name="stack_analysis_agent",
            description="A senior software architect agent that analyzes GitHub repository URLs to infer the project's purpose, architecture, and complete tech stack including frontend frameworks, backend technologies, databases, infrastructure, and deployment strategies.",
            graph=stack_analysis_graph,
        ),
        LangGraphAgent(
            name="copywriter_agent",
            description="An elite master copywriter with 20+ years of experience. Creates high-converting sales copy, landing pages, ad copy, email campaigns, and marketing content. Uses persuasion frameworks (AIDA, PAS, FAB), conducts market research, develops strategic briefs, and generates A/B test variations for optimal conversion rates.",
            graph=copywriter_agent_graph,
        ),
    ]
)

# Add CopilotKit endpoint
add_fastapi_endpoint(app, sdk, "/copilotkit")


# ==================== API ENDPOINTS ====================

@app.get("/healthz")
def health():
    """Health check endpoint."""
    return {"status": "ok", "message": "All agents operational"}


@app.get("/")
def root():
    """Root endpoint with system information."""
    return {
        "message": "Multi-Agent AI System",
        "version": "2.0.0",
        "agents": [
            {
                "name": "Post Generation Agent",
                "capabilities": ["LinkedIn posts", "Twitter/X posts", "Social media research"]
            },
            {
                "name": "Stack Analysis Agent",
                "capabilities": ["GitHub repo analysis", "Tech stack detection", "Architecture review"]
            },
            {
                "name": "Copywriter Agent",
                "capabilities": ["Sales copy", "Ad copy", "Landing pages", "Email marketing", "A/B testing"]
            }
        ],
        "docs": "/docs"
    }


@app.get("/docs-info")
def docs_info():
    """Helpful message for testing docs and endpoints."""
    return {
        "message": "Swagger UI available at /docs",
        "endpoints": {
            "health": "/healthz",
            "root": "/",
            "copilotkit": "/copilotkit",
            "agents": "/agents",
            "docs": "/docs"
        },
        "agent_count": 3
    }


@app.get("/agents")
def list_agents():
    """List all available agents with their capabilities."""
    return {
        "total_agents": 3,
        "agents": [
            {
                "id": "post_generation_agent",
                "name": "Post Generation Agent",
                "description": "Generates social media posts with web research",
                "platforms": ["LinkedIn", "Twitter/X"],
                "features": [
                    "Web search for trending topics",
                    "Platform-optimized formatting",
                    "Emoji and hashtag integration",
                    "Engagement-focused content"
                ],
                "use_cases": [
                    "Create LinkedIn post about [topic]",
                    "Generate X post about [topic]",
                    "Write social media content for [subject]"
                ]
            },
            {
                "id": "stack_analysis_agent",
                "name": "GitHub Stack Analysis Agent",
                "description": "Analyzes GitHub repositories for architecture and tech stack",
                "capabilities": [
                    "Repository purpose identification",
                    "Frontend framework detection",
                    "Backend technology analysis",
                    "Database identification",
                    "Infrastructure assessment",
                    "Code quality evaluation"
                ],
                "use_cases": [
                    "Analyze github.com/user/repo",
                    "What tech stack does [repo] use?",
                    "Review the architecture of [GitHub URL]"
                ]
            },
            {
                "id": "copywriter_agent",
                "name": "Expert Copywriter Agent",
                "description": "Elite copywriting with persuasion psychology and conversion optimization",
                "expertise": [
                    "Sales and landing pages",
                    "Ad copy (Google, Facebook, LinkedIn)",
                    "Email marketing campaigns",
                    "Website copy",
                    "Product descriptions",
                    "Video scripts"
                ],
                "frameworks": [
                    "AIDA (Attention, Interest, Desire, Action)",
                    "PAS (Problem, Agitate, Solution)",
                    "FAB (Features, Advantages, Benefits)",
                    "BAB (Before, After, Bridge)"
                ],
                "process": [
                    "Market research and competitor analysis",
                    "Strategic framework development",
                    "Copy creation with psychological triggers",
                    "A/B test variation generation"
                ],
                "use_cases": [
                    "Write sales copy for [product]",
                    "Create landing page for [service]",
                    "Generate ad copy for [campaign]",
                    "Write email sequence for [offer]"
                ]
            }
        ]
    }


@app.get("/agent/{agent_name}")
def get_agent_info(agent_name: str):
    """Get detailed information about a specific agent."""
    agents_info = {
        "post_generation_agent": {
            "name": "Post Generation Agent",
            "status": "active",
            "description": "Generates engaging social media content with web research capabilities",
            "workflow": [
                "1. Analyze user request",
                "2. Perform web research for trending topics",
                "3. Generate platform-specific content",
                "4. Optimize with emojis and formatting"
            ],
            "platforms": ["LinkedIn", "Twitter/X"],
            "example_queries": [
                "Create a LinkedIn post about AI advancements",
                "Generate a tweet about sustainable technology",
                "Write social media content about productivity tools"
            ]
        },
        "stack_analysis_agent": {
            "name": "GitHub Stack Analysis Agent",
            "status": "active",
            "description": "Provides comprehensive analysis of GitHub repositories",
            "workflow": [
                "1. Extract repository URL",
                "2. Analyze codebase structure",
                "3. Identify technologies and frameworks",
                "4. Assess architecture patterns",
                "5. Generate detailed report"
            ],
            "analysis_areas": [
                "Frontend technologies",
                "Backend frameworks",
                "Database systems",
                "Infrastructure setup",
                "Code quality metrics"
            ],
            "example_queries": [
                "Analyze https://github.com/vercel/next.js",
                "What tech stack does this repo use: [URL]",
                "Review the architecture of [GitHub repo]"
            ]
        },
        "copywriter_agent": {
            "name": "Expert Copywriter Agent",
            "status": "active",
            "description": "Elite copywriting with 20+ years of expertise and conversion optimization",
            "workflow": [
                "1. Research phase: Market trends and audience insights",
                "2. Strategy phase: Framework selection and positioning",
                "3. Creation phase: Craft compelling copy",
                "4. Variation phase: Generate A/B test alternatives"
            ],
            "specializations": [
                "Sales pages and landing pages",
                "Ad copy (all major platforms)",
                "Email marketing sequences",
                "Website and product copy",
                "Video scripts and VSLs"
            ],
            "persuasion_techniques": [
                "Psychological triggers",
                "Emotional storytelling",
                "Social proof integration",
                "Scarcity and urgency",
                "Benefit-driven messaging"
            ],
            "example_queries": [
                "Write sales copy for a SaaS product",
                "Create Facebook ad copy for e-commerce store",
                "Generate landing page copy for course launch",
                "Write email sequence for product promotion"
            ]
        }
    }
    
    if agent_name not in agents_info:
        return {
            "error": "Agent not found",
            "available_agents": list(agents_info.keys())
        }
    
    return agents_info[agent_name]


@app.post("/test-agent")
async def test_agent(agent_name: str, query: str):
    """
    Test endpoint for direct agent invocation (useful for debugging).
    In production, use the /copilotkit endpoint instead.
    """
    return {
        "message": "This is a test endpoint. Use /copilotkit for production interactions.",
        "agent": agent_name,
        "query": query,
        "recommendation": "Connect through CopilotKit frontend for full functionality"
    }


# ==================== STARTUP/SHUTDOWN EVENTS ====================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    print("=" * 60)
    print("🚀 Multi-Agent AI System Starting Up")
    print("=" * 60)
    print("✅ Post Generation Agent - READY")
    print("✅ GitHub Stack Analysis Agent - READY")
    print("✅ Expert Copywriter Agent - READY")
    print("=" * 60)
    print(f"📡 Server running on port {os.getenv('PORT', '8000')}")
    print(f"📚 API Docs: http://localhost:{os.getenv('PORT', '8000')}/docs")
    print(f"🔗 CopilotKit Endpoint: http://localhost:{os.getenv('PORT', '8000')}/copilotkit")
    print("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("\n" + "=" * 60)
    print("👋 Multi-Agent AI System Shutting Down")
    print("=" * 60)


# ==================== MAIN RUNNER ====================

def main():
    """Run the uvicorn server."""
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    print("\n" + "=" * 60)
    print("🎯 MULTI-AGENT AI SYSTEM")
    print("=" * 60)
    print("Available Agents:")
    print("  1. 📱 Post Generation Agent")
    print("  2. 🔧 GitHub Stack Analysis Agent")
    print("  3. ✍️  Expert Copywriter Agent")
    print("=" * 60)
    print(f"Starting server on {host}:{port}")
    print("=" * 60 + "\n")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()