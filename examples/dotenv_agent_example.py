"""
An iterative agent example for the Agentis MCP framework using dotenv for secret management.

This example demonstrates how to use environment variables from a .env file 
for API keys and other secrets, improving security by keeping sensitive data
out of the codebase.
"""

import asyncio
import os
import time
from typing import Dict, Any

# Add the parent directory to the Python path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agentis_mcp.app import AgentisApp
from src.agentis_mcp.agents.iterative_agent import IterativeAgent
from src.agentis_mcp.agents import LLMConfig
from src.agentis_mcp.config.settings import Settings, MCPSettings, MCPServerSettings
from src.agentis_mcp.utils.secrets import get_api_key, get_secret, generate_env_template

# Get API key from environment variables or .env file
ANTHROPIC_API_KEY = get_api_key("anthropic")

# If the API key isn't set, generate a template and inform the user
if not ANTHROPIC_API_KEY:
    print("WARNING: ANTHROPIC_API_KEY not found in environment or .env file.")
    generate_env_template()
    print("A .env.example file has been created. Rename it to .env and add your API keys.")
    print("This example will run, but API calls will fail without a valid key.")


async def handle_human_input(request: Dict[str, Any]) -> str:
    """
    Handle human input requests from the agent.
    
    Args:
        request: Input request with prompt and metadata.
        
    Returns:
        User's input as a string.
    """
    prompt = request.get("prompt", "Please provide input:")
    print(f"\n[Claude needs your input] {prompt}")
    return input("> ")


async def main():
    """Main entry point for the iterative agent example."""
    
    # Create settings with filesystem, fetch, brave search, and playwright servers
    settings = Settings(
        mcp=MCPSettings(
            servers={
                "filesystem": MCPServerSettings(
                    command="npx",
                    args=["-y", "@modelcontextprotocol/server-filesystem", os.getcwd()],
                ),
                "fetch": MCPServerSettings(
                    command="uvx",
                    args=["mcp-server-fetch"],
                ),
                "brave-search": MCPServerSettings(
                    command="npx",
                    args=["-y", "@modelcontextprotocol/server-brave-search"],
                    env={"BRAVE_API_KEY": get_secret("BRAVE_API_KEY", "")}
                ),
                "playwright": MCPServerSettings(
                    command="npx",
                    args=[
                        "-y", 
                        "@executeautomation/playwright-mcp-server", 
                        "--keep-browser-open", 
                        "--timeout", "300000",
                        "--session-timeout", "600000",
                        "--debug"
                    ],
                    env={"NODE_ENV": "development", "DEBUG": "playwright*,mcp*"},
                ),
            }
        ),
    )
    
    # Set the Anthropic API key if found
    if settings.anthropic is None:
        from agentis_mcp.config.settings import AnthropicSettings
        settings.anthropic = AnthropicSettings()
    settings.anthropic.api_key = ANTHROPIC_API_KEY
    
    # Create the app
    app = AgentisApp(
        name="dotenv_agent_example",
        settings=settings,
        human_input_handler=handle_human_input,
    )
    
    # Run the app
    async with app.run() as app_context:
        # Create an iterative agent
        agent = IterativeAgent(
            context=app_context.context,
            agent_name="iterative_claude",
            server_names=["filesystem", "fetch", "brave-search", "playwright"],
            connection_persistence=True,  # Keep MCP server connections open
            instruction=(
                "You are Claude, an AI assistant with iterative reasoning capabilities. "
                "You can break down complex tasks into steps and use tools to gather information."
            ),
            llm_config=LLMConfig(
                provider="anthropic",
                model="claude-3-7-sonnet-20250219",  # Claude 3.7 Sonnet
                api_key=ANTHROPIC_API_KEY,
                temperature=0.7,
                max_tokens=4000,
                system_prompt=(
                    "You are Claude, an AI assistant that can perform iterative reasoning. "
                    "When given complex queries, break them down into steps and use available tools to gather information. "
                    "Think step-by-step and explain your reasoning. "
                    "You have access to these tools:"
                    "\n- Use filesystem tools to explore, read, and manipulate files"
                    "\n- Use fetch tools to retrieve information from URLs"
                    "\n- Use the brave-search-search tool to search the web. This tool takes a 'query' parameter with your search terms."
                    "\n- Use playwright tools to automate web browser interactions and extract content from web pages."
                    "\nWhen searching the web for current information like prices or news, use brave-search-search with a query parameter."
                    "\nExample: brave-search-search(query='bitcoin price')"
                    "\nWhen you need to interact with web pages or extract content from them, use playwright tools."
                    "\nFor playwright interactions, remember to first navigate to a page with playwright_navigate, "
                    "\nthen extract content with playwright_get_inner_text, playwright_extract or playwright_screenshot, "
                    "\nand interact with the page using playwright_click, playwright_fill, etc."
                    "\nThe current working directory is: " + os.getcwd() + "\n"
                    "\nAlways try to use these tools rather than asking the human for information that you can get yourself."
                ),
            ),
            max_iterations=10,
            max_tool_calls_per_iteration=5,
            stop_on_success=True,
        )
        
        print("\n=== Iterative Claude 3.7 Sonnet with MCP Tools (using dotenv) ===")
        print("Type 'exit' or 'quit' to end the session.")
        print("Try asking complex questions that require multiple steps to answer.")
        
        # Start the conversation
        async with agent:
            # List available tools
            try:
                tools = await agent.list_tools()
                print("\nAvailable tools:")
                for tool in tools:
                    if isinstance(tool, dict):
                        print(f"- {tool.get('name')}: {tool.get('description')}")
                    else:
                        print(f"- {tool}")
            except Exception as e:
                print(f"Error listing tools: {e}")
            
            print("\nClaude 3.7: I'm Claude, an AI assistant with iterative reasoning capabilities. I can break down complex tasks into steps and use tools to gather information. What would you like me to help you with today?")
            
            # Chat loop
            while True:
                # Get user input
                user_input = input("\nYou: ")
                
                # Check if user wants to exit
                if user_input.lower() in ["exit", "quit", "bye"]:
                    print("\nClaude 3.7: Goodbye! It was nice helping you.")
                    break
                
                # Get response from the agent
                try:
                    print("\nThinking and working through steps...\n")
                    start_time = time.time()
                    
                    # Use iterative reasoning to process the query
                    response = await agent.run_iterative(user_input)
                    
                    end_time = time.time()
                    
                    # Print response with timing information
                    print(f"\nClaude 3.7 ({(end_time - start_time):.2f}s): {response}")
                except Exception as e:
                    print(f"\nError: {e}")
                    if "api_key" in str(e).lower():
                        print("Please check your Anthropic API key in the .env file.")
                    break


if __name__ == "__main__":
    asyncio.run(main())