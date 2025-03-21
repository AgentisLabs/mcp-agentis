"""
MCP-enabled chat agent example for the Agentis MCP framework.

This example creates a chat agent that can use tools from the filesystem, fetch, web search,
and playwright MCP servers, providing the agent with capabilities to interact with files,
fetch data, search the web, and automate browser interactions.
Uses Claude 3.5 Sonnet as the LLM.
"""

import asyncio
import os
import time
from typing import Dict, Any

from agentis_mcp.app import AgentisApp
from agentis_mcp.agents import Agent, LLMConfig
from agentis_mcp.config.settings import Settings, MCPSettings, MCPServerSettings

# Set your API key via environment variable or .env file
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
if not ANTHROPIC_API_KEY:
    print("Warning: Anthropic API key not set. Please set ANTHROPIC_API_KEY environment variable or use a .env file.")
else:
    print("Using Anthropic API key from environment.")


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
    """Main entry point for the MCP chat agent example."""
    
    # Create settings with filesystem, fetch, browser, and playwright servers
    # This mirrors the approach from the old framework
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
                    env={"BRAVE_API_KEY": os.environ.get("BRAVE_API_KEY", "")}
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
    
    # Set the Anthropic API key
    if settings.anthropic is None:
        from agentis_mcp.config.settings import AnthropicSettings
        settings.anthropic = AnthropicSettings()
    settings.anthropic.api_key = ANTHROPIC_API_KEY
    
    # Create the app with explicit settings
    app = AgentisApp(
        name="mcp_chat_agent",
        settings=settings,
        human_input_handler=handle_human_input,
    )
    
    # Run the app
    async with app.run() as mcp_app:
        # Create an MCP-enabled agent
        agent = Agent(
            context=mcp_app.context,
            agent_name="claude_mcp_assistant",
            server_names=["filesystem", "fetch", "brave-search", "playwright"],
            connection_persistence=True,  # Keep connections open for multiple tool calls
            instruction=(
                "You are Claude, a helpful AI assistant with access to filesystem, fetch, web search, and browser automation tools. "
                "You can help users with file-related tasks, fetching information from URLs, searching the web, and interacting with web pages. "
                "When asked about files, directories, or current information, use your tools to provide accurate information."
            ),
            llm_config=LLMConfig(
                provider="anthropic",
                model="claude-3-5-sonnet-20240620",
                api_key=ANTHROPIC_API_KEY,
                temperature=0.7,
                max_tokens=4000,
                system_prompt=(
                    "You are Claude, a helpful AI assistant created by Anthropic with access to filesystem, fetch, web search, and browser automation tools. "
                    "When asked about files or directories, use your filesystem tools to provide accurate information. "
                    "When asked to fetch data from URLs, use your fetch tools. "
                    "When asked for current information or to search the web, use the brave-search-search tool to search the internet. "
                    "To use the web search tool: brave-search-search(query='your search terms')"
                    "When you need to interact with web pages, extract content, or automate browser tasks, use playwright tools. "
                    "\nFor playwright interactions, follow this sequence:"
                    "\n1. First navigate to a page with playwright_navigate(url='https://example.com')"
                    "\n2. Extract content with playwright_get_inner_text, playwright_extract or playwright_screenshot"
                    "\n3. Interact with the page using playwright_click, playwright_fill, etc."
                    "\n4. Continue with additional extractions or interactions as needed"
                    "\nIMPORTANT: Always make multiple playwright tool calls in sequence to complete tasks. Do not stop after navigation."
                    "\nThe current working directory is: " + os.getcwd() + "\n"
                    "Always try to use these tools rather than asking the human for information that you can get yourself."
                ),
            ),
        )
        
        print("\n=== Claude 3.5 Sonnet with MCP Tools (Filesystem, Fetch, Web Search & Playwright) ===")
        print("Type 'exit' or 'quit' to end the session.")
        print("Try asking about files, listing directories, searching the web, fetching data from URLs, or automating browser tasks.")
        
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
            
            print("\nClaude: I'm Claude, an AI assistant created by Anthropic with access to filesystem, fetch, web search, and browser automation tools. I can help you with file operations, fetching data from URLs, searching the web, and interacting with web pages. What would you like to do?")
            
            # Chat loop
            while True:
                # Get user input
                user_input = input("\nYou: ")
                
                # Check if user wants to exit
                if user_input.lower() in ["exit", "quit", "bye"]:
                    print("\nClaude: Goodbye! It was nice helping you.")
                    break
                
                # Get response from the agent
                try:
                    start_time = time.time()
                    response = await agent.run(user_input)
                    end_time = time.time()
                    
                    # Print response with timing information
                    print(f"\nClaude ({(end_time - start_time):.2f}s): {response}")
                except Exception as e:
                    print(f"\nError: {e}")
                    if "api_key" in str(e).lower():
                        print("Please check your Anthropic API key.")
                    break


if __name__ == "__main__":
    asyncio.run(main())