"""
LLM-powered agent example for Agentis MCP framework.
"""

import asyncio
import os
import sys
from typing import Dict, Any

# Add the package directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from agentis_mcp import AgentContext, load_config
from agentis_mcp.agents import LLMAgent, LLMConfig


async def main():
    """Run the LLM agent example."""
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), "agentis_mcp.config.yaml")
    config = load_config(config_path)
    
    # Check for secrets file and load if it exists
    secrets_path = os.path.join(os.path.dirname(__file__), "agentis_mcp.secrets.yaml")
    if os.path.exists(secrets_path):
        secrets_config = load_config(secrets_path)
        # Manually merge the API key from secrets if available
        if secrets_config.agents.get("llm_agent", {}).get("llm", {}).get("api_key"):
            config.agents["llm_agent"]["llm"]["api_key"] = secrets_config.agents["llm_agent"]["llm"]["api_key"]
    
    # Create context
    context = AgentContext(config, agent_name="llm_agent")
    
    # Optional: Override LLM configuration from config file
    # If you want to use a specific configuration for this run only
    llm_config = None
    
    # For example, to use the mock provider for testing:
    # llm_config = LLMConfig(
    #     provider="mock",
    #     model="mock-model",
    #     temperature=0.7,
    #     system_prompt="You are a helpful assistant with access to various tools."
    # )
    
    # Create and run agent (will use config file settings by default)
    async with LLMAgent(
        context=context,
        llm_config=llm_config  # Pass None to use config file settings
    ) as agent:
        # Process a query
        user_query = "What's the weather in San Francisco, and can you also tell me a fun fact about the city?"
        print(f"\nUser query: {user_query}\n")
        
        result = await agent.run(user_query, max_iterations=5)
        print("\nAgent response:")
        print(result)
        
        # Process another query to demonstrate conversation context
        if input("\nSend follow-up query? (y/n): ").lower() == "y":
            user_query = "What's another interesting fact about San Francisco?"
            print(f"\nUser query: {user_query}\n")
            
            result = await agent.run(user_query, max_iterations=5)
            print("\nAgent response:")
            print(result)


if __name__ == "__main__":
    asyncio.run(main())