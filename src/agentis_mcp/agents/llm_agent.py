"""
LLM Agent implementation for Agentis MCP framework.
"""

from typing import Dict, List, Any, Optional, Union, Callable

from agentis_mcp.agents.agent import Agent, LLMConfig
from agentis_mcp.core.context import AgentContext

class LLMAgent(Agent):
    """
    LLM-powered agent for the Agentis MCP framework.
    
    A subclass of Agent that provides enhanced LLM-specific functionality.
    """
    
    def __init__(
        self,
        context: AgentContext,
        agent_name: Optional[str] = None,
        server_names: Optional[List[str]] = None,
        connection_persistence: Optional[bool] = None,
        instruction: Optional[Union[str, Callable[[Dict], str]]] = None,
        functions: Optional[List[Callable]] = None,
        human_input_callback: Optional[Callable] = None,
        llm_config: Optional[LLMConfig] = None,
    ):
        """
        Initialize an LLM agent.
        
        Args:
            context: Agent context with configuration and server registry.
            agent_name: Name of this agent, used for configuration lookup.
            server_names: List of server names to connect to.
            connection_persistence: Whether to maintain persistent connections.
            instruction: Agent instruction or callable that returns instruction.
            functions: List of local functions to expose as tools.
            human_input_callback: Callback for handling human input requests.
            llm_config: Configuration for the LLM.
        """
        super().__init__(
            context=context,
            agent_name=agent_name,
            server_names=server_names,
            connection_persistence=connection_persistence,
            instruction=instruction,
            functions=functions,
            human_input_callback=human_input_callback,
            llm_config=llm_config,
        )
    
    async def summarize_conversation(self) -> str:
        """
        Generate a summary of the current conversation.
        
        Returns:
            Summary of the conversation.
        """
        # Create a prompt for summarization
        messages = self.messages.copy()
        
        # Add a system instruction for summarization
        summarize_instruction = {
            "role": "system",
            "content": "You are a helpful assistant tasked with summarizing the conversation so far. "
                      "Please provide a concise summary of the key points and topics discussed."
        }
        
        # Create a temporary messages list with the summarize instruction
        temp_messages = [summarize_instruction] + messages
        
        # Generate summary
        try:
            summary = await self.llm_provider.generate(
                messages=temp_messages,
                model=self.llm_config.model,
                temperature=0.3,  # Lower temperature for more focused response
                max_tokens=200,   # Limit token count for summary
            )
            return summary
        except Exception as e:
            self.logger.error(f"Error generating conversation summary: {e}")
            return "Error generating summary"
    
    async def analyze_tool_usage(self) -> Dict[str, Any]:
        """
        Analyze and report on tool usage in the current session.
        
        Returns:
            Analysis report with counts and patterns.
        """
        # Count total tools used
        total_count = len(self.tool_call_history)
        
        # Count by tool type
        tool_counts = {}
        for call in self.tool_call_history:
            tool_name = call.get("tool", "unknown")
            if tool_name in tool_counts:
                tool_counts[tool_name] += 1
            else:
                tool_counts[tool_name] = 1
        
        # Count successful vs failed calls
        success_count = sum(1 for call in self.tool_call_history if "error" not in call)
        failed_count = total_count - success_count
        
        return {
            "total_calls": total_count,
            "success_count": success_count,
            "failed_count": failed_count,
            "tool_counts": tool_counts,
            "success_rate": success_count / total_count if total_count > 0 else 0
        }