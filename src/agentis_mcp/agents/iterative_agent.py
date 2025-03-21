"""
Advanced agent capabilities with iterative reasoning loops.

This module extends the base Agent class with advanced features like:
- Iterative reasoning loops
- Memory management
- Structured output parsing
- Error handling with fallbacks
"""

import asyncio
import json
import uuid
from typing import Dict, List, Optional, Any, Union, Callable, Tuple

from agentis_mcp.agents.agent import Agent, LLMConfig
from agentis_mcp.utils.logging import Logger, get_logger

logger = get_logger(__name__)


class IterativeAgent(Agent):
    """
    An agent that can run in iterative loops to solve complex tasks.
    
    This agent extends the base Agent with the ability to:
    - Run multiple reasoning steps
    - Maintain memory between steps
    - Handle errors with fallbacks
    - Parse structured output
    """
    
    def __init__(
        self,
        context,
        agent_name: str,
        instruction: str,
        llm_config: LLMConfig,
        server_names: Optional[List[str]] = None,
        connection_persistence: bool = True,  # Default to True for iterative agents
        max_iterations: int = 5,
        max_tool_calls_per_iteration: int = 5,
        stop_on_success: bool = True,
        memory: Optional[Dict] = None,
        human_input_callback: Optional[Callable] = None,
        *args,
        **kwargs
    ):
        """
        Initialize an iterative agent.
        
        Args:
            context: The agent context.
            agent_name: Name of this agent.
            instruction: Instruction for the agent.
            llm_config: LLM configuration.
            server_names: MCP server names to connect to.
            connection_persistence: Whether to maintain persistent connections to MCP servers.
            max_iterations: Maximum number of reasoning iterations.
            max_tool_calls_per_iteration: Maximum number of tool calls per iteration.
            stop_on_success: Whether to stop iterating when a successful response is found.
            memory: Initial memory for the agent.
            human_input_callback: Callback for requesting human input.
        """
        super().__init__(
            context=context,
            agent_name=agent_name,
            instruction=instruction,
            llm_config=llm_config,
            server_names=server_names,
            connection_persistence=connection_persistence,
            human_input_callback=human_input_callback,
            *args,
            **kwargs
        )
        
        # Initialize LLM from config
        from agentis_mcp.agents.agent import create_llm_provider
        self.llm = create_llm_provider(llm_config)
        
        # Make sure API key is set
        if hasattr(self.llm, "provider") and self.llm.provider == "anthropic":
            # Handle the case where the API key might be in the context config
            if not getattr(self.llm, "api_key", None) and context and hasattr(context, "config"):
                if hasattr(context.config, "anthropic") and context.config.anthropic:
                    self.llm.api_key = context.config.anthropic.api_key
        
        self.max_iterations = max_iterations
        self.max_tool_calls_per_iteration = max_tool_calls_per_iteration
        self.stop_on_success = stop_on_success
        self.memory = memory or {"conversation": [], "tool_results": {}}
        self.logger = Logger(f"{__name__}.{agent_name}")
    
    async def add_to_memory(self, key: str, value: Any) -> None:
        """
        Add information to the agent's memory.
        
        Args:
            key: Memory key.
            value: Value to store.
        """
        self.memory[key] = value
    
    async def get_from_memory(self, key: str, default: Any = None) -> Any:
        """
        Get information from the agent's memory.
        
        Args:
            key: Memory key.
            default: Default value if key doesn't exist.
            
        Returns:
            The stored value or default.
        """
        return self.memory.get(key, default)
    
    async def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the conversation history.
        
        Args:
            role: Message role (user, assistant, system).
            content: Message content.
        """
        await super().add_message(role, content)
        
        # Also store in memory
        if "conversation" not in self.memory:
            self.memory["conversation"] = []
        
        self.memory["conversation"].append({"role": role, "content": content})
    
    async def run_iterative(
        self,
        query: str,
        max_iterations: Optional[int] = None,
        system_prompt_override: Optional[str] = None,
    ) -> str:
        """
        Run the agent in an iterative reasoning loop.
        
        Args:
            query: The user query to process.
            max_iterations: Maximum number of reasoning iterations (overrides instance setting).
            system_prompt_override: Override the system prompt for this run.
            
        Returns:
            The final response from the agent.
        """
        if not self.initialized:
            await self.__aenter__()
        
        # Add the query to the conversation history
        await self.add_message("user", query)
        
        # Initialize the iteration state
        max_iters = max_iterations or self.max_iterations
        current_iteration = 1
        
        # Initialize the scratchpad for chain-of-thought reasoning
        scratchpad = []
        success = False
        final_response = "I couldn't complete the task successfully."
        
        # Initialize LLM if needed
        if not hasattr(self, 'llm') or self.llm is None:
            from agentis_mcp.agents.agent import create_llm_provider
            if hasattr(self, 'llm_config') and self.llm_config:
                self.llm = create_llm_provider(self.llm_config)
            
        # Store original system prompt to restore it later if needed
        original_system_prompt = None
        if system_prompt_override and hasattr(self, 'llm') and self.llm:
            original_system_prompt = getattr(self.llm, 'system_prompt', None)
            if hasattr(self.llm, 'system_prompt'):
                self.llm.system_prompt = system_prompt_override
        
        try:
            # Main iteration loop
            while current_iteration <= max_iters and not success:
                self.logger.info(f"Starting iteration {current_iteration}/{max_iters}")
                
                # Create the combined prompt with scratchpad
                scratchpad_text = "\n\n".join(scratchpad) if scratchpad else ""
                iteration_prompt = (
                    f"Iteration {current_iteration}/{max_iters}\n\n"
                    f"User query: {query}\n\n"
                    f"Scratchpad from previous iterations:\n{scratchpad_text}\n\n"
                    f"In this iteration, think about what information you need to answer "
                    f"the query and what tools you should use. Implement a plan to answer "
                    f"the user's query. You can use available tools. "
                    f"If you have enough information to give a final answer, say FINAL ANSWER: "
                    f"followed by your response to the user."
                )
                
                # Get the LLM response for this iteration
                try:
                    # Ensure API key is set before each call
                    if self.llm and hasattr(self.llm, "provider") and self.llm.provider == "anthropic":
                        # Make sure API key is set
                        if not getattr(self.llm, "api_key", None) and self.context and self.context.config:
                            if hasattr(self.context.config, "anthropic") and self.context.config.anthropic:
                                self.llm.api_key = self.context.config.anthropic.api_key
                                
                    response = await self.run(iteration_prompt)
                    
                    # Record the iteration in the scratchpad
                    scratchpad.append(f"Iteration {current_iteration}:\n{response}")
                    
                    # Check if we have a final answer
                    if "FINAL ANSWER:" in response:
                        final_answer_parts = response.split("FINAL ANSWER:", 1)
                        if len(final_answer_parts) > 1:
                            final_response = final_answer_parts[1].strip()
                            success = True
                            self.logger.info(
                                f"Found final answer in iteration {current_iteration}",
                                data={"final_answer": final_response}
                            )
                    
                    # If we're successful and configured to stop, exit the loop
                    if success and self.stop_on_success:
                        break
                    
                except Exception as e:
                    self.logger.error(
                        f"Error in iteration {current_iteration}",
                        data={"error": str(e)}
                    )
                    scratchpad.append(
                        f"Iteration {current_iteration} ERROR:\n{str(e)}\n"
                        "Continuing to next iteration..."
                    )
                
                current_iteration += 1
            
            # If we didn't get a final answer after all iterations, generate one
            if not success:
                self.logger.info(
                    f"No final answer found after {max_iters} iterations, generating summary."
                )
                
                summary_prompt = (
                    f"You have been working on this query through multiple iterations: {query}\n\n"
                    f"Here is your work so far:\n{scratchpad_text}\n\n"
                    f"Although you haven't explicitly provided a FINAL ANSWER, "
                    f"please summarize what you've learned and provide the best possible "
                    f"response to the user based on the information gathered so far."
                )
                
                try:
                    # Initialize LLM if needed
                    if not hasattr(self, 'llm') or self.llm is None:
                        from agentis_mcp.agents.agent import create_llm_provider
                        if hasattr(self, 'llm_config') and self.llm_config:
                            self.llm = create_llm_provider(self.llm_config)
                                
                    # Ensure API key is set before the call
                    if hasattr(self, 'llm') and self.llm and hasattr(self.llm, "provider") and self.llm.provider == "anthropic":
                        # Make sure API key is set
                        if not getattr(self.llm, "api_key", None) and self.context and hasattr(self.context, "config"):
                            if hasattr(self.context.config, "anthropic") and self.context.config.anthropic:
                                self.llm.api_key = self.context.config.anthropic.api_key
                                
                    final_response = await self.run(summary_prompt)
                except Exception as e:
                    self.logger.error(
                        "Error generating summary response",
                        data={"error": str(e)}
                    )
                    final_response = (
                        "I've been working on your query but encountered some issues. "
                        "Based on what I've found so far: " + 
                        "\n\n".join(scratchpad[-2:] if len(scratchpad) >= 2 else scratchpad)
                    )
        
        finally:
            # Restore the original system prompt if we changed it
            if original_system_prompt and hasattr(self, 'llm') and self.llm and hasattr(self.llm, 'system_prompt'):
                self.llm.system_prompt = original_system_prompt
        
        # Add the final response to the conversation history
        await self.add_message("assistant", final_response)
        
        return final_response
    
    async def run_with_fallbacks(
        self,
        query: str,
        fallback_prompts: List[str] = None,
        max_attempts: int = 3,
    ) -> str:
        """
        Run the agent with fallback mechanisms for error handling.
        
        Args:
            query: The user query to process.
            fallback_prompts: Alternative prompts to try if the main one fails.
            max_attempts: Maximum number of attempts before giving up.
            
        Returns:
            The response from the agent, or a fallback response if all attempts fail.
        """
        if not fallback_prompts:
            fallback_prompts = [
                "Let's try a different approach to answer this question: {query}",
                "Please provide a simple, direct answer to: {query}",
                "What's the most basic information you can provide about: {query}"
            ]
        
        # Try the main query first
        try:
            # Initialize LLM if needed
            if not hasattr(self, 'llm') or self.llm is None:
                from agentis_mcp.agents.agent import create_llm_provider
                if hasattr(self, 'llm_config') and self.llm_config:
                    self.llm = create_llm_provider(self.llm_config)
                        
            # Ensure API key is set before the call
            if hasattr(self, 'llm') and self.llm and hasattr(self.llm, "provider") and self.llm.provider == "anthropic":
                # Make sure API key is set
                if not getattr(self.llm, "api_key", None) and self.context and hasattr(self.context, "config"):
                    if hasattr(self.context.config, "anthropic") and self.context.config.anthropic:
                        self.llm.api_key = self.context.config.anthropic.api_key
                        
            return await self.run_iterative(query)
        except Exception as e:
            self.logger.warning(
                f"Error in main query execution, trying fallbacks",
                data={"error": str(e)}
            )
        
        # If main query fails, try fallbacks
        for i, fallback_template in enumerate(fallback_prompts[:max_attempts-1]):
            try:
                fallback_query = fallback_template.format(query=query)
                self.logger.info(f"Trying fallback {i+1}/{len(fallback_prompts)}")
                
                # Initialize LLM if needed
                if not hasattr(self, 'llm') or self.llm is None:
                    from agentis_mcp.agents.agent import create_llm_provider
                    if hasattr(self, 'llm_config') and self.llm_config:
                        self.llm = create_llm_provider(self.llm_config)
                            
                # Ensure API key is set before the call
                if hasattr(self, 'llm') and self.llm and hasattr(self.llm, "provider") and self.llm.provider == "anthropic":
                    # Make sure API key is set
                    if not getattr(self.llm, "api_key", None) and self.context and hasattr(self.context, "config"):
                        if hasattr(self.context.config, "anthropic") and self.context.config.anthropic:
                            self.llm.api_key = self.context.config.anthropic.api_key
                            
                return await self.run(fallback_query)
            except Exception as e:
                self.logger.warning(
                    f"Fallback {i+1} failed",
                    data={"error": str(e)}
                )
        
        # If all fallbacks fail, return a graceful error message
        return (
            "I'm having trouble processing your request right now. "
            "Could you please rephrase or try a different question?"
        )
    
    async def extract_structured_output(self, query: str, output_format: Dict) -> Dict:
        """
        Extract structured data from a query response.
        
        Args:
            query: The user query to process.
            output_format: JSON schema defining the expected output structure.
            
        Returns:
            Structured data extracted from the response.
        """
        schema_str = json.dumps(output_format, indent=2)
        
        structured_prompt = (
            f"Please answer the following query and format your response exactly according "
            f"to this JSON schema:\n\n{schema_str}\n\n"
            f"Query: {query}\n\n"
            f"Respond only with valid JSON that matches the schema."
        )
        
        for attempt in range(3):  # Try up to 3 times to get valid JSON
            try:
                # Initialize LLM if needed
                if not hasattr(self, 'llm') or self.llm is None:
                    from agentis_mcp.agents.agent import create_llm_provider
                    if hasattr(self, 'llm_config') and self.llm_config:
                        self.llm = create_llm_provider(self.llm_config)
                            
                # Ensure API key is set before each call
                if hasattr(self, 'llm') and self.llm and hasattr(self.llm, "provider") and self.llm.provider == "anthropic":
                    # Make sure API key is set
                    if not getattr(self.llm, "api_key", None) and self.context and hasattr(self.context, "config"):
                        if hasattr(self.context.config, "anthropic") and self.context.config.anthropic:
                            self.llm.api_key = self.context.config.anthropic.api_key
                            
                response = await self.run(structured_prompt)
                
                # Try to extract JSON from the response
                match = False
                
                # Look for JSON block in markdown format
                if "```json" in response and "```" in response.split("```json", 1)[1]:
                    json_str = response.split("```json", 1)[1].split("```", 1)[0].strip()
                    match = True
                # Look for JSON block with just the backticks
                elif "```" in response and "```" in response.split("```", 1)[1]:
                    json_str = response.split("```", 1)[1].split("```", 1)[0].strip()
                    match = True
                # Check if the entire response is JSON
                elif response.strip().startswith("{") and response.strip().endswith("}"):
                    json_str = response.strip()
                    match = True
                
                if match:
                    parsed_json = json.loads(json_str)
                    return parsed_json
                
                # If we can't parse the response as JSON, try again with a clearer prompt
                structured_prompt = (
                    f"Your previous response couldn't be parsed as JSON. "
                    f"Please reply with ONLY the JSON object, strictly following this schema:\n\n"
                    f"{schema_str}\n\n"
                    f"Query: {query}\n\n"
                    f"Do not include any explanations, markdown formatting, or backticks."
                )
                
            except json.JSONDecodeError:
                self.logger.warning(f"Failed to parse JSON on attempt {attempt+1}/3")
                
                if attempt == 2:  # Last attempt
                    # Try one last time with an even stricter prompt
                    structured_prompt = (
                        f"I'm having trouble parsing your response as JSON. "
                        f"Please provide ONLY a JSON object like this example:\n\n"
                        f"{json.dumps(self._create_example_from_schema(output_format), indent=2)}\n\n"
                        f"but with information that answers: {query}"
                    )
            
            except Exception as e:
                self.logger.error(
                    "Error in structured output extraction",
                    data={"error": str(e), "attempt": attempt+1}
                )
        
        # If all attempts fail, return a best-effort parsed response
        try:
            # Try to extract any JSON-like structure from the response
            # Initialize LLM if needed
            if not hasattr(self, 'llm') or self.llm is None:
                from agentis_mcp.agents.agent import create_llm_provider
                if hasattr(self, 'llm_config') and self.llm_config:
                    self.llm = create_llm_provider(self.llm_config)
                        
            # Ensure API key is set before the call
            if hasattr(self, 'llm') and self.llm and hasattr(self.llm, "provider") and self.llm.provider == "anthropic":
                # Make sure API key is set
                if not getattr(self.llm, "api_key", None) and self.context and hasattr(self.context, "config"):
                    if hasattr(self.context.config, "anthropic") and self.context.config.anthropic:
                        self.llm.api_key = self.context.config.anthropic.api_key
                        
            response = await self.run(
                f"Based on the query '{query}', create a JSON object that follows this schema: {schema_str}. "
                f"Respond ONLY with the JSON object, no explanation or formatting."
            )
            
            # Extract anything that looks like JSON
            import re
            json_match = re.search(r'(\{.*\})', response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
            
            # Last resort - create an empty structure based on the schema
            return self._create_empty_from_schema(output_format)
            
        except Exception as e:
            self.logger.error(
                "Failed to extract structured output after multiple attempts",
                data={"error": str(e)}
            )
            # Return an empty structure based on the schema
            return self._create_empty_from_schema(output_format)
    
    def _create_example_from_schema(self, schema: Dict) -> Dict:
        """Create an example object from a JSON schema."""
        result = {}
        
        properties = schema.get("properties", {})
        for key, prop in properties.items():
            if prop.get("type") == "string":
                result[key] = f"Example {key}"
            elif prop.get("type") == "number" or prop.get("type") == "integer":
                result[key] = 42
            elif prop.get("type") == "boolean":
                result[key] = True
            elif prop.get("type") == "array":
                if "items" in prop and "properties" in prop["items"]:
                    result[key] = [self._create_example_from_schema(prop["items"])]
                else:
                    result[key] = ["example item"]
            elif prop.get("type") == "object":
                result[key] = self._create_example_from_schema(prop)
            else:
                result[key] = None
                
        return result
    
    def _create_empty_from_schema(self, schema: Dict) -> Dict:
        """Create an empty structure based on a JSON schema."""
        result = {}
        
        properties = schema.get("properties", {})
        for key, prop in properties.items():
            if prop.get("type") == "array":
                result[key] = []
            elif prop.get("type") == "object":
                result[key] = self._create_empty_from_schema(prop)
            else:
                result[key] = None
                
        return result