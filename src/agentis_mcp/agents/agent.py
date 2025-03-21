"""
Integrated Agent implementation for Agentis MCP framework.
"""

import asyncio
import json
import re
import uuid
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, Tuple

import aiohttp

from agentis_mcp.mcp.aggregator import ServerAggregator
from agentis_mcp.core.context import AgentContext
from agentis_mcp.utils.logging import get_logger

logger = get_logger(__name__)

# TypeVar for function tools and providers
T = TypeVar('T')
P = TypeVar('P', bound='LLMProvider')


def create_llm_provider(llm_config: 'LLMConfig') -> 'LLMProvider':
    """
    Create an LLM provider instance from configuration.
    
    Args:
        llm_config: LLM configuration object.
        
    Returns:
        LLM provider instance.
        
    Raises:
        ValueError: If provider is not supported.
    """
    provider = llm_config.provider.lower()
    
    if provider == "openai":
        return OpenAIProvider(
            api_key=llm_config.api_key,
            api_base=getattr(llm_config, "api_base", None),
            organization=getattr(llm_config, "organization", None)
        )
    elif provider == "anthropic":
        return AnthropicProvider(
            api_key=llm_config.api_key,
            api_base=getattr(llm_config, "api_base", None)
        )
    elif provider == "mock":
        return MockProvider()
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


class LLMProvider:
    """Base class for LLM providers."""
    
    async def generate(
        self, 
        messages: List[Dict[str, str]], 
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        **kwargs
    ) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            messages: List of conversation messages.
            model: Model name.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional model parameters.
            
        Returns:
            Generated text response.
        """
        raise NotImplementedError("Subclasses must implement generate()")
    
    @classmethod
    def create(cls: type[P], provider_type: str, api_key: str, api_base: Optional[str] = None) -> P:
        """
        Factory method to create an LLM provider based on type.
        
        Args:
            provider_type: Type of provider (openai, anthropic, etc.)
            api_key: API key for the provider.
            api_base: Optional API base URL.
            
        Returns:
            An instance of the appropriate LLM provider.
        """
        provider_map = {
            "openai": OpenAIProvider,
            "anthropic": AnthropicProvider,
            "mock": MockProvider,
        }
        
        provider_class = provider_map.get(provider_type.lower())
        if not provider_class:
            raise ValueError(f"Unsupported provider type: {provider_type}")
        
        return provider_class(api_key=api_key, api_base=api_base)


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""
    
    def __init__(self, api_key: str, api_base: Optional[str] = None):
        """
        Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key.
            api_base: Optional API base URL.
        """
        self.api_key = api_key
        self.api_base = api_base or "https://api.openai.com/v1"
    
    async def generate(
        self, 
        messages: List[Dict[str, str]], 
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate a response using OpenAI API.
        
        Args:
            messages: List of conversation messages.
            model: Model name.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional model parameters.
            
        Returns:
            Generated text response.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        
        # Add any additional parameters
        for key, value in kwargs.items():
            if value is not None:
                payload[key] = value
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ValueError(f"OpenAI API error: {response.status} - {error_text}")
                
                result = await response.json()
                
                # Check for tool calls in the response if OpenAI format returns them
                if "choices" in result and result["choices"] and "message" in result["choices"][0]:
                    message = result["choices"][0]["message"]
                    if "tool_calls" in message and message["tool_calls"]:
                        # Return structured tool call format for newer API versions
                        return json.dumps(message)
                    
                    return message["content"]
                
                raise ValueError(f"Unexpected OpenAI API response format: {result}")


class AnthropicProvider(LLMProvider):
    """Anthropic API provider."""
    
    def __init__(self, api_key: str, api_base: Optional[str] = None):
        """
        Initialize Anthropic provider.
        
        Args:
            api_key: Anthropic API key.
            api_base: Optional API base URL.
        """
        self.api_key = api_key
        self.api_base = api_base or "https://api.anthropic.com/v1"
    
    async def generate(
        self, 
        messages: List[Dict[str, str]], 
        model: str = "claude-3-opus-20240229",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate a response using Anthropic API.
        
        Args:
            messages: List of conversation messages.
            model: Model name.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional model parameters.
            
        Returns:
            Generated text response.
        """
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        # Format messages in Anthropic's expected structure
        system_prompt = None
        formatted_messages = []
        
        for message in messages:
            if message["role"] == "system":
                system_prompt = message["content"]
            else:
                formatted_messages.append({
                    "role": message["role"],
                    "content": message["content"]
                })
        
        payload = {
            "model": model,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 4096,  # Default to 4096 if not specified
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        # Add any additional parameters
        for key, value in kwargs.items():
            if value is not None and key not in ["system"]:
                payload[key] = value
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_base}/messages",
                headers=headers,
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ValueError(f"Anthropic API error: {response.status} - {error_text}")
                
                result = await response.json()
                
                # Handle Anthropic's content structure
                if "content" in result and result["content"]:
                    # Since Claude might return multiple content blocks, combine them
                    contents = []
                    for content_block in result["content"]:
                        if "text" in content_block:
                            contents.append(content_block["text"])
                    
                    return "\n".join(contents)
                
                raise ValueError(f"Unexpected Anthropic API response format: {result}")


class MockProvider(LLMProvider):
    """Mock provider for testing."""
    
    async def generate(
        self, 
        messages: List[Dict[str, str]], 
        model: str = "mock-model",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate a mock response for testing.
        
        Args:
            messages: List of conversation messages.
            model: Model name (ignored).
            temperature: Sampling temperature (ignored).
            max_tokens: Maximum tokens to generate (ignored).
            **kwargs: Additional model parameters (ignored).
            
        Returns:
            Generated text response.
        """
        # Get the last user message
        user_message = None
        for message in reversed(messages):
            if message["role"] == "user":
                user_message = message["content"]
                break
        
        if not user_message:
            return "I'm not sure what you're asking about."
        
        # Simple pattern-based responses for demonstration
        user_message_lower = user_message.lower()
        
        if "tool" in user_message_lower and ("search" in user_message_lower or "weather" in user_message_lower):
            return """I'll help you with that.

I'll use the search tool to find information about the weather:

```json
{"tool": "search_server-search", "parameters": {"query": "weather in San Francisco"}}
```"""
        
        elif "tool" in user_message_lower and "fun fact" in user_message_lower:
            return """I'll help you with that.

I'll use the fun facts tool to find interesting information:

```json
{"tool": "fun_facts_server-get_fun_fact", "parameters": {"topic": "San Francisco"}}
```"""
        
        elif "tool results" in user_message_lower:
            return """Based on the information I've gathered:

The weather in San Francisco is sunny with a high of 75°F.

Here's a fun fact about San Francisco: The Golden Gate Bridge is actually painted 'International Orange', not gold.

Is there anything else you'd like to know about San Francisco?"""
        
        return "I'll help you with that request."


class LLMConfig:
    """Configuration for LLM API calls."""
    
    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        system_prompt: Optional[str] = None,
        tool_format: str = "json_block",
        streaming: bool = False,
    ):
        """
        Initialize LLM configuration.
        
        Args:
            provider: Provider name ('openai', 'anthropic', or 'mock').
            model: Model name.
            api_key: API key.
            api_base: API base URL.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            top_p: Top-p sampling parameter.
            system_prompt: System prompt for the conversation.
            tool_format: Format for tool calls (json_block, openai_native).
            streaming: Enable streaming responses.
        """
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.system_prompt = system_prompt or self.default_system_prompt()
        self.tool_format = tool_format
        self.streaming = streaming
    
    @staticmethod
    def default_system_prompt() -> str:
        """Default system prompt for the LLM agent."""
        return """You are a helpful assistant with access to a set of tools.
When you need information or want to perform an action, use the appropriate tool.
Always think step-by-step about what tool would be most helpful to answer the user's question.
Use tools when necessary, but avoid overusing them when a simple answer would suffice.
"""

    def create_provider(self) -> LLMProvider:
        """
        Create an LLM provider instance based on configuration.
        
        Returns:
            LLM provider instance.
        """
        return LLMProvider.create(
            provider_type=self.provider,
            api_key=self.api_key,
            api_base=self.api_base
        )
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'LLMConfig':
        """
        Create an LLMConfig from a dictionary.
        
        Args:
            config_dict: Configuration dictionary.
            
        Returns:
            LLMConfig instance.
        """
        return cls(
            provider=config_dict.get("provider", "openai"),
            model=config_dict.get("model", "gpt-4o"),
            api_key=config_dict.get("api_key"),
            api_base=config_dict.get("api_base"),
            temperature=config_dict.get("temperature", 0.7),
            max_tokens=config_dict.get("max_tokens"),
            top_p=config_dict.get("top_p", 1.0),
            system_prompt=config_dict.get("system_prompt"),
            tool_format=config_dict.get("tool_format", "json_block"),
            streaming=config_dict.get("streaming", False),
        )


class Agent:
    """
    LLM-powered agent for the Agentis MCP framework.
    
    An agent connects to MCP servers, exposes local functions as tools,
    and uses an LLM to decide how to respond to queries and which tools to use.
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
        Initialize an agent.
        
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
        self.context = context
        self.agent_name = agent_name or context.agent_name
        
        # Create a session ID if not already in context
        self.session_id = context.session_id or str(uuid.uuid4())
        
        # Load agent config
        agent_config = context.get_agent_config(self.agent_name)
        
        # Use provided values or fall back to config
        self.server_names = server_names or agent_config.get("server_names", [])
        self.connection_persistence = connection_persistence if connection_persistence is not None else agent_config.get("connection_persistence", False)
        self.instruction = instruction or agent_config.get("instruction", "You are a helpful assistant.")
        self.functions = functions or []
        
        # Function tools mapping
        self._function_tool_map: Dict[str, Dict[str, Any]] = {}
        
        # Initialize server aggregator
        self.server_aggregator = ServerAggregator(
            server_names=self.server_names,
            connection_persistence=self.connection_persistence,
            context=self.context,
            name=self.agent_name
        )
        
        # Human input handling
        self.human_input_callback = human_input_callback
        if not human_input_callback and hasattr(context, "human_input_handler"):
            self.human_input_callback = context.human_input_handler
        
        # Set up logging
        self.logger = get_logger(f"agent.{self.agent_name}" if self.agent_name else "agent")
        
        # Set up LLM configuration and provider
        self.llm_config = llm_config or self._load_llm_config_from_agent_config(agent_config)
        self.llm_provider = self.llm_config.create_provider()
        
        # Conversation history
        self.messages = []
        if self.llm_config.system_prompt:
            self.messages.append({"role": "system", "content": self.llm_config.system_prompt})
        
        # Track tool calls for analysis
        self.tool_call_history = []
        
        # Initialization flag
        self.initialized = False
    
    def _load_llm_config_from_agent_config(self, agent_config: Dict[str, Any]) -> LLMConfig:
        """
        Load LLM configuration from agent configuration.
        
        Args:
            agent_config: Agent configuration dictionary.
            
        Returns:
            LLM configuration.
        """
        llm_config = agent_config.get("llm", {})
        return LLMConfig.from_dict(llm_config)
    
    async def __aenter__(self):
        """Initialize the agent when entering an async context."""
        if not self.initialized:
            # Initialize the server aggregator
            await self.server_aggregator.__aenter__()
            # Register function tools
            await self._register_function_tools()
            self.initialized = True
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up the agent when exiting an async context."""
        await self.server_aggregator.__aexit__(exc_type, exc_val, exc_tb)
    
    async def _register_function_tools(self):
        """Register function tools from the functions list."""
        for function in self.functions:
            # Extract function metadata
            name = function.__name__
            doc = function.__doc__ or "No description available"
            
            # TODO: Add parameter schema extraction using function annotations
            # For now, use a simple placeholder schema
            parameters = {
                "type": "object",
                "properties": {},
                "required": []
            }
            
            # Register the function as a tool
            self._function_tool_map[name] = {
                "name": name,
                "description": doc,
                "parameters": parameters,
                "function": function
            }
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        List all available tools from connected servers and local functions.
        
        Returns:
            List of tool information dictionaries.
        """
        if not self.initialized:
            await self.__aenter__()
            
        tools_result = await self.server_aggregator.list_tools()
        tools = [tool.model_dump() for tool in tools_result.tools]
        
        # Add function tools
        for tool in self._function_tool_map.values():
            tools.append({
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"]
            })
        
        # Add human input tool if callback is available
        if self.human_input_callback:
            tools.append({
                "name": "__human_input__",
                "description": "Request input from a human user",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The prompt to show to the human"
                        },
                        "timeout_seconds": {
                            "type": "number",
                            "description": "Timeout in seconds for the human input"
                        }
                    },
                    "required": ["prompt"]
                }
            })
        
        return tools
    
    async def call_tool(
        self, 
        tool_name: str, 
        arguments: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Call a tool by name.
        
        Args:
            tool_name: Name of the tool to call.
            arguments: Arguments to pass to the tool.
            
        Returns:
            Result of the tool call.
        """
        if not self.initialized:
            await self.__aenter__()
            
        self.logger.info(f"Calling tool: {tool_name} with arguments: {arguments}")
        
        # Check if it's a function tool
        if tool_name in self._function_tool_map:
            try:
                function = self._function_tool_map[tool_name]["function"]
                result = await function(**arguments) if arguments else await function()
                return result
            except Exception as e:
                self.logger.error(f"Error calling function tool {tool_name}: {e}")
                raise RuntimeError(f"Function tool call failed: {str(e)}")
        
        # Check if it's a human input request
        if tool_name == "__human_input__" and self.human_input_callback:
            try:
                result = await self.request_human_input(
                    prompt=arguments.get("prompt", "Please provide input:"),
                    timeout_seconds=arguments.get("timeout_seconds", 300)
                )
                return result
            except Exception as e:
                self.logger.error(f"Error handling human input request: {e}")
                raise RuntimeError(f"Human input request failed: {str(e)}")
        
        # Otherwise, call server tool
        result = await self.server_aggregator.call_tool(name=tool_name, arguments=arguments)
        
        if result.isError:
            error_msg = result.content[0].text if result.content else "Unknown error"
            self.logger.error(f"Error calling tool {tool_name}: {error_msg}")
            raise RuntimeError(f"Tool call failed: {error_msg}")
        
        # Extract text content from result
        if result.content and len(result.content) > 0:
            content_item = result.content[0]
            if hasattr(content_item, "text"):
                return content_item.text
            elif hasattr(content_item, "data"):
                return content_item.data
        
        return result.content
    
    async def request_human_input(
        self,
        prompt: str,
        timeout_seconds: Optional[int] = 300
    ) -> str:
        """
        Request input from a human user.
        
        Args:
            prompt: The prompt to show to the human.
            timeout_seconds: Timeout in seconds for the human input.
            
        Returns:
            The human's input as a string.
            
        Raises:
            RuntimeError: If human input callback is not available.
            TimeoutError: If the timeout is exceeded.
        """
        if not self.human_input_callback:
            raise RuntimeError("Human input callback not available")
        
        request_id = f"human_input_{self.agent_name}_{uuid.uuid4()}"
        
        try:
            # Create a request object
            request = {
                "request_id": request_id,
                "prompt": prompt,
                "timeout_seconds": timeout_seconds
            }
            
            # Call the human input callback
            result = await self.human_input_callback(request)
            return result
        except asyncio.TimeoutError:
            raise TimeoutError(f"Human input request timed out after {timeout_seconds} seconds")
        except Exception as e:
            raise RuntimeError(f"Error in human input request: {str(e)}")
    
    async def _get_tool_descriptions(self) -> str:
        """
        Get a string with all available tool descriptions.
        
        Returns:
            String with tool descriptions formatted for LLM context.
        """
        tools = await self.list_tools()
        
        tool_descriptions = []
        for tool in tools:
            description = f"""Tool Name: {tool['name']}
Description: {tool.get('description', 'No description available')}
Parameters: {json.dumps(tool.get('parameters', {}), indent=2)}
"""
            tool_descriptions.append(description)
        
        return "\n".join(tool_descriptions)
    
    async def _generate_llm_response(self, prompt: Optional[str] = None) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: Optional prompt to add to the conversation.
            
        Returns:
            LLM's response.
        """
        if prompt:
            self.messages.append({"role": "user", "content": prompt})
        
        try:
            llm_response = await self.llm_provider.generate(
                messages=self.messages,
                model=self.llm_config.model,
                temperature=self.llm_config.temperature,
                max_tokens=self.llm_config.max_tokens,
                top_p=self.llm_config.top_p,
            )
        except Exception as e:
            self.logger.error(f"Error generating LLM response: {e}")
            llm_response = f"Error generating response: {str(e)}"
        
        # Add the response to the conversation history
        self.messages.append({"role": "assistant", "content": llm_response})
        
        return llm_response
    
    async def _extract_tool_calls(self, llm_response: str) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Extract tool calls from the LLM response.
        
        Supports multiple formats:
        1. JSON blocks: ```json {"tool": "tool_name", "parameters": {...}} ```
        2. Native OpenAI format: response containing tool_calls field
        
        Args:
            llm_response: The response from the LLM.
            
        Returns:
            List of (tool_name, parameters) tuples.
        """
        tool_calls = []
        
        # Check if this is a JSON string containing OpenAI tool calls
        try:
            parsed = json.loads(llm_response)
            if "tool_calls" in parsed:
                for tool_call in parsed["tool_calls"]:
                    if "function" in tool_call:
                        function = tool_call["function"]
                        if "name" in function and "arguments" in function:
                            # Parse arguments JSON string to dict
                            try:
                                args = json.loads(function["arguments"])
                                tool_calls.append((function["name"], args))
                            except json.JSONDecodeError:
                                self.logger.warning(f"Failed to parse tool call arguments: {function['arguments']}")
                return tool_calls
        except json.JSONDecodeError:
            # Not JSON, continue to regex parsing
            pass
        
        # Look for JSON blocks in markdown-style code blocks
        pattern = r"```(?:json)?\s*({.*?})\s*```"
        matches = re.finditer(pattern, llm_response, re.DOTALL)
        
        for match in matches:
            try:
                json_str = match.group(1)
                data = json.loads(json_str)
                
                # Check if the JSON has tool and parameters keys
                if "tool" in data and "parameters" in data:
                    tool_calls.append((data["tool"], data["parameters"]))
            except json.JSONDecodeError:
                continue
        
        return tool_calls
    
    async def run(self, query: str, **kwargs) -> str:
        """
        Run the agent with a query.
        
        This is the main entry point for agent execution.
        
        Args:
            query: The query or request to process.
            **kwargs: Additional arguments for processing.
            
        Returns:
            Result of processing the query.
        """
        if not self.initialized:
            await self.__aenter__()
            
        self.logger.info(f"Processing query: {query}")
        
        # Reset tool call history for this run
        self.tool_call_history = []
        
        # Get tool descriptions for the LLM context
        tool_descriptions = await self._get_tool_descriptions()
        
        # Create the initial prompt with tool information
        prompt = f"""I'll help you with this query: {query}

I have access to the following tools:

{tool_descriptions}

When you want to use a tool, format your response using JSON inside markdown code blocks like this:
```json
{{"tool": "tool_name", "parameters": {{"param1": "value1"}}}}
```

What would you like me to help you with?
"""
        
        # Generate an initial response from the LLM
        llm_response = await self._generate_llm_response(prompt)
        
        # Extract tool calls from the response
        tool_calls = await self._extract_tool_calls(llm_response)
        
        final_response = llm_response
        max_iterations = kwargs.get("max_iterations", 5)
        
        # Process tool calls in a loop
        iteration = 0
        while tool_calls and iteration < max_iterations:
            self.logger.info(f"Iteration {iteration+1}: Found {len(tool_calls)} tool calls")
            
            # Execute each tool call
            tool_results = []
            for tool_name, parameters in tool_calls:
                try:
                    result = await self.call_tool(tool_name, parameters)
                    tool_result = {
                        "tool": tool_name,
                        "parameters": parameters,
                        "result": result
                    }
                    self.tool_call_history.append(tool_result)
                    tool_results.append(tool_result)
                except Exception as e:
                    error = str(e)
                    self.logger.error(f"Error calling tool {tool_name}: {error}")
                    tool_result = {
                        "tool": tool_name,
                        "parameters": parameters,
                        "error": error
                    }
                    self.tool_call_history.append(tool_result)
                    tool_results.append(tool_result)
            
            # Format tool results for the next LLM call
            tool_results_str = json.dumps(tool_results, indent=2)
            next_prompt = f"""I've executed the tool calls with the following results:

```json
{tool_results_str}
```

Please analyze these results and either:
1. Make additional tool calls if needed to complete the task
2. Provide a final answer to the user's query

Remember to use the JSON format in a markdown code block for any additional tool calls."""
            
            # Generate next LLM response based on tool results
            llm_response = await self._generate_llm_response(next_prompt)
            final_response = llm_response
            
            # Extract any new tool calls
            tool_calls = await self._extract_tool_calls(llm_response)
            
            # Increment iteration counter
            iteration += 1
            
            # If no more tool calls, we're done
            if not tool_calls:
                break
        
        # If we hit max iterations, add a note
        if iteration >= max_iterations and tool_calls:
            self.logger.warning(f"Hit maximum iterations ({max_iterations}) with tool calls remaining")
            final_response += "\n\n(Note: Reached maximum number of tool call iterations)"
        
        # Extract a clean response without tool call formatting
        clean_response = self._clean_response(final_response)
        return clean_response
    
    def _clean_response(self, response: str) -> str:
        """
        Clean the response by removing tool call formatting.
        
        Args:
            response: The raw LLM response.
            
        Returns:
            Cleaned response suitable for the end user.
        """
        # Remove JSON code blocks
        clean = re.sub(r"```(?:json)?\s*{.*?}\s*```", "", response, flags=re.DOTALL)
        
        # Remove any remaining markdown formatting
        clean = re.sub(r"```.*?```", "", clean, flags=re.DOTALL)
        
        # Remove extra newlines
        clean = re.sub(r"\n{3,}", "\n\n", clean)
        
        return clean.strip()
    
    def get_tool_call_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of tool calls and their results.
        
        Returns:
            List of tool call dictionaries with tool name, parameters, and results.
        """
        return self.tool_call_history
    
    async def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the conversation history.
        
        Args:
            role: The role (user, assistant, system).
            content: The message content.
        """
        self.messages.append({"role": role, "content": content})
    
    async def clear_messages(self, keep_system: bool = True) -> None:
        """
        Clear conversation history.
        
        Args:
            keep_system: Whether to keep the system message.
        """
        if keep_system and self.messages and self.messages[0]["role"] == "system":
            system_message = self.messages[0]
            self.messages = [system_message]
        else:
            self.messages = []
    
    async def shutdown(self):
        """Shutdown the agent and close all connections."""
        await self.server_aggregator.__aexit__(None, None, None)