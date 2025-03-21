"""
LLM-based router implementation for directing queries.
"""

import asyncio
import inspect
import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from agentis_mcp.agents import Agent, LLMConfig, LLMProvider
from agentis_mcp.core.context import AgentContext
from agentis_mcp.utils.logging import get_logger
from agentis_mcp.workflows.router.router_base import Router, RouterResult, RouteType, CallableFn

logger = get_logger(__name__)


class LLMRouter:
    """
    LLM-based router that uses a language model to determine the best handler for a request.
    
    This router uses a language model to analyze the request and determine whether it should
    be handled by an agent, server, or function based on their descriptions and capabilities.
    """
    
    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        agents: Optional[List[Agent]] = None,
        server_names: Optional[List[str]] = None,
        functions: Optional[List[CallableFn]] = None,
        context: Optional[AgentContext] = None,
    ):
        """
        Initialize the LLM router.
        
        Args:
            llm_config: Configuration for the LLM used for routing.
            agents: List of agents available for routing.
            server_names: List of server names available for routing.
            functions: List of functions available for routing.
            context: Agent context for configuration and server access.
        """
        self.llm_config = llm_config or LLMConfig(
            provider="mock",  # Default to mock for testing
            system_prompt=self._get_system_prompt(),
            temperature=0.1,  # Low temperature for more deterministic routing
        )
        
        self.agents = agents or []
        self.server_names = server_names or []
        self.functions = functions or []
        self.context = context
        
        # Create provider from config
        self.llm_provider = self.llm_config.create_provider()
        
        # Cache for function descriptions
        self._function_descriptions: Dict[str, str] = {}
        for func in self.functions:
            self._function_descriptions[func.__name__] = self._get_function_description(func)
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the router."""
        return """You are a routing assistant responsible for determining the most appropriate handler for a user's request.
Your task is to analyze the request and match it to the best available agent, server, or function based on their capabilities.
Always consider the specific requirements and constraints of the request.
Provide a brief explanation for your choice, focusing on why the selected handler is the best match.
Format your responses as valid JSON objects that include name, score, and explanation fields.
"""
    
    def _get_function_description(self, func: CallableFn) -> str:
        """Extract a description from a function's docstring and signature."""
        doc = inspect.getdoc(func) or "No description available."
        sig = inspect.signature(func)
        params = []
        
        for name, param in sig.parameters.items():
            if name != 'self':
                annotation = ""
                if param.annotation != inspect.Parameter.empty:
                    annotation = f": {param.annotation.__name__}"
                default = ""
                if param.default != inspect.Parameter.empty:
                    default = f" = {param.default}"
                params.append(f"{name}{annotation}{default}")
        
        params_str = ", ".join(params)
        return f"{func.__name__}({params_str}): {doc}"
    
    async def _get_agent_descriptions(self) -> List[Dict[str, Any]]:
        """Get descriptions of all available agents."""
        agent_descriptions = []
        
        for agent in self.agents:
            # Get agent tools if initialized
            tools = []
            if hasattr(agent, 'initialized') and agent.initialized:
                tools = await agent.list_tools()
            
            agent_descriptions.append({
                "name": agent.agent_name,
                "description": getattr(agent, "instruction", "No description available."),
                "tools_count": len(tools) if tools else "unknown",
                "server_names": agent.server_names,
            })
        
        return agent_descriptions
    
    async def _get_server_descriptions(self) -> List[Dict[str, Any]]:
        """Get descriptions of all available servers."""
        server_descriptions = []
        
        if not self.context:
            return [{"name": name, "description": "No description available"} for name in self.server_names]
        
        for server_name in self.server_names:
            server_config = self.context.config.mcp.servers.get(server_name)
            if server_config:
                description = f"Server for {server_name}"
                if server_config.command:
                    description = f"Server running {server_config.command} {' '.join(server_config.args or [])}"
                
                server_descriptions.append({
                    "name": server_name,
                    "description": description,
                })
            else:
                server_descriptions.append({
                    "name": server_name,
                    "description": f"Server for {server_name}",
                })
        
        return server_descriptions
    
    async def _get_function_descriptions(self) -> List[Dict[str, Any]]:
        """Get descriptions of all available functions."""
        return [
            {"name": func.__name__, "description": self._function_descriptions[func.__name__]}
            for func in self.functions
        ]
    
    async def _route_with_llm(
        self, 
        request: str, 
        candidates: List[Dict[str, Any]], 
        route_type: str,
        top_k: int = 1
    ) -> List[RouterResult]:
        """
        Use the LLM to route the request to the appropriate candidates.
        
        Args:
            request: The user's query or request.
            candidates: List of candidate objects with name and description.
            route_type: Type of routing (agent, server, function).
            top_k: Number of top matches to return.
            
        Returns:
            List of RouterResult objects, sorted by confidence score.
        """
        if not candidates:
            return []
        
        prompt = f"""
I need to determine the most appropriate {route_type} for the following user request:

"{request}"

Available {route_type}s:
"""
        
        for i, candidate in enumerate(candidates, 1):
            prompt += f"\n{i}. {candidate['name']}: {candidate['description']}"
        
        prompt += f"""

Please respond with a JSON array containing the top {top_k} most suitable {route_type}(s) for this request.
For each {route_type}, include:
1. name: The name of the {route_type}
2. score: A confidence score between 0 and 1
3. explanation: A brief explanation of why this {route_type} is appropriate

Example format:
```json
[
  {{
    "name": "example_{route_type}_1",
    "score": 0.95,
    "explanation": "This {route_type} is most appropriate because..."
  }},
  ...
]
```
"""

        # Generate response from LLM
        messages = [
            {"role": "system", "content": self.llm_config.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self.llm_provider.generate(
                messages=messages,
                model=self.llm_config.model,
                temperature=self.llm_config.temperature,
                max_tokens=self.llm_config.max_tokens,
            )
            
            # Extract JSON from response
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```|(\[[\s\S]*\])', response)
            if json_match:
                json_str = json_match.group(1) or json_match.group(2)
                rankings = json.loads(json_str)
            else:
                # Try to extract JSON without code block markers
                try:
                    rankings = json.loads(response)
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse LLM response as JSON: {response}")
                    return []
            
            # Convert to RouterResult objects
            results = []
            route_type_enum = RouteType(route_type.upper())
            
            for ranking in rankings[:top_k]:
                name = ranking.get("name")
                score = float(ranking.get("score", 0.0))
                explanation = ranking.get("explanation", "No explanation provided.")
                
                # Find the actual object for this route
                result_obj = None
                
                if route_type_enum == RouteType.AGENT:
                    for agent in self.agents:
                        if agent.agent_name == name:
                            result_obj = agent
                            break
                elif route_type_enum == RouteType.SERVER:
                    if name in self.server_names:
                        result_obj = name
                elif route_type_enum == RouteType.FUNCTION:
                    for func in self.functions:
                        if func.__name__ == name:
                            result_obj = func
                            break
                
                if result_obj:
                    results.append(RouterResult(
                        route_type=route_type_enum,
                        name=name,
                        score=score,
                        description=explanation,
                        result=result_obj
                    ))
            
            # Sort by score
            results.sort(key=lambda x: x.score, reverse=True)
            return results
        
        except Exception as e:
            logger.error(f"Error routing with LLM: {e}")
            return []
    
    async def route(
        self, request: str, top_k: int = 1
    ) -> List[RouterResult]:
        """
        Route a request to the most appropriate handler (agent, server, or function).
        
        Args:
            request: The user's query or request.
            top_k: Number of top matches to return.
            
        Returns:
            List of RouterResult objects, sorted by confidence score.
        """
        # Get all possible routes
        routes = []
        
        # Concurrently get all route types
        agent_future = self.route_to_agent(request, top_k=top_k)
        server_future = self.route_to_server(request, top_k=top_k)
        function_future = self.route_to_function(request, top_k=top_k)
        
        results = await asyncio.gather(agent_future, server_future, function_future)
        
        # Combine all results
        for result_list in results:
            routes.extend(result_list)
        
        # Sort by score and return top_k
        routes.sort(key=lambda x: x.score, reverse=True)
        return routes[:top_k]
    
    async def route_to_agent(
        self, request: str, top_k: int = 1
    ) -> List[RouterResult]:
        """
        Route a request specifically to an agent.
        
        Args:
            request: The user's query or request.
            top_k: Number of top agent matches to return.
            
        Returns:
            List of RouterResult objects for agents, sorted by confidence score.
        """
        if not self.agents:
            return []
        
        agent_descriptions = await self._get_agent_descriptions()
        return await self._route_with_llm(request, agent_descriptions, "agent", top_k)
    
    async def route_to_server(
        self, request: str, top_k: int = 1
    ) -> List[RouterResult]:
        """
        Route a request specifically to a server.
        
        Args:
            request: The user's query or request.
            top_k: Number of top server matches to return.
            
        Returns:
            List of RouterResult objects for servers, sorted by confidence score.
        """
        if not self.server_names:
            return []
        
        server_descriptions = await self._get_server_descriptions()
        return await self._route_with_llm(request, server_descriptions, "server", top_k)
    
    async def route_to_function(
        self, request: str, top_k: int = 1
    ) -> List[RouterResult]:
        """
        Route a request specifically to a function.
        
        Args:
            request: The user's query or request.
            top_k: Number of top function matches to return.
            
        Returns:
            List of RouterResult objects for functions, sorted by confidence score.
        """
        if not self.functions:
            return []
        
        function_descriptions = await self._get_function_descriptions()
        return await self._route_with_llm(request, function_descriptions, "function", top_k)