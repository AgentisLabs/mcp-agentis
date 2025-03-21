"""
Workflow components for the Agentis MCP framework.

This module provides workflow components that extend the core functionality
of agents, including routing, parallel processing, and other advanced patterns.
"""

# Import router components
from agentis_mcp.workflows.router import Router, RouterResult, LLMRouter

# Import parallel processing components
from agentis_mcp.workflows.parallel import (
    FanOut, FanOutCallable, 
    FanIn, FanInInput, 
    ParallelLLM
)