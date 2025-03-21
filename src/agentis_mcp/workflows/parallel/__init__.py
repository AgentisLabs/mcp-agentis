"""
Parallel processing workflows for the Agentis MCP framework.

This module provides components for parallel processing, including:
- FanOut: Distributes work to multiple agents or functions
- FanIn: Aggregates results from multiple sources
- ParallelLLM: Coordinates parallel processing with fan-out and fan-in
"""

from agentis_mcp.workflows.parallel.fan_out import FanOut, FanOutCallable
from agentis_mcp.workflows.parallel.fan_in import FanIn, FanInInput
from agentis_mcp.workflows.parallel.parallel_llm import ParallelLLM

__all__ = ["FanOut", "FanOutCallable", "FanIn", "FanInInput", "ParallelLLM"]