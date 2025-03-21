# Agentis MCP

A flexible multi-agent framework for building powerful AI agents with MCP server connectivity.

## Features

- Connect to MCP servers for tool access and resource retrieval
- Build multi-agent workflows with powerful orchestration
- Simple and intuitive API for creating custom agents
- Flexible configuration system
- Support for different transport mechanisms (stdio, SSE)
- Persistent and temporary connection management
- Aggregation of multiple tool servers

## Installation

```bash
pip install agentis-mcp
```

## Quick Start

```python
import asyncio
from agentis_mcp import Agent, AgentContext
from agentis_mcp.config import load_config

async def main():
    # Load the configuration from a YAML file
    config = load_config("config.yaml")
    
    # Create an agent context
    context = AgentContext(config)
    
    # Create an agent with the context
    async with Agent(context) as agent:
        # Run a task with the agent
        result = await agent.run("What's the weather in San Francisco?")
        print(result)

asyncio.run(main())
```

## Documentation

For detailed documentation, see the [docs](docs/) directory.

## License

MIT