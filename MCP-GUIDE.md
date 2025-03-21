# MCP Integration Guide

This document describes how Agentis MCP integrates with MCP servers and the overall architecture of the MCP connectivity layer.

## MCP Overview

MCP (Machine Conversation Protocol) is a protocol for communication between AI agents and tool-providing servers. It allows agents to discover and use tools, access resources, and engage in structured conversation.

## Architecture

The Agentis MCP framework implements several key components for MCP integration:

### Server Registry

The `ServerRegistry` manages server configurations loaded from YAML files. It handles:

- Server configuration storage and validation
- Server initialization and launching
- Transport selection (stdio, SSE)
- Registration of initialization hooks

### Connection Manager

The `ConnectionManager` manages the lifecycle of MCP server connections, supporting:

- Persistent and temporary connections
- Connection pooling
- Reconnection handling
- Concurrent connections to multiple servers

### MCP Aggregator

The `MCPAggregator` combines multiple MCP servers into a unified interface:

- Namespaces tools by server name to avoid conflicts
- Routes tool calls to appropriate servers
- Provides a unified API for accessing tools across servers
- Supports persistent connections to frequently used servers

### Client Sessions

The `AgentisClientSession` extends the standard MCP client session with:

- Enhanced logging
- Sampling request handling
- Error recovery
- Progress notifications

## Connection Flow

1. Server configuration is loaded from YAML files
2. An agent requests a connection to a server
3. The connection manager establishes the appropriate transport
4. A client session is created and initialized
5. The agent can call tools and access resources
6. For persistent connections, the connection is maintained in the connection pool

## Transport Mechanisms

The framework supports multiple transport mechanisms:

### stdio

Uses subprocess communication for local servers, with:
- stdin/stdout for JSON-RPC messages
- stderr redirection through rich console

### SSE (Server-Sent Events)

Uses HTTP for remote servers, supporting:
- Bidirectional communication over HTTP
- Connection to remote servers with URL configuration

## Configuration

Server configurations are defined in YAML files:

```yaml
mcp:
  servers:
    my_server:
      transport: stdio
      command: python
      args: ["-m", "my_server"]
      env:
        MY_ENV_VAR: value
      read_timeout_seconds: 30
    
    remote_server:
      transport: sse
      url: https://example.com/mcp-server
      read_timeout_seconds: 10
```

## Multi-Agent Communication

Agents can communicate with each other using:

1. Direct tool calls
2. Shared state in the context
3. Workflow orchestration patterns

Each agent has its own connection manager and can access different servers based on its configuration.

## Server Development

To create an MCP server that's compatible with Agentis MCP:

1. Use FastMCP to create a server
2. Register tools using decorators
3. Implement handlers for tool requests
4. Support the same transport mechanisms