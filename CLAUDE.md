# Agentis MCP Development Guide

## Build/Test/Lint Commands
- Install: `pip install -e .`
- Run tests: `pytest`
- Run single test: `pytest <path_to_test_file>::<test_function_name>`
- Format code: `black . && isort .`
- Lint code: `ruff check .`
- Type check: `mypy .`

## Code Style Guidelines
- **Python Version**: 3.10+
- **Formatting**: Black (line length 88), isort for imports
- **Linting**: Ruff (error, flake8, import checks)
- **Type Checking**: mypy with strict settings
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Imports**: Group by standard lib, third-party, local; sort alphabetically
- **Docstrings**: Triple quotes with param descriptions and return types
- **Error Handling**: Use try/except with specific exception types
- **Async Pattern**: Use async/await throughout with proper exception handling
- **Configuration**: YAML files with environment variable fallbacks

## Important Framework Settings
- **MCP Connection Persistence**: When creating an Agent, set `connection_persistence=True` to keep MCP server connections open for multiple tool calls. By default, connections close after each tool call.
```python
agent = Agent(
    context=running_app.context,
    agent_name="assistant",
    server_names=["filesystem", "fetch"],
    connection_persistence=True,  # Keep MCP server connections open
    # other parameters...
)
```

## Secret Management

The framework provides several methods for managing API keys and other secrets:

### Using .env Files (Recommended)

1. Create a `.env` file in your project root:
```
# .env
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
BRAVE_API_KEY=...
```

2. Use the secrets utility in your code:
```python
from agentis_mcp.utils.secrets import get_api_key

# Get API key from .env file or environment variables
ANTHROPIC_API_KEY = get_api_key("anthropic")
```

3. Generate a template .env file:
```python
from agentis_mcp.utils.secrets import generate_env_template
generate_env_template()  # Creates .env.example
```

### Using Environment Variables

API keys can be set directly as environment variables:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
```

### Using Secrets YAML (Legacy)

API keys can also be stored in `agentis_mcp.secrets.yaml`:
```yaml
# agentis_mcp.secrets.yaml
anthropic:
  api_key: "sk-ant-..."
openai:
  api_key: "sk-..."
```

**Note**: All secret files (.env, *.secrets.yaml) are excluded in .gitignore to prevent accidental commits.