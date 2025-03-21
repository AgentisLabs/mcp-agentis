"""
Fun facts server for the LLM agent example.
"""

import asyncio
import os
import sys

# Add the package directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mcp.server import NotificationOptions
from mcp.server.fastmcp import FastMCP
from mcp.server.stdio import stdio_server


app = FastMCP("fun-facts-server")


@app.tool
async def get_fun_fact(topic: str) -> str:
    """
    Get a fun fact about a specific topic.
    
    Args:
        topic: The topic to get a fun fact about.
        
    Returns:
        A fun fact about the topic.
    """
    # This is a mock implementation with pre-defined facts
    print(f"Received fun fact request for: {topic}", file=sys.stderr)
    
    # Dictionary of fun facts for different cities and topics
    facts = {
        "san francisco": [
            "San Francisco's cable cars are the only National Historic Monument that can move.",
            "The Golden Gate Bridge is actually painted 'International Orange', not gold.",
            "San Francisco was founded on June 29, 1776, when colonists from Spain established the Presidio of San Francisco.",
            "The city's famous sourdough bread has a unique taste due to a bacterium called Lactobacillus sanfranciscensis.",
            "Alcatraz Island was home to a federal prison from 1934 to 1963 that housed famous inmates like Al Capone."
        ],
        "new york": [
            "The New York Public Library has over 50 million books and other items and is the second largest library system in the nation.",
            "The first pizzeria in the United States opened in New York City in 1905.",
            "New York City's Federal Reserve Bank has the largest gold storage in the world.",
            "The Empire State Building has its own ZIP code: 10118."
        ],
        "tokyo": [
            "Tokyo was formerly known as Edo until 1868.",
            "Tokyo has the world's busiest pedestrian crossing at Shibuya Crossing.",
            "There are over 200 earthquakes in Tokyo every year, though most are too weak to feel."
        ],
        "python": [
            "The Python programming language is named after the comedy group Monty Python, not the snake.",
            "Python was created by Guido van Rossum in the late 1980s.",
            "Python's design philosophy emphasizes code readability with its notable use of whitespace."
        ],
        "coffee": [
            "Coffee is the second most traded commodity in the world, after oil.",
            "The world's most expensive coffee, Kopi Luwak, comes from beans eaten and excreted by a civet cat.",
            "Coffee beans are actually the pit of a cherry-like berry."
        ]
    }
    
    # Normalize topic to lowercase
    topic_lower = topic.lower()
    
    # Find the closest match
    for key, fact_list in facts.items():
        if key in topic_lower or topic_lower in key:
            import random
            return random.choice(fact_list)
    
    return f"I don't have any fun facts about {topic}, but here's one about San Francisco: San Francisco is built on more than 50 hills!"


@app.tool
async def list_available_topics() -> str:
    """
    List all topics that have fun facts available.
    
    Returns:
        A list of available topics.
    """
    return "Available topics: San Francisco, New York, Tokyo, Python, Coffee"


async def run():
    """Run the fun facts server."""
    async with stdio_server() as (read_stream, write_stream):
        await app._mcp_server.run(
            read_stream,
            write_stream,
            app._mcp_server.create_initialization_options(
                notification_options=NotificationOptions(
                    tools_changed=True, resources_changed=True
                )
            ),
        )


if __name__ == "__main__":
    asyncio.run(run())