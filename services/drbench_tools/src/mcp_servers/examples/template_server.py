"""
This module provides a template for creating new MCP servers.
Use this as a starting point when implementing a new service.
"""

import logging
from typing import Any, Dict

from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

mcp = FastMCP(
    name="service-mcp-template",
    version="0.1.0",
    description="Template for MCP servers",
    port=9091,
)


@mcp.resource("template://info")
def get_info() -> Dict[str, str]:
    """Get information about this MCP server template."""
    return {
        "name": "service-mcp-template",
        "version": "0.1.0",
        "description": "Template for MCP servers",
    }


@mcp.tool()
async def example_tool(parameter1: str, parameter2: int = 0) -> Dict[str, Any]:
    """
    Example tool with parameters.

    Args:
        parameter1: A string parameter
        parameter2: An optional integer parameter

    Returns:
        Dictionary with results
    """
    try:
        result = {
            "success": True,
            "parameter1": parameter1,
            "parameter2": parameter2,
            "message": f"Processed {parameter1} with value {parameter2}",
        }
        return result
    except Exception as e:
        logger.error(f"Error in example_tool: {str(e)}")
        return {"success": False, "error": str(e)}



if __name__ == "__main__":
    try:
        logger.info(f"Starting template MCP server on port {mcp.settings.port}")
        
        logger.info("Using SSE transport")
        mcp.run(transport="sse")
        
    except KeyboardInterrupt:
        logger.info("Template MCP server stopped by user")
    except Exception as e:
        logger.error(f"Error starting template MCP server: {str(e)}")
        raise
