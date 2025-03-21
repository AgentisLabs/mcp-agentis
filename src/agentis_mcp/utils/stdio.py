"""
Custom implementation of stdio_client that handles stderr through rich console.
"""

from contextlib import asynccontextmanager
import subprocess
import anyio
from anyio.streams.text import TextReceiveStream
from mcp.client.stdio import StdioServerParameters, get_default_environment
import mcp.types as types
from agentis_mcp.utils.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def stdio_client_with_rich_stderr(server: StdioServerParameters):
    """
    Modified version of stdio_client that captures stderr and routes it through rich console.
    Enhanced for better resilience in persistent connection mode.
    
    Args:
        server: The server parameters for the stdio connection.
        
    Yields:
        A tuple of (read_stream, write_stream) for communication with the server.
    """
    read_stream_writer, read_stream = anyio.create_memory_object_stream(0)
    write_stream, write_stream_reader = anyio.create_memory_object_stream(0)

    # Open process with stderr piped for capture
    try:
        process = await anyio.open_process(
            [server.command, *server.args],
            env=server.env if server.env is not None else get_default_environment(),
            stderr=subprocess.PIPE,
        )

        if process.pid:
            logger.debug(f"Started process '{server.command}' with PID: {process.pid}")

        if process.returncode is not None:
            logger.debug(f"return code (early){process.returncode}")
            raise RuntimeError(
                f"Process terminated immediately with code {process.returncode}"
            )

        async def stdout_reader():
            assert process.stdout, "Opened process is missing stdout"
            try:
                buffer = ""
                async for chunk in TextReceiveStream(
                    process.stdout,
                    encoding=server.encoding,
                    errors=server.encoding_error_handler,
                ):
                    lines = (buffer + chunk).split("\n")
                    buffer = lines.pop()

                    for line in lines:
                        if not line:
                            continue
                        try:
                            message = types.JSONRPCMessage.model_validate_json(line)
                        except Exception as exc:
                            await read_stream_writer.send(exc)
                            continue

                        await read_stream_writer.send(message)
            except anyio.ClosedResourceError:
                logger.debug(f"Stdout stream closed for {server.command}")
            except Exception as e:
                logger.error(f"Error in stdout reader: {e}")
            finally:
                # Ensure we don't leave resources open
                await anyio.lowlevel.checkpoint()

        async def stderr_reader():
            assert process.stderr, "Opened process is missing stderr"
            try:
                async for chunk in TextReceiveStream(
                    process.stderr,
                    encoding=server.encoding,
                    errors=server.encoding_error_handler,
                ):
                    if chunk.strip():
                        # Use standard logging method for stderr output
                        stderr_line = chunk.rstrip()
                        if "[ERROR]" in stderr_line or "Error" in stderr_line:
                            logger.error(f"MCP SERVER STDERR: {stderr_line}")
                        else:
                            logger.debug(f"MCP SERVER STDERR: {stderr_line}")
            except anyio.ClosedResourceError:
                logger.debug(f"Stderr stream closed for {server.command}")
            except Exception as e:
                logger.error(f"Error in stderr reader: {e}")
            finally:
                # Ensure we don't leave resources open
                await anyio.lowlevel.checkpoint()

        async def stdin_writer():
            assert process.stdin, "Opened process is missing stdin"
            try:
                async for message in write_stream_reader:
                    json = message.model_dump_json(by_alias=True, exclude_none=True)
                    await process.stdin.send(
                        (json + "\n").encode(
                            encoding=server.encoding,
                            errors=server.encoding_error_handler,
                        )
                    )
            except anyio.ClosedResourceError:
                logger.debug(f"Stdin stream closed for {server.command}")
            except Exception as e:
                logger.error(f"Error in stdin writer: {e}")
            finally:
                # Ensure we don't leave resources open
                await anyio.lowlevel.checkpoint()

        # Use context managers to handle cleanup automatically
        task_group = None
        try:
            task_group = anyio.create_task_group()
            async with task_group as tg, process:
                tg.start_soon(stdout_reader)
                tg.start_soon(stdin_writer)
                tg.start_soon(stderr_reader)
                yield read_stream, write_stream
        except Exception as e:
            logger.error(f"Error in stdio client context: {e}")
            if task_group is not None:
                try:
                    await task_group.cancel_scope.cancel()
                except Exception:
                    pass
            raise
    except Exception as e:
        logger.error(f"Failed to open process '{server.command}': {e}")
        # Make sure to close stream writers if process creation fails
        await read_stream_writer.aclose()
        await write_stream_reader.aclose()
        raise