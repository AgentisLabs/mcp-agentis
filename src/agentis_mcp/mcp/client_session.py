"""
Custom client session for the Agentis MCP framework.

This extends the base MCP client session with additional functionality
for handling sampling requests and notifications.
"""

from typing import Optional, TYPE_CHECKING

from mcp import ClientSession
from mcp.shared.session import (
    RequestResponder,
    ReceiveResultT,
    ReceiveNotificationT,
    RequestId,
    SendNotificationT,
    SendRequestT,
    SendResultT,
)
from mcp.types import (
    ClientResult,
    CreateMessageRequest,
    CreateMessageResult,
    ErrorData,
    JSONRPCNotification,
    JSONRPCRequest,
    ServerRequest,
    TextContent,
    ListRootsRequest,
    ListRootsResult,
    Root,
)

from agentis_mcp.config import MCPServerSettings
from agentis_mcp.utils.logging import get_logger

if TYPE_CHECKING:
    from agentis_mcp.core.context import AgentContext

logger = get_logger(__name__)


class AgentisMCPClientSession(ClientSession):
    """
    Client session for Agentis MCP framework connections to MCP servers.
    
    Supports:
    - Enhanced logging
    - Sampling request handling
    - Notification handling
    - MCP root configuration
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_config: Optional[MCPServerSettings] = None
        self.context: Optional["AgentContext"] = None

    async def _received_request(
        self, responder: RequestResponder[ServerRequest, ClientResult]
    ) -> None:
        logger.debug("Received request:", data=responder.request.model_dump())
        request = responder.request.root

        if isinstance(request, CreateMessageRequest):
            return await self.handle_sampling_request(request, responder)
        elif isinstance(request, ListRootsRequest):
            # Handle list_roots request by returning configured roots
            if hasattr(self, "server_config") and self.server_config.roots:
                roots = [
                    Root(
                        uri=root.server_uri_alias or root.uri,
                        name=root.name,
                    )
                    for root in self.server_config.roots
                ]

                await responder.respond(ListRootsResult(roots=roots))
            else:
                await responder.respond(ListRootsResult(roots=[]))
            return

        # Handle other requests as usual
        await super()._received_request(responder)

    async def send_request(
        self,
        request: SendRequestT,
        result_type: type[ReceiveResultT],
    ) -> ReceiveResultT:
        logger.debug("send_request: request=", data=request.model_dump())
        try:
            result = await super().send_request(request, result_type)
            logger.debug("send_request: response=", data=result.model_dump())
            return result
        except Exception as e:
            logger.error(f"send_request failed: {e}")
            raise

    async def send_notification(self, notification: SendNotificationT) -> None:
        logger.debug("send_notification:", data=notification.model_dump())
        try:
            return await super().send_notification(notification)
        except Exception as e:
            logger.error("send_notification failed", data=e)
            raise

    async def _send_response(
        self, request_id: RequestId, response: SendResultT | ErrorData
    ) -> None:
        logger.debug(
            f"send_response: request_id={request_id}, response=",
            data=response.model_dump(),
        )
        return await super()._send_response(request_id, response)

    async def _received_notification(self, notification: ReceiveNotificationT) -> None:
        """
        Handle a notification from the server.
        
        Can be overridden by subclasses to handle a notification without needing
        to listen on the message stream.
        """
        logger.info(
            "_received_notification: notification=",
            data=notification.model_dump(),
        )
        return await super()._received_notification(notification)

    async def send_progress_notification(
        self, progress_token: str | int, progress: float, total: float | None = None
    ) -> None:
        """
        Send a progress notification for a request that is being processed.
        """
        logger.debug(
            f"send_progress_notification: progress_token={progress_token}, progress={progress}, total={total}"
        )
        return await super().send_progress_notification(
            progress_token=progress_token, progress=progress, total=total
        )

    async def handle_sampling_request(
        self,
        request: CreateMessageRequest,
        responder: RequestResponder[ServerRequest, ClientResult],
    ):
        """
        Handle a sampling request from the server.
        
        Args:
            request: The sampling request from the server.
            responder: The responder to send the result back to the server.
        """
        logger.info("Handling sampling request: %s", request)
        
        # If we have a context with an upstream session
        if hasattr(self, "context") and self.context:
            try:
                if hasattr(self.context, "upstream_session") and self.context.upstream_session:
                    # If a session is available, pass-through the sampling request to the upstream client
                    result = await self.context.upstream_session.send_request(
                        request=ServerRequest(request), result_type=CreateMessageResult
                    )
                    
                    # Pass the result back to the server
                    await responder.send_result(result)
                    return
            except Exception as e:
                logger.error(f"Error handling upstream sampling request: {e}")
                    
        # If no upstream session or error, respond with an error
        await responder.respond(
            ErrorData(
                code=-32603, 
                message="No upstream LLM provider available for handling this request"
            )
        )