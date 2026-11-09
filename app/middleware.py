"""Tope de tamaño del cuerpo de cada pedido.

Tiene que cortar acá y no en cada endpoint: el formulario de un archivo subido se
termina de leer antes de que corra el código del endpoint.
"""

# La de FastAPI, no la de Starlette: al leer el formulario, FastAPI convierte en 400 cualquier
# excepción que no sea la suya, y se perdería el 413.
from fastapi import HTTPException
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from config import get_settings

settings = get_settings()

# Margen para el armado del multipart, además del archivo en sí.
MULTIPART_OVERHEAD_BYTES = 1024 * 1024
_TOO_LARGE_DETAIL = "El cuerpo del pedido supera el tamaño máximo permitido"


class BodySizeLimitMiddleware:
    def __init__(
        self, app: ASGIApp, max_json_bytes: int | None = None, max_upload_bytes: int | None = None
    ) -> None:
        self.app = app
        self.max_json_bytes = max_json_bytes or settings.MAX_JSON_BODY_BYTES
        self.max_upload_bytes = max_upload_bytes or settings.MAX_UPLOAD_BYTES

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        if headers.get("content-type", "").lower().startswith("multipart/form-data"):
            limit = self.max_upload_bytes + MULTIPART_OVERHEAD_BYTES
        else:
            limit = self.max_json_bytes

        too_large = JSONResponse({"detail": _TOO_LARGE_DETAIL}, status_code=413)

        declared = headers.get("content-length", "")
        if declared.isdigit() and int(declared) > limit:
            await too_large(scope, receive, send)
            return

        # Content-Length puede faltar o mentir, así que también se cuentan los bytes reales.
        received = 0
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise HTTPException(status_code=413, detail=_TOO_LARGE_DETAIL)
            return message

        async def tracked_send(message: Message) -> None:
            nonlocal response_started
            response_started = response_started or message["type"] == "http.response.start"
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except HTTPException as exc:
            if exc.status_code != 413 or response_started:
                raise
            await too_large(scope, receive, send)
