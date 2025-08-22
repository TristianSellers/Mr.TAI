import logging
import time
from typing import Callable
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

# Configure root logger once (simple, readable format)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("mr_tai.request")

class RequestLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        start = time.perf_counter()
        client = request.client.host if request.client else "-"
        method = request.method
        path = request.url.path

        try:
            response = await call_next(request)
            duration_ms = (time.perf_counter() - start) * 1000.0
            logger.info(
                "client=%s method=%s path=%s status=%s duration_ms=%.2f",
                client, method, path, response.status_code, duration_ms
            )
            return response
        except Exception:
            duration_ms = (time.perf_counter() - start) * 1000.0
            logger.exception(
                "client=%s method=%s path=%s status=%s duration_ms=%.2f UNHANDLED",
                client, method, path, 500, duration_ms
            )
            raise

def register_request_logging(app):
    app.add_middleware(RequestLogMiddleware)
