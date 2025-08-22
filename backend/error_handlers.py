import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

logger = logging.getLogger("mr_tai.errors")

def register_error_handlers(app: FastAPI):
    @app.exception_handler(HTTPException)
    async def http_exc_handler(request: Request, exc: HTTPException):
        logger.warning(
            "HTTPException path=%s status=%s detail=%r",
            request.url.path, exc.status_code, exc.detail
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": str(exc.detail) if exc.detail else "HTTP error"},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exc_handler(request: Request, exc: RequestValidationError):
        logger.warning(
            "ValidationError path=%s errors=%s",
            request.url.path, exc.errors()
        )
        return JSONResponse(
            status_code=422,
            content={"error": "Validation error", "details": exc.errors()},
        )

    @app.exception_handler(Exception)
    async def unhandled_exc_handler(request: Request, exc: Exception):
        logger.exception("Unhandled error at path=%s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"},
        )
