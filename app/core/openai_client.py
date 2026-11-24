"""Translate provider failures into actionable client-facing messages."""

from collections.abc import Iterator
from contextlib import contextmanager

import openai
from openai import AsyncOpenAI

from app.core.errors import AppError


def require_client(client: AsyncOpenAI | None) -> AsyncOpenAI:
    if client is None:
        raise AppError(503, "Set OPENAI_API_KEY on the API server to use OpenAI.")
    return client


@contextmanager
def provider_errors() -> Iterator[None]:
    try:
        yield
    except openai.RateLimitError as error:
        raise AppError(429, "OpenAI rate limit or quota reached. Check your quota or try again.") from error
    except (openai.AuthenticationError, openai.PermissionDeniedError) as error:
        raise AppError(503, "Check the server's OpenAI API key and model permissions.") from error
    except openai.APITimeoutError as error:
        raise AppError(504, "OpenAI timed out. Try again.") from error
    except openai.APIConnectionError as error:
        raise AppError(503, "OpenAI is temporarily unreachable. Try again.") from error
    except openai.APIError as error:
        raise AppError(502, "OpenAI could not complete the request.") from error
