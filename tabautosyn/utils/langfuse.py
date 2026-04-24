"""Langfuse client initialization for judge traces."""

import json
import os
import secrets
import socket
from typing import Any
from urllib.parse import urlparse

from langfuse import Langfuse
from dotenv import load_dotenv

# Initialize Langfuse client for uploading maseval judge traces
# This sets the global OpenTelemetry tracer provider for pydantic AI instrumentation
_langfuse_judge_client = None
_instrumentation_initialized = False
_langfuse_disabled_for_process = False
load_dotenv()


def _is_truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _langfuse_host_reachable(host: str, timeout_sec: float = 0.5) -> bool:
    """Best-effort TCP reachability check for LANGFUSE_BASE_URL."""
    try:
        parsed = urlparse(host)
        hostname = parsed.hostname
        if not hostname:
            return False
        if parsed.port is not None:
            port = parsed.port
        elif parsed.scheme == "https":
            port = 443
        else:
            port = 80
        with socket.create_connection((hostname, port), timeout=timeout_sec):
            return True
    except Exception:
        return False


def get_langfuse_judge_client() -> Langfuse | None:
    """Get or create the Langfuse client for uploading maseval judge traces.

    This client uses LANGFUSE_PUBLIC_KEY_JUDGE and LANGFUSE_SECRET_KEY_JUDGE
    environment variables. It sets the global OpenTelemetry tracer provider,
    which enables pydantic AI agents to automatically upload traces.

    IMPORTANT: Call this AFTER you've finished downloading traces with the
    download client, to ensure the judge client's tracer provider is active
    when evaluations run.

    Returns:
        Langfuse | None: The initialized Langfuse client for judge traces,
        or ``None`` when Langfuse env vars are not configured.
    """
    global _langfuse_judge_client, _instrumentation_initialized, _langfuse_disabled_for_process

    # Explicit global switch for local/offline runs.
    if os.getenv("LANGFUSE_ENABLED") and not _is_truthy(os.getenv("LANGFUSE_ENABLED")):
        return None

    if _langfuse_disabled_for_process:
        return None

    if _langfuse_judge_client is None:
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        host = os.getenv("LANGFUSE_BASE_URL")

        # Langfuse tracing is optional. If credentials are not configured,
        # skip initialization quietly so local runs do not fail/noise.
        if not public_key or not secret_key or not host:
            return None
        # If endpoint is unreachable, disable Langfuse for this process to avoid
        # repeated OTEL export timeouts and noisy stack traces.
        probe_timeout = float(os.getenv("LANGFUSE_PROBE_TIMEOUT_SEC", "0.5"))
        if not _langfuse_host_reachable(host, timeout_sec=probe_timeout):
            _langfuse_disabled_for_process = True
            return None

        # Initialize with judge keys for uploading traces
        # This sets the global OpenTelemetry tracer provider
        _langfuse_judge_client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )

        # Enable instrumentation after judge client is initialized
        if not _instrumentation_initialized:
            from pydantic_ai import Agent

            Agent.instrument_all()
            _instrumentation_initialized = True

    return _langfuse_judge_client


def langfuse_output_payload(
    obj: Any,
    *,
    key: str = "result",
    max_json_chars: int = 250_000,
) -> dict[str, Any]:
    """Build Langfuse ``output`` dict: full *obj* when JSON-serializable and small, else truncated JSON string."""
    try:
        raw = json.dumps(obj, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return {key: str(obj)}
    if len(raw) <= max_json_chars:
        return {key: obj}
    cut = max_json_chars - 80
    return {
        f"{key}_json": raw[:cut] + "\n\n... [truncated by tabautosyn]",
        "truncated": True,
        "original_json_chars": len(raw),
    }


def langfuse_safe_trace(
    langfuse_client: Any,
    name: str,
    input_payload: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    *,
    new_trace: bool = False,
) -> Any:
    """Root observation: Langfuse v3+ ``start_span`` (call ``langfuse_safe_end``) or legacy ``trace``.

    If *new_trace* is True, pass a fresh ``trace_context`` so the span is not attached to the
    active OpenTelemetry span (e.g. ``plugin.*``), which otherwise can swallow or merge spans.
    """
    if not langfuse_client:
        return None
    if hasattr(langfuse_client, "start_span"):
        span_kwargs: dict[str, Any] = {
            "name": name,
            "input": input_payload or {},
            "metadata": metadata or {},
        }
        if new_trace:
            span_kwargs["trace_context"] = {"trace_id": secrets.token_hex(16)}
        try:
            span = langfuse_client.start_span(**span_kwargs)
        except TypeError:
            try:
                span = langfuse_client.start_span(name=name)
            except Exception:
                return None
        except Exception:
            return None
        else:
            if hasattr(span, "update_trace"):
                try:
                    span.update_trace(
                        name=name,
                        input=input_payload or {},
                        metadata=metadata or {},
                    )
                except TypeError:
                    try:
                        span.update_trace(name=name)
                    except Exception:
                        pass
                except Exception:
                    pass
            return span
    if hasattr(langfuse_client, "trace"):
        try:
            return langfuse_client.trace(
                name=name, input=input_payload, metadata=metadata
            )
        except TypeError:
            try:
                return langfuse_client.trace(name=name, metadata=metadata)
            except Exception:
                return None
        except Exception:
            return None
    return None


def langfuse_safe_span(
    parent: Any,
    name: str,
    input_payload: dict[str, Any] | None = None,
) -> Any:
    """Child span: ``parent.start_span`` (v3+) or legacy ``parent.span``."""
    if not parent:
        return None
    if hasattr(parent, "start_span"):
        try:
            return parent.start_span(name=name, input=input_payload or {})
        except TypeError:
            try:
                return parent.start_span(name=name)
            except Exception:
                return None
        except Exception:
            return None
    if hasattr(parent, "span"):
        try:
            return parent.span(name=name, input=input_payload)
        except TypeError:
            try:
                return parent.span(name=name)
            except Exception:
                return None
        except Exception:
            return None
    return None


def langfuse_safe_end(obj: Any) -> None:
    if not obj or not hasattr(obj, "end"):
        return
    try:
        obj.end()
    except Exception:
        return


def langfuse_safe_update(
    obj: Any,
    output_payload: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    level: str | None = None,
    status_message: str | None = None,
) -> None:
    if not obj or not hasattr(obj, "update"):
        return
    kwargs: dict[str, Any] = {}
    if output_payload is not None:
        kwargs["output"] = output_payload
    if metadata is not None:
        kwargs["metadata"] = metadata
    if level is not None:
        kwargs["level"] = level
    if status_message is not None:
        kwargs["status_message"] = status_message
    if not kwargs:
        return
    try:
        obj.update(**kwargs)
    except TypeError:
        fallback_kwargs = {
            k: v for k, v in kwargs.items() if k in {"output", "metadata"}
        }
        if fallback_kwargs:
            try:
                obj.update(**fallback_kwargs)
            except Exception:
                return
    except Exception:
        return
