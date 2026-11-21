"""OpenAI-compatible endpoints over the Gemini web app.

Three routes matter: `/v1/chat/completions`, `/v1/responses`, and `/v1/quota`, which the whole
point of this server is to make visible. Every turn is a real Gemini web request on a signed-in
account, so the limit that applies is that account's own.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from gemini_webapi.exceptions import (
    APIError,
    AuthError,
    GeminiError,
    ModelInvalidError,
    TemporarilyBlockedError,
    UsageLimitExceededError,
)
from gemini_webapi.exceptions import (
    TimeoutError as GeminiTimeoutError,
)
from loguru import logger

from . import prompt as prompts
from ._version import __version__
from .config import Settings, load
from .keys import KeyStore
from .pool import Pool, StoredChat, history_key

SSE_DONE = "data: [DONE]\n\n"


def _error(
    status: int, message: str, kind: str = "invalid_request_error", code: str | None = None
) -> HTTPException:
    payload = {"error": {"message": message, "type": kind, "param": None, "code": code}}
    return HTTPException(status_code=status, detail=payload)


def _upstream_error(exc: Exception) -> HTTPException:
    """Map a library exception onto the status an OpenAI client understands."""
    if isinstance(exc, prompts.PromptError):
        return _error(400, str(exc))
    if isinstance(exc, UsageLimitExceededError):
        return _error(
            429, f"Gemini usage limit reached for this account: {exc}", "rate_limit_error"
        )
    if isinstance(exc, TemporarilyBlockedError):
        return _error(429, f"Gemini is rate limiting this account: {exc}", "rate_limit_error")
    if isinstance(exc, AuthError):
        return _error(401, f"Gemini sign-in is no longer valid: {exc}", "authentication_error")
    if isinstance(exc, ModelInvalidError):
        return _error(400, f"Gemini rejected this model: {exc}")
    if isinstance(exc, GeminiTimeoutError):
        return _error(504, f"Gemini did not answer in time: {exc}", "server_error")
    if isinstance(exc, APIError):
        return _error(502, f"Gemini returned an unusable response: {exc}", "server_error")
    if isinstance(exc, GeminiError):
        return _error(502, f"Gemini refused the request: {exc}", "server_error")
    return _error(502, f"Gemini request failed: {exc}", "server_error")


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or load()
    minted = KeyStore(resolved.keys_path)
    # Vision is advertised as false until images actually reach Gemini. `gemini-webapi` names
    # every in-memory upload `input_NNNNNNN.txt` and derives the content type from that, so an
    # image is announced to Google as text/plain and the generation request never answers.
    capabilities = {
        "supports_vision": False,
        "supports_reasoning": True,
    }
    # The prompt is cut at `max_prompt_chars`, so that is the budget a client should plan with
    # (four characters per token is the usual rough conversion).
    context_window = max(1, resolved.max_prompt_chars // 4)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        pool = Pool(resolved)
        await pool.start()
        app.state.pool = pool
        reaper = asyncio.create_task(_reap_loop(pool))
        try:
            yield
        finally:
            reaper.cancel()
            await asyncio.gather(reaper, return_exceptions=True)
            await pool.close()

    app = FastAPI(title="gemini2api", version="0.1.0", lifespan=lifespan)

    def get_pool(request: Request) -> Pool:
        return request.app.state.pool

    def key_required() -> bool:
        """Auth applies when the operator set a key, or when any client has minted one."""
        return bool(resolved.api_key) or bool(minted)

    def require_key(request: Request) -> None:
        if not key_required():
            return
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        supplied = token if scheme.lower() == "bearer" else request.headers.get("x-api-key")
        if not supplied:
            raise _error(401, "missing API key", "authentication_error")
        if resolved.api_key and secrets.compare_digest(supplied, resolved.api_key):
            return
        if minted.verify(supplied):
            return
        raise _error(403, "wrong API key", "permission_error")

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, exc: HTTPException):
        detail = exc.detail
        if isinstance(detail, dict) and "error" in detail:
            return JSONResponse(status_code=exc.status_code, content=detail)
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"message": str(detail), "type": "invalid_request_error"}},
        )

    @app.get("/health")
    async def health(pool: Pool = Depends(get_pool)):
        """Unsigned-in state, plus what a client needs to recognise and configure this server."""
        return {
            "server": "gemini2api",
            "version": __version__,
            "key_required": key_required(),
            "api": {
                "protocol": "openai-completions",
                "base_path": "/v1",
                **capabilities,
                "context_window": context_window,
            },
            **pool.health(),
        }

    @app.get("/v1/models", dependencies=[Depends(require_key)])
    async def list_models(pool: Pool = Depends(get_pool)):
        now = int(time.time())
        data = []
        for model in pool.models():
            for suffix, name_suffix in (("", ""), (TEMPORARY_SUFFIX, " (temporary)")):
                data.append(
                    _model_entry(
                        model,
                        f"{model['name']}{suffix}",
                        f"{model['name']}{name_suffix}",
                        now,
                    )
                )
        return {"object": "list", "data": data}

    def _model_entry(model: dict[str, Any], model_id: str, display: str, now: int) -> dict:
        return {
            "id": model_id,
            "object": "model",
            "created": now,
            "owned_by": "google",
            # `name` stays the model id on purpose: a picker should show `gemini-neon`, not the
            # marketing phrase Google attaches to it ("Deeper analysis"), which says nothing
            # about which model is answering. The phrase is kept under `gemini.display_name`.
            "name": model_id,
            "context_window": context_window,
            **capabilities,
            "gemini": {
                "display_name": display,
                "description": model["description"],
                "advanced_only": model["advanced"],
                "temporary": model_id.endswith(TEMPORARY_SUFFIX),
                "accounts": model["accounts"],
            },
        }

    @app.get("/v1/quota", dependencies=[Depends(require_key)])
    async def quota(pool: Pool = Depends(get_pool)):
        """Per-account plan, quotas and usage, read from Google rather than estimated."""
        return pool.quota()

    @app.get("/v1/chats", dependencies=[Depends(require_key)])
    async def chats(pool: Pool = Depends(get_pool)):
        """Conversations this server is still continuing, so nothing is invisible."""
        return {
            "count": pool.chat_count(),
            "chat_mode": resolved.chat_mode,
            "auto_delete_ttl": resolved.auto_delete_ttl,
        }

    @app.post("/v1/keys", status_code=201, dependencies=[Depends(require_key)])
    async def mint_key(request: Request):
        """Issue a key for one client. The plaintext is returned once and stored nowhere."""
        body = await _optional_json_body(request)
        label = body.get("label")
        if label is not None and not isinstance(label, str):
            raise _error(400, "label must be a string")
        return minted.mint(label)

    @app.get("/v1/keys", dependencies=[Depends(require_key)])
    async def list_keys():
        return {"object": "list", "data": minted.keys()}

    @app.delete("/v1/keys/{key_id}", dependencies=[Depends(require_key)])
    async def revoke_key(key_id: str):
        if not minted.revoke(key_id):
            raise _error(404, f"no minted key with id {key_id!r}", "not_found_error")
        return {"revoked": key_id}

    @app.post("/v1/chat/completions", dependencies=[Depends(require_key)])
    async def chat_completions(request: Request, pool: Pool = Depends(get_pool)):
        body = await _json_body(request)
        stream = bool(body.get("stream"))
        result = await _run_turn(pool, body, endpoint="chat")
        if not stream:
            return _chat_response(result, body)
        return StreamingResponse(
            _chat_stream(result, body),
            media_type="text/event-stream",
            headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
        )

    @app.post("/v1/responses", dependencies=[Depends(require_key)])
    async def responses(request: Request, pool: Pool = Depends(get_pool)):
        body = await _json_body(request)
        stream = bool(body.get("stream"))
        normalised = _responses_to_chat(body)
        result = await _run_turn(pool, normalised, endpoint="responses")
        if not stream:
            return _responses_response(result, body)
        return StreamingResponse(
            _responses_stream(result, body),
            media_type="text/event-stream",
            headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
        )

    return app


async def _reap_loop(pool: Pool) -> None:
    """Expire idle chats and delete the ones this server opened in auto_delete mode."""
    while True:
        try:
            expired = await pool.reap_idle()
            if expired:
                logger.info(f"expired {expired} idle conversation(s)")
            removed = await pool.drain_deletions()
            if removed:
                logger.info(f"deleted {removed} conversation(s) from the Google account")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"cleanup pass failed: {exc}")
        await asyncio.sleep(min(3600, max(60, pool.settings.auto_delete_ttl // 2)))


async def _optional_json_body(request: Request) -> dict[str, Any]:
    """A body that may be absent or empty, which is how a bare mint request arrives."""
    raw = await request.body()
    if not raw.strip():
        return {}
    try:
        body = json.loads(raw)
    except ValueError as exc:
        raise _error(400, f"request body is not valid JSON: {exc}") from exc
    if not isinstance(body, dict):
        raise _error(400, "request body must be a JSON object")
    return body


async def _json_body(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except Exception as exc:
        raise _error(400, f"request body is not valid JSON: {exc}") from exc
    if not isinstance(body, dict):
        raise _error(400, "request body must be a JSON object")
    return body


def _responses_to_chat(body: dict[str, Any]) -> dict[str, Any]:
    """Translate one Responses request into the Chat Completions shape used internally."""
    raw_input = body.get("input")
    messages: list[dict[str, Any]] = []
    instructions = body.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        messages.append({"role": "system", "content": instructions})

    if isinstance(raw_input, str):
        messages.append({"role": "user", "content": raw_input})
    elif isinstance(raw_input, list):
        for item in raw_input:
            if not isinstance(item, dict):
                raise _error(400, "each input item must be an object")
            kind = item.get("type")
            if kind in {None, "message"}:
                messages.append(
                    {"role": item.get("role") or "user", "content": item.get("content")}
                )
            elif kind == "function_call":
                call = {
                    "id": item.get("call_id") or item.get("id"),
                    "type": "function",
                    "function": {
                        "name": item.get("name"),
                        "arguments": item.get("arguments") or "{}",
                    },
                }
                previous = messages[-1] if messages else None
                if previous is not None and previous.get("role") == "assistant":
                    previous.setdefault("tool_calls", []).append(call)
                else:
                    messages.append({"role": "assistant", "content": None, "tool_calls": [call]})
            elif kind == "function_call_output":
                output = item.get("output")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item.get("call_id"),
                        "content": output if isinstance(output, str) else json.dumps(output),
                    }
                )
            elif kind == "reasoning":
                continue  # thinking is never replayed to the web app
            else:
                raise _error(400, f"unsupported input item type: {kind!r}")
    else:
        raise _error(400, "input must be a string or a list of items")

    translated: dict[str, Any] = {
        "model": body.get("model"),
        "messages": messages,
        "stream": bool(body.get("stream")),
        "tools": body.get("tools"),
        "tool_choice": body.get("tool_choice"),
    }
    text_format = body.get("text")
    if isinstance(text_format, dict) and isinstance(text_format.get("format"), dict):
        translated["response_format"] = text_format["format"]
    if body.get("thinking") is not None:
        translated["thinking"] = body["thinking"]
    if body.get("reasoning") is not None:
        translated["thinking"] = body["reasoning"]
    return translated


def _temporary_flag(body: dict[str, Any], settings: Settings) -> bool:
    """Whether this turn should stay out of the account's Gemini history.

    Per request, so one client can mix: `"temporary": true` on a request, or the server-wide
    `chat_mode`. A temporary turn is never stored locally either, because the chat it opens
    cannot be continued anyway - Google closes the window at will and the context is lost.
    """
    raw = body.get("temporary")
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return settings.temporary


def _thinking_flag(body: dict[str, Any], settings: Settings) -> bool:
    """Whether this turn asks for extended thinking.

    Accepts `thinking: true/false`, `thinking: {enabled: ...}`, or the OpenAI `reasoning.effort`
    field, so a caller can turn it on per request instead of restarting the server.
    """
    raw = body.get("thinking")
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, dict):
        if isinstance(raw.get("enabled"), bool):
            return raw["enabled"]
        effort = raw.get("effort")
        if isinstance(effort, str):
            return effort.lower() not in {"none", "minimal", "low"}
    elif isinstance(raw, str):
        return raw.lower() not in {"none", "minimal", "low", "off", "false"}
    return settings.default_thinking


TEMPORARY_SUFFIX = "-temp"


def _model_name(body: dict[str, Any]) -> str:
    model = body.get("model")
    if not isinstance(model, str) or not model:
        raise _error(400, "model is required")
    if model.endswith(TEMPORARY_SUFFIX):
        # `gemini-neon-temp` keeps a model's behaviour and changes only where the chat is kept,
        # which is the one thing a client like Kip cannot express through the request body.
        body["temporary"] = True
        return model[: -len(TEMPORARY_SUFFIX)]
    return model


async def _run_turn(pool: Pool, body: dict[str, Any], *, endpoint: str) -> dict[str, Any]:
    """One Gemini turn: reuse the matching chat if it exists, otherwise open a new one."""
    model = _model_name(body)
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise _error(400, "messages is required")
    tools = prompts.normalise_tools(body.get("tools"))
    choice = prompts.normalise_tool_choice(body.get("tool_choice"))
    response_format = body.get("response_format")
    if response_format is not None and not isinstance(response_format, dict):
        raise _error(400, "response_format must be an object")

    try:
        rendered = prompts.build_prompt(
            messages,
            tools=tools,
            tool_choice=choice,
            response_format=response_format,
            max_chars=pool.settings.max_prompt_chars,
        )
    except prompts.PromptError as exc:
        raise _error(400, str(exc)) from exc

    temporary = _temporary_flag(body, pool.settings)
    stored = None if temporary else pool.find_chat(model, messages)
    if stored is not None:
        account_id = stored.account_id
        client = pool.client(account_id)
        metadata = stored.metadata
        # Only what this chat has not already been told is sent to it.
        fresh = messages[stored.matched_messages :]
        sent = prompts.build_prompt(
            fresh or messages[-1:],
            tools=tools,
            tool_choice=choice,
            response_format=response_format,
            max_chars=pool.settings.max_prompt_chars,
        )
    else:
        account_id, client = pool.next_client()
        metadata = None
        sent = rendered
    thinking = _thinking_flag(body, pool.settings)

    started = time.time()
    try:
        async with pool.lock(account_id):
            session = client.start_chat(metadata=metadata, model=model)
            output = await session.send_message(
                sent,
                temporary=temporary,
                extended_thinking=thinking,
            )
    except Exception as exc:
        if isinstance(exc, prompts.PromptError):
            raise _error(502, str(exc), "server_error") from exc
        logger.warning(f"{endpoint}: {account_id} failed: {exc}")
        raise _upstream_error(exc) from exc

    reply = prompts.parse_reply(output.text or "")
    try:
        prompts.check_calls(reply.calls, tools, choice)
        content = prompts.validate_structured(reply.text, response_format)
    except prompts.PromptError as exc:
        raise _error(502, str(exc), "server_error") from exc

    # A temporary chat is never recorded: continuing it is not possible, so claiming to hold it
    # would only mislead the next request into sending a fragment of the conversation.
    if temporary:
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "created": int(started),
            "model": model,
            "account": account_id,
            "content": content,
            "calls": reply.calls,
            "thoughts": output.thoughts or None,
            "citations": [],
            "images": [image.url for image in (output.images or []) if getattr(image, "url", None)],
            "elapsed": round(time.time() - started, 3),
            "prompt_chars": len(sent),
            "reply_chars": len(reply.text),
        }

    # Record this turn's whole history, not only the first one: each later turn then matches a
    # longer prefix, so the amount sent to Google keeps shrinking as a conversation grows.
    metadata = list(session.metadata)
    pool.store_chat(
        history_key(model, messages),
        StoredChat(
            account_id=account_id,
            metadata=metadata,
            model=model,
            turns=(stored.turns + 1) if stored else 1,
            auto_delete=pool.settings.chat_mode == "auto_delete",
        ),
    )
    if stored is not None and history_key(model, messages) != history_key(
        model, messages[: stored.matched_messages]
    ):
        stored.metadata = metadata
        stored.turns += 1

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "created": int(started),
        "model": model,
        "account": account_id,
        "content": content,
        "calls": reply.calls,
        "thoughts": output.thoughts or None,
        "citations": [
            {"title": citation.title, "url": citation.url}
            for citation in (output.citations or [])
            if getattr(citation, "url", None)
        ],
        "images": [image.url for image in (output.images or []) if getattr(image, "url", None)],
        "elapsed": round(time.time() - started, 3),
        # Character counts, not tokens: the web app reports no usage at all.
        "prompt_chars": len(sent),
        "reply_chars": len(reply.text),
    }


def _chat_usage(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt_tokens": result["prompt_chars"],
        "completion_tokens": result["reply_chars"],
        "total_tokens": result["prompt_chars"] + result["reply_chars"],
    }


def _chat_response(result: dict[str, Any], body: dict[str, Any]) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": result["content"] or None}
    if result["calls"]:
        message["tool_calls"] = result["calls"]
    if result["thoughts"]:
        message["reasoning_content"] = result["thoughts"]
    if result["citations"]:
        message["annotations"] = [
            {
                "type": "url_citation",
                "url_citation": {"url": item["url"], "title": item.get("title") or ""},
            }
            for item in result["citations"]
        ]
    return {
        "id": result["id"],
        "object": "chat.completion",
        "created": result["created"],
        "model": result["model"],
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if result["calls"] else "stop",
            }
        ],
        "usage": _chat_usage(result),
        "system_fingerprint": f"gemini2api:{result['account']}",
    }


async def _chat_stream(result: dict[str, Any], body: dict[str, Any]):
    """Emit the reply as SSE. The turn has already finished, so this never half-fails."""
    base = {
        "id": result["id"],
        "object": "chat.completion.chunk",
        "created": result["created"],
        "model": result["model"],
    }

    def chunk(delta: dict[str, Any], finish: str | None = None, **extra: Any) -> str:
        payload = {
            **base,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            **extra,
        }
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    yield chunk({"role": "assistant", "content": ""})
    if result["thoughts"]:
        yield chunk({"reasoning_content": result["thoughts"]})
    for index, call in enumerate(result["calls"]):
        yield chunk({"tool_calls": [{"index": index, **call}]})
    if result["content"]:
        yield chunk({"content": result["content"]})
    if result["citations"]:
        yield chunk(
            {
                "annotations": [
                    {
                        "type": "url_citation",
                        "url_citation": {"url": item["url"], "title": item.get("title") or ""},
                    }
                    for item in result["citations"]
                ]
            }
        )
    include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
    final: dict[str, Any] = {}
    if include_usage:
        final["usage"] = _chat_usage(result)
    yield chunk({}, "tool_calls" if result["calls"] else "stop", **final)
    if include_usage:
        # OpenAI sends usage in a trailing chunk with an empty choices list.
        yield f"data: {json.dumps({**base, 'choices': [], 'usage': _chat_usage(result)})}\n\n"
    yield SSE_DONE


def _responses_output(result: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if result["thoughts"]:
        items.append(
            {
                "id": f"rs_{uuid.uuid4().hex[:24]}",
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": result["thoughts"]}],
            }
        )
    if result["content"]:
        items.append(
            {
                "id": f"msg_{uuid.uuid4().hex[:24]}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": result["content"], "annotations": []}],
            }
        )
    for call in result["calls"]:
        items.append(
            {
                "id": f"fc_{uuid.uuid4().hex[:24]}",
                "type": "function_call",
                "call_id": call["id"],
                "name": call["function"]["name"],
                "arguments": call["function"]["arguments"],
                "status": "completed",
            }
        )
    return items


def _responses_response(result: dict[str, Any], body: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"resp_{uuid.uuid4().hex}",
        "object": "response",
        "created_at": result["created"],
        "model": result["model"],
        "status": "completed",
        "output": _responses_output(result),
        "usage": _chat_usage(result),
        "metadata": body.get("metadata") or {},
    }


async def _responses_stream(result: dict[str, Any], body: dict[str, Any]):
    """The Responses event sequence: item events first, then one terminal `response.completed`."""
    response = _responses_response(result, body)

    def event(name: str, payload: dict[str, Any]) -> str:
        return f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    created = {**response, "status": "in_progress", "output": []}
    yield event("response.created", {"type": "response.created", "response": created})
    for index, item in enumerate(response["output"]):
        yield event(
            "response.output_item.added",
            {"type": "response.output_item.added", "output_index": index, "item": item},
        )
        if item["type"] == "message":
            text = item["content"][0]["text"]
            yield event(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "item_id": item["id"],
                    "output_index": index,
                    "content_index": 0,
                    "delta": text,
                },
            )
        yield event(
            "response.output_item.done",
            {"type": "response.output_item.done", "output_index": index, "item": item},
        )
    yield event("response.completed", {"type": "response.completed", "response": response})
