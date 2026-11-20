"""Turn OpenAI-shaped messages into one Gemini prompt, and its answer back into tool calls.

The web app has no role channel and no constrained decoding: it accepts one user turn. So the
conversation is rendered as a marked transcript inside that turn, and function declarations are
taught as a fenced block the model is asked to emit. Both directions are strictly validated here
rather than trusted.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

TOOL_FENCE = "tool_call"
MAX_TOOL_CALLS = 32
MAX_TOOL_BYTES = 256 * 1024

USER_MARK = "[User]"
ASSISTANT_MARK = "[Assistant]"
RESULT_MARK = "[Tool result]"
TOOL_CALL_MARK = "[tool calls above were executed by the caller; their results follow]"


class PromptError(ValueError):
    """The request cannot be rendered into a prompt, or the reply fails validation."""


@dataclass
class Reply:
    """One parsed assistant turn."""

    text: str = ""
    calls: list[dict[str, Any]] = field(default_factory=list)


def _content_text(content: Any) -> str:
    """Flatten OpenAI content, which is a string or a list of typed parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise PromptError("message content must be a string or a list of parts")
    pieces: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            raise PromptError("content part must be an object")
        kind = part.get("type")
        if kind in {"text", "input_text", "output_text"}:
            pieces.append(part.get("text") or "")
        elif kind in {"image_url", "input_image"}:
            continue  # handled by the caller, which needs the bytes, not the text
        elif kind in {"file", "input_file"} or kind in {"refusal", "reasoning"}:
            continue
        else:
            raise PromptError(f"unsupported content part type: {kind!r}")
    return "\n".join(pieces)


def image_urls(content: Any) -> list[str]:
    """The image URLs in one message, in order."""
    if not isinstance(content, list):
        return []
    found: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "image_url":
            image = part.get("image_url")
            url = image.get("url") if isinstance(image, dict) else image
            if isinstance(url, str) and url:
                found.append(url)
        elif part.get("type") == "input_image":
            url = part.get("image_url")
            if isinstance(url, str) and url:
                found.append(url)
    return found


def build_tool_instruction(tools: list[dict[str, Any]], tool_choice: str | None) -> str:
    """The system-level explanation of the caller's function protocol."""
    tools = normalise_tools(tools)
    if not tools:
        return ""
    declarations = [
        {
            "name": tool["name"],
            "description": tool.get("description") or "",
            "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
        }
        for tool in tools
    ]
    lines = [
        "# Local tools",
        "These functions run on the caller's machine. You cannot run them yourself.",
        json.dumps(declarations, ensure_ascii=False, separators=(",", ":")),
        "",
        "To use one, reply with ONLY this block and nothing else, no code fence language tag:",
        "```" + TOOL_FENCE,
        '{"name":"<tool name>","arguments":{<json object matching that tool parameters>}}',
        "```",
        "Rules: emit one block per call, at most "
        f"{MAX_TOOL_CALLS} per reply. `arguments` must be a JSON object. Never invent a tool "
        "name, never guess a result, and never write the caller's results yourself. Stop right "
        "after your last block so the caller can run them.",
        "For an example of the syntax in prose, use a `json` fence instead.",
    ]
    if tool_choice == "required":
        lines.append("You MUST request at least one of the declared functions this turn.")
    elif isinstance(tool_choice, str) and tool_choice not in {"auto", "none", "required"}:
        lines.append(f"You MUST request only `{tool_choice}` this turn.")
    elif tool_choice == "none":
        lines.append("You MUST NOT request any function this turn; answer in prose.")
    return "\n".join(lines)


def build_prompt(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | None = None,
    response_format: dict[str, Any] | None = None,
    max_chars: int = 900_000,
) -> str:
    """One Gemini user turn carrying the whole conversation.

    Nothing is summarised or dropped: a truncated transcript would let the model answer a
    different question than the one asked, so an oversized request is refused instead.
    """
    system_parts: list[str] = []
    transcript: list[str] = []
    # A call id belongs to the assistant message that issued it; the results that follow refer
    # back to it by id only, so the ids are kept alongside the transcript for the model.
    for message in messages:
        if not isinstance(message, dict):
            raise PromptError("each message must be an object")
        role = message.get("role")
        text = _content_text(message.get("content"))
        if role in {"system", "developer"}:
            if text.strip():
                system_parts.append(text)
            continue
        if role == "user":
            transcript.append(f"{USER_MARK}\n{text}")
            continue
        if role == "assistant":
            calls = message.get("tool_calls") or []
            rendered = text
            if calls:
                payload = [
                    {
                        "id": call.get("id"),
                        "name": (call.get("function") or {}).get("name"),
                        "arguments": (call.get("function") or {}).get("arguments"),
                    }
                    for call in calls
                ]
                rendered = (rendered + "\n" if rendered else "") + TOOL_CALL_MARK + "\n"
                rendered += json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            transcript.append(f"{ASSISTANT_MARK}\n{rendered}")
            continue
        if role == "tool":
            call_id = message.get("tool_call_id") or ""
            name = message.get("name") or ""
            # JSON quoting keeps caller text from forging a role marker of its own.
            transcript.append(
                f"{RESULT_MARK}\n"
                + json.dumps(
                    {"tool_call_id": call_id, "name": name, "content": text},
                    ensure_ascii=False,
                )
            )
            continue
        raise PromptError(f"unsupported role: {role!r}")

    if not transcript:
        raise PromptError("at least one user or assistant message is required")

    preamble = [part for part in system_parts if part.strip()]
    instruction = build_tool_instruction(tools or [], tool_choice)
    if instruction:
        preamble.append(instruction)
    structured = build_structured_instruction(response_format)
    if structured:
        preamble.append(structured)

    prompt = ""
    if preamble:
        prompt += "[System instructions]\n" + "\n\n".join(preamble) + "\n\n"
    prompt += "[Conversation]\n" + "\n\n".join(transcript)
    prompt += f"\n\n{ASSISTANT_MARK}\n"

    if len(prompt) > max_chars:
        raise PromptError(
            f"prompt is {len(prompt)} characters, over the {max_chars} limit; "
            "shorten the conversation or raise max_prompt_chars"
        )
    if tools and not any(m.get("role") == "user" for m in messages if isinstance(m, dict)):
        raise PromptError("a request with tools needs at least one user message")
    return prompt


def build_structured_instruction(response_format: dict[str, Any] | None) -> str:
    """Tell the model the exact JSON shape the caller will validate against."""
    if not response_format:
        return ""
    kind = response_format.get("type")
    if kind == "json_object":
        return "# Output format\nReply with one JSON object and nothing else. No prose, no fences."
    if kind != "json_schema":
        return ""
    spec = response_format.get("json_schema") or {}
    schema = spec.get("schema")
    if schema is None and "properties" in spec:
        schema = spec
    if schema is not None:
        try:
            rendered = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise PromptError(f"response_format schema is not serialisable: {exc}") from exc
    else:
        rendered = '{"type":"object"}'
    name = spec.get("name") or "response"
    lines = [
        "# Output format",
        f"Reply with one JSON value matching the schema `{name}` and nothing else.",
        "No prose, no markdown fence, no explanation.",
        rendered,
    ]
    return "\n".join(lines)


_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})[ \t]*([A-Za-z0-9_+-]*)[ \t]*$")


def _strip_fence(text: str) -> str:
    """Drop one wrapping markdown fence, when the whole reply is inside it."""
    stripped = text.strip()
    match = _FENCE_RE.match(stripped.splitlines()[0] if stripped else "")
    if not match:
        return stripped
    opening = match.group(1)
    lines = stripped.splitlines()
    if not lines or not lines[-1].strip().startswith(opening[0] * len(opening)):
        return stripped
    return "\n".join(lines[1:-1]).strip()


def parse_reply(text: str) -> Reply:
    """Split a reply into prose and validated tool calls.

    A block becomes a call only if it is a complete fence holding one object with a string
    `name` and an object `arguments`. Anything else stays as text, so a half-written call can
    never be mistaken for a request to run something.
    """
    if not isinstance(text, str):
        raise PromptError("reply must be text")
    lines = text.splitlines()
    prose: list[str] = []
    calls: list[dict[str, Any]] = []
    index = 0
    while index < len(lines):
        match = _FENCE_RE.match(lines[index])
        if not match or match.group(2).strip() != TOOL_FENCE:
            prose.append(lines[index])
            index += 1
            continue
        opening = match.group(1)
        closing = None
        for probe in range(index + 1, len(lines)):
            tail = lines[probe].strip()
            if tail.startswith(opening[0] * len(opening)):
                closing = probe
                break
        if closing is None:
            # An unterminated block is dropped rather than run: it may be truncated output.
            prose.extend(lines[index:])
            break
        body = "\n".join(lines[index + 1 : closing]).strip()
        calls.append(_parse_call(body, len(calls)))
        index = closing + 1
    merged = "\n".join(prose).strip()
    return Reply(text=merged, calls=calls)


def _parse_call(body: str, slot: int) -> dict[str, Any]:
    if len(body.encode("utf-8")) > MAX_TOOL_BYTES:
        raise PromptError("tool call block is too large")
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise PromptError(f"tool call block is not valid JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise PromptError("tool call block must be a JSON object")
    extra = set(parsed) - {"name", "arguments"}
    if extra:
        raise PromptError(f"tool call block has unexpected keys: {sorted(extra)}")
    name = parsed.get("name")
    arguments = parsed.get("arguments")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
        raise PromptError(f"tool call name is not usable: {name!r}")
    if not isinstance(arguments, dict):
        raise PromptError("tool call arguments must be a JSON object")
    return {
        "id": f"call_{slot:02d}_{abs(hash(body)) % (10**12):012d}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
        },
    }


def check_calls(
    calls: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    tool_choice: str | None,
) -> None:
    """Refuse a reply whose calls do not match what this request declared."""
    if len(calls) > MAX_TOOL_CALLS:
        raise PromptError(
            f"reply requested {len(calls)} tool calls, over the {MAX_TOOL_CALLS} limit"
        )
    declared = {tool["name"] for tool in normalise_tools(tools)}
    for call in calls:
        name = call["function"]["name"]
        if name not in declared:
            raise PromptError(f"reply requested undeclared tool {name!r}")
        if tool_choice not in {None, "auto", "required"} and name != tool_choice:
            raise PromptError(f"reply requested {name!r} but tool_choice forced {tool_choice!r}")
    if tool_choice == "none" and calls:
        raise PromptError("reply requested a tool while tool_choice was 'none'")
    if tool_choice == "required" and not calls:
        raise PromptError("tool_choice was 'required' but the reply requested no tool")


def validate_structured(text: str, response_format: dict[str, Any] | None) -> str:
    """Return the canonical JSON payload, or raise if the reply does not conform."""
    if not response_format:
        return text
    kind = response_format.get("type")
    if kind not in {"json_object", "json_schema"}:
        return text
    body = _strip_fence(text)
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise PromptError(f"reply is not valid JSON: {exc.msg}") from exc
    if kind == "json_object" and not isinstance(payload, dict):
        raise PromptError("json_object requires a JSON object reply")
    if kind == "json_schema":
        spec = response_format.get("json_schema") or {}
        schema = spec.get("schema")
        if schema is None and "properties" in spec:
            schema = spec
        if schema is not None:
            from jsonschema import ValidationError
            from jsonschema import validate as jsonschema_validate

            try:
                jsonschema_validate(instance=payload, schema=schema)
            except ValidationError as exc:
                raise PromptError(f"reply does not match the schema: {exc.message}") from exc
            except Exception as exc:
                raise PromptError(f"the request's schema could not be evaluated: {exc}") from exc
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def normalise_tools(raw: Any) -> list[dict[str, Any]]:
    """Flatten declared functions, accepting both the flat and the nested `function` shape."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise PromptError("tools must be a list")
    flattened: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise PromptError("each tool must be an object")
        if item.get("type") not in {None, "function"}:
            raise PromptError(f"unsupported tool type: {item.get('type')!r}")
        spec = item.get("function") if isinstance(item.get("function"), dict) else item
        name = spec.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
            raise PromptError(f"tool name is not usable: {name!r}")
        if name in seen:
            raise PromptError(f"tool {name!r} is declared twice")
        seen.add(name)
        parameters = spec.get("parameters")
        if parameters is not None and not isinstance(parameters, dict):
            raise PromptError(f"tool {name!r} parameters must be an object")
        flattened.append(
            {
                "name": name,
                "description": spec.get("description"),
                "parameters": parameters or {"type": "object", "properties": {}},
            }
        )
    return flattened


def normalise_tool_choice(raw: Any) -> str | None:
    """`auto`, `none`, `required`, or the name of one forced function."""
    if raw is None:
        return None
    if isinstance(raw, str):
        if raw in {"auto", "none", "required"}:
            return raw
        raise PromptError(f"unsupported tool_choice: {raw!r}")
    if isinstance(raw, dict):
        spec = raw.get("function") if isinstance(raw.get("function"), dict) else raw
        name = spec.get("name")
        if isinstance(name, str) and name:
            return name
        raise PromptError("tool_choice object needs a function name")
    raise PromptError("tool_choice must be a string or an object")
