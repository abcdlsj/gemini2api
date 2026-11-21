# gemini2api

The Gemini web app (gemini.google.com) behind an OpenAI-compatible API, built for one thing the
existing wrappers leave out: **seeing the quota**.

It runs your signed-in subscription account through
[`gemini-webapi`](https://github.com/HanaokaYuzu/Gemini-API) and exposes:

| Route | What it does |
|---|---|
| `POST /v1/chat/completions` | OpenAI Chat Completions, streaming and tools |
| `POST /v1/responses` | OpenAI Responses |
| `GET /v1/models` | Every model **your account** offers, with tier info |
| `GET /v1/quota` | Plan, quota windows and usage, read from Google |
| `GET /v1/chats` | How many conversations are being continued |
| `GET /health` | Sign-in state, plus the version and capabilities a client negotiates with |
| `POST /v1/keys` | Mint a key for one client (authenticated by an existing key) |
| `GET /v1/keys` | Minted keys, prefixes only |
| `DELETE /v1/keys/{id}` | Revoke one |

## Why another one

| | This | Wrapping an existing gateway |
|---|---|---|
| Quota visibility | `GET /v1/quota`, straight from Google | usually absent |
| Thinking per request | `"thinking": true` / `reasoning.effort` | global flag, restart to change |
| Conversation reuse | continues the Google-side chat and sends only messages it has not seen | often resends the whole history |
| Account history | `auto_delete` keeps it clean by removing chats once idle | leaks into your history |
| Code size | ~900 lines you can read | 10k+ lines, vendored protocol |

## Install

```sh
git clone git@github.com:abcdlsj/gemini2api.git
cd gemini2api
uv sync --extra dev
```

## Cookies

The Gemini web app authenticates with cookies, not an API key. **Use Firefox**, and let the server
read them for you:

```sh
# sign in to gemini.google.com in Firefox first
uv run python -m gemini2api.cookies --write --proxy http://127.0.0.1:7890
```

That writes every Google cookie the browser holds into `config.yaml` (mode `600`) and clears the
rest.

Three things this saves you from, all measured:

- **The whole cookie jar is required.** A request carrying only `__Secure-1PSID` and
  `__Secure-1PSIDTS`, or the classic `SID`/`SAPISID` set, is redirected to ServiceLogin even
  though the browser session is valid. The library's docs still say two cookies are enough; they
  are not.
- **Chrome cannot be used.** Chromium's Device Bound Session Credentials pin the session to the
  browser that created it, so a copied cookie is rejected immediately — not hours later. Firefox
  does not implement DBSC.
- **The proxy must be set.** `gemini.google.com` needs it, and `gemini-webapi`'s own redirect
  policy reports a proxied request as `curl: (7) Redirect to internal IP 127.0.0.1 rejected
  (SSRF protection)`, which looks like a cookie problem but is not.

Manual alternative, if you would rather not have the tool read your profile: copy the full
`Cookie` request header from a `gemini.google.com` request (DevTools → Network) and put it in
`cookies:` as `name=value; name=value`.

## Configure

```sh
cp config.example.yaml config.yaml
chmod 600 config.yaml
$EDITOR config.yaml
```

```yaml
accounts:
  - id: ultra
    secure_1psid: "g.a000..."
    secure_1psidts: "sidts-Cjc..."

api_key: "sk-local-something"
chat_mode: auto_delete
```

Every value can also come from the environment, which keeps cookies out of files:

```sh
export G2A_SECURE_1PSID=...
export G2A_SECURE_1PSIDTS=...
export G2A_API_KEY=sk-local-something
```

## Verify before serving

```sh
uv run python -m gemini2api.doctor
```

Runs in about ten seconds. It reports the account, the four models Google actually offers it
(`gemini-neon`, `gemini-carbon`, `gemini-4-hydrogen`, `gemini-auto`), and the quota windows.

Prints each account's sign-in state, the models it can actually use, which of them are
subscription-only, and the quota windows. Nothing is generated, so it costs no quota — if the
list is empty or every model is `advanced=False`, the cookies are wrong or the account is not the
paid one.

## Run

```sh
uv run python -m gemini2api
```

```sh
curl -s localhost:8790/v1/models -H "authorization: Bearer $G2A_API_KEY" | jq
curl -s localhost:8790/v1/quota  -H "authorization: Bearer $G2A_API_KEY" | jq
```

`/v1/quota` reports the plan Google has on the account and every quota window:

```json
{
  "firefox": {
    "signed_in": true,
    "plan": "ULTRA",
    "plan_id": 6,
    "windows": {
      "current_5h": { "usage_percentage": 5, "remaining_credits": 11375,
                      "reset_at": "2026-11-20T23:18:25+08:00" },
      "weekly":     { "usage_percentage": 0, "remaining_credits": 241202,
                      "reset_at": "2026-11-27T11:18:25+08:00" }
    }
  }
}
```

The plan label comes from Google's `usage_info.tier`, not from which models happen to be
listed — an Ultra account and a free account can both see `gemini-neon`, and only the tier
tells them apart.

Then point any OpenAI client at `http://127.0.0.1:8790/v1` with the model name from `/v1/models`.

## Keys

`api_key` is the operator's key. A client — Kip, for instance — does not have to keep it: it
presents it once and mints its own.

```sh
curl -s -X POST localhost:8790/v1/keys \
  -H "authorization: Bearer $G2A_API_KEY" \
  -H 'content-type: application/json' \
  -d '{"label":"kip"}'
# {"id":"3f2a...","label":"kip","prefix":"g2k-AbCdEf","created_at":1770000000,"key":"g2k-..."}
```

The plaintext is returned once and never stored — only its SHA-256, in
`state_dir/keys.json` (mode 600). Minted keys are accepted anywhere `api_key` is, and
`DELETE /v1/keys/{id}` revokes one without disturbing the others. Minting itself requires a
key that already works: loopback is not a way to hand yourself quota.

Once a key exists, `/health` reports `key_required: true` and every `/v1` route needs one.

## Negotiation

A client that finds this server can configure itself from two responses — nothing has to be
typed in by hand except the port:

```json
// GET /health
{"server": "gemini2api", "version": "0.1.0", "key_required": true,
 "api": {"protocol": "openai-completions", "base_path": "/v1",
         "supports_vision": false, "supports_reasoning": true, "context_window": 225000},
 "accounts": {"ultra": {"signed_in": true, "status": "AVAILABLE"}}, "chats": 0, "closed": 0}
```

`GET /v1/models` carries the same `context_window`, `supports_vision` and
`supports_reasoning` per model, so a client can advertise them without inventing numbers.
`supports_vision` is `false` on purpose: image parts are parsed and then dropped, so claiming
otherwise would make a client send images that never reach Gemini. `context_window` is
`max_prompt_chars / 4`, the budget the server itself enforces.

## Thinking control

Per request, no restart:

```json
{"model": "...", "thinking": true, "messages": [...]}
{"model": "...", "reasoning": {"effort": "high"}, "messages": [...]}
{"model": "...", "reasoning": {"effort": "low"}, "messages": [...]}
```

`thinking: true` maps to Gemini's extended thinking. `effort` of `none`/`minimal`/`low` turns it
off, anything higher turns it on.

## Tool calling

Gemini's web app has no function-calling channel and no constrained decoding, so tools are taught
as a fenced block:

````
```tool_call
{"name":"read_file","arguments":{"path":"a.txt"}}
```
````

A block becomes a `tool_calls` entry only if the fence is complete, the JSON parses, the name was
declared in this request, and the arguments are an object. Anything else stays as text — a
truncated call is never executed. `tool_choice: required` that the model ignores is a `502`, not a
silently empty turn.

## Chat modes

| `chat_mode` | Google account history | Context across turns |
|---|---|---|
| `normal` | kept | reliable |
| `temporary` | never saved | **best effort** — Google may drop the window silently |
| `auto_delete` (default) | kept, then deleted once idle | reliable |

`auto_delete` uses normal chats so context is reliable, then removes them after
`auto_delete_ttl` seconds idle. Only chats this server opened are ever deleted.

## Honest limits

- **No image input.** `gemini-webapi` names every in-memory upload `input_NNNNNNN.txt` and
  derives the content type from that, so an image is announced to Google as `text/plain` and the
  generation request never answers. `/v1/models` therefore reports `supports_vision: false`, and a
  turn that carries an image tells the model one was attached instead of pretending to see it.
- **Token counts are characters.** The web app reports no usage, so `usage` fields are character
  counts. Fine for context budgeting, wrong for billing.
- **`temperature`, `top_p`, `max_tokens` do nothing.** The web app exposes no sampling controls;
  they are accepted so clients do not break.
- **No model downgrade check.** If Google quietly answers with a weaker model, this server cannot
  tell. Confirm against `/v1/quota` and the web UI when it matters.
- **Cookies expire.** This is the ceiling of every web-app wrapper, not a bug here.

## License

MIT for this code. It depends on `gemini-webapi`, which is **AGPL-3.0** — running this as a
network service for other people carries that library's obligations. Personal use is unaffected.
