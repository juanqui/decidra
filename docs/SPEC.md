# jev-engine API specification

This document is the contract implemented by `src/`. It ports the behavior of
`/srv/models/jev-awq/decide_shim.py` for the `JEV-27B-VL` AWQ W4A16 artifact served by
vLLM as `jev-awq` (default `http://127.0.0.1:8002`). Where this document and the Python
shim differ, the difference is called out explicitly.

## 1. Conventions

- All request and response bodies are UTF-8 JSON.
- Model ids: the client-facing name is `autotrust/JEV-27B-VL`; `/v1/models` also lists
  `autotrust/JEV-27B` and the configured vLLM id (`jev-awq`).
- The wire protocol string is always `jev27-bare-v1`.
- Probabilities are floats in `[0, 1]` and sum to 1 up to floating-point error.
- `state` is a string, a JSON object, or an array of content parts.
- A client request that fails validation returns **400**.
- An upstream vLLM HTTP 4xx/5xx returns **502**; an upstream timeout returns **504**.
- Malformed JSON, wrong field types, and unsupported enum values never panic; they return
  structured errors.

### Error envelope

```json
{
  "detail": "human-readable message",
  "error": {
    "message": "human-readable message",
    "type": "invalid_request_error | upstream_error | upstream_timeout | internal_error",
    "code": 400
  }
}
```

`detail` is included for compatibility with clients written against FastAPI's
`HTTPException`. Streaming passthrough does not use this envelope; it relays the upstream
status and bytes.

## 2. States and content parts

### Text state

- A JSON string is used verbatim.
- A JSON object is rendered with CPython-compatible
  `json.dumps(value, ensure_ascii=False)`: insertion order, `", "` between members,
  `": "` between key and value, raw non-ASCII, Python float notation.
- Any other scalar is rejected with 400 (`state must be a string, an object, or an array`).

### List state (`_parts`)

For each element in order:

| Element | Output part |
|---|---|
| string | `{"type":"text","text": value}` |
| object containing `image` | `{"type":"image_url","image_url":{"url": value.image}}` |
| object with `type == "image_url"` | the object unchanged |
| object with `type == "text"` | the object unchanged |
| anything else (number, bool, null, array, object) | `{"type":"text","text": json.dumps(value)}` |

The list is considered an image request if at least one part is an image. At most one image
is accepted; more than one returns 400
`this backend reads one image per request, got N`.

### Prompt

The exact text prompt, used by `/v1/completions` for every text-only pass:

```text
[kind] {kind}
[state] {state}
[question] {question}
[options]
{lines}
[decision]:
```

- `{kind}` is `noul`, `score`, or `choice`.
- `{state}` is the concatenation of all text parts with no separator.
- For `choice`, `{lines}` is `A) option0\nB) option1\n...`, using the label table below.
- For `noul` and `score`, `{lines}` is the options joined with `\n`.

For image requests the same three sections are represented as chat content parts:

```text
[{"type":"text","text":"[kind] ...\n[state] "},
 ...state parts...,
 {"type":"text","text":"\n[question] ...\n[options]\n...\n[decision]:"}]
```

### Raw chat template

Image requests send this per-request `chat_template` string:

```text
{%- for m in messages -%}{%- if m['content'] is string -%}{{ m['content'] }}{%- else -%}
{%- for c in m['content'] -%}{%- if c['type'] == 'text' -%}{{ c['text'] }}{%- else -%}
<|vision_start|><|image_pad|><|vision_end|>{%- endif -%}{%- endfor -%}{%- endif -%}
{%- endfor -%}
```

(without the line breaks shown here for readability). It is sent with
`add_generation_prompt: false`.

## 3. Labels

`LABELS` is `A..Z` followed by `AA..ZZ` in lexicographic product order. The first 256 labels
that satisfy both conditions are used:

1. `tokenizer.encode(label, add_special_tokens=False)` has exactly one id.
2. That id occurs in
   `tokenizer.encode("x\n" + label + ") y", add_special_tokens=False)`.

The first 16 **must** equal `decision_head.verbalizer_ids[8:24]`; startup fails otherwise.
With the production Qwen tokenizer, ids are `32..57` (A-Z), then `AA..` tokens, for a total
of 256 labels. `/v1/decide/info` reports the actual count as `max_options`.

## 4. Decision head and probabilities

Inputs:

- `bias[]` and `verbalizer_ids[]` from `decision_head.json`.
- Slot ranges `[lo, hi)` for `noul` (2), `score` (6), and `choice` (16 trained slots).
- `per_kind` temperatures from `calibration.json`.

For one pass over options:

- `choice`: option `i` uses token id `labels[i]`. Bias is `bias[lo+i]` for `i < 16`
  (i.e. while `lo+i < hi`), otherwise `0.0`.
- `noul`/`score`: option `i` uses `verbalizer_ids[lo+i]` and `bias[lo+i]`.

The upstream returns logprobs keyed by token id. Missing ids use `-1e9`, and values are
clamped from below at `-1e9`:

```text
scaled[i] = (max(logprob(token_i), -1e9) + bias_i) / T_kind
```

If `JEV_CALIBRATION_FITTED` is configured and contains `kind`, the shim's transform is
applied to raw head logits:

```text
raw[i]    = scaled[i] * T_kind
scaled[i] = raw[i] * scale + fit_bias[i]
```

If the fit's bias vector is shorter than the option count, missing entries are `0.0` (the
shim would index out of range; this server does not panic). Then:

```text
p[i] = exp(scaled[i] - max(scaled)) / sum_j exp(scaled[j] - max(scaled))
```

Tie-breaking for the reported `choice_index` is Python's `max(range(n), key=...)`: the lowest
index wins on exact ties.

## 5. Upstream read-out

### Text

For each block of at most 128 ids, `POST /v1/completions` with:

```json
{
  "model": "jev-awq",
  "prompt": "<exact prompt>",
  "max_tokens": 1,
  "temperature": 1.0,
  "top_p": 1.0,
  "top_k": 0,
  "min_p": 0.0,
  "repetition_penalty": 1.0,
  "add_special_tokens": false,
  "return_tokens_as_token_ids": true,
  "logprobs": 128,
  "logprob_token_ids": [3721, 1802]
}
```

`logprobs` is the chunk length. The parsed logprobs are
`choices[0].logprobs.top_logprobs[0]`, an object whose keys are `"token_id:<id>"`.
Usage is accumulated from `usage.prompt_tokens` and `usage.completion_tokens`; each block
counts as one model request.

### Image

`POST /v1/chat/completions` with the same sampling fields plus:

```json
{
  "messages": [{"role": "user", "content": [...]}],
  "chat_template": "<raw template>",
  "add_generation_prompt": false,
  "logprobs": true,
  "top_logprobs": 1,
  "logprob_token_ids": [3721, 1802]
}
```

Logprobs are parsed from
`choices[0].logprobs.content[0].top_logprobs`, an array of
`{"token": "token_id:<id>", "logprob": <float>}`. Usage accounting is identical.

Any malformed upstream shape (missing choice, non-numeric logprob, malformed token key,
non-JSON body) returns 502 with `invalid backend response`.

## 6. `POST /v1/decide`

### Request

```json
{
  "kind": "noul | score | choice",
  "state": "text | object | array",
  "question": "string",
  "options": ["..."],
  "strategy": "auto | single | tournament | permute",
  "model": "ignored"
}
```

- `state` defaults to `""`; `strategy` defaults to `auto`; `model` is ignored.
- `kind` must be one of the three values.
- `choice`: 2 to `max_options` options are required.
- `noul`: options are exactly `["false","true"]`; any supplied `options` are ignored.
- `score`: options are exactly `["0","1","2","3","4","5"]`.
- More than one image part is a 400.

### Strategy resolution

```text
effective = auto ? tournament : strategy
single_pass if kind != choice OR len(options) <= 16 OR effective == single
otherwise permute or tournament
```

`adaptation` is `"native"` when `kind != choice` or `len(options) <= 16`; otherwise it is the
effective strategy (`single`, `tournament`, or `permute`).

### Tournament

Let `n = len(options)`, `g = ceil(n / 16)`, `base = floor(n / g)`, `extra = n mod g`. Group
sizes are `base + 1` for the first `extra` groups and `base` for the rest; options are
assigned in order.

```text
in_group[o] = probability of option o inside its group pass
chosen      = for each group, argmax(in_group[o], tie: lowest o)
rest        = all non-chosen options sorted by (-in_group[o], o)
finalists   = sorted(chosen + first max(0, 16 - len(chosen)) of rest)
final[o]    = probability of finalist o in the final pass over finalists

share[g] = sum(final[f] for finalists f in group g)
cap[g]   = sum(in_group[f] for finalists f in group g)
among    = sum(share[g] * cap[g])

p[o] = final[o] * among               if o is a finalist
p[o] = share[group(o)] * in_group[o]  otherwise
p    = p / sum(p)
```

Group passes and the final pass use the same concurrency semaphore. Request count:
`g + 1`.

### Permute

`seed = zlib.crc32(question.encode("utf-8"))` (CRC-32/ISO-HDLC). Using CPython's
`random.Random(seed)` semantics:

```text
orders = [identity] + [rng.sample(range(n), n) for _ in range(3)]
```

Each order is evaluated over the correspondingly permuted option list. Probabilities are
mapped back to original indices:

```text
p[original_index] = sum over the 4 orders of permuted_probability / 4
```

No additional normalization is applied by the shim. Request count: 4. The MT19937 seeding,
`getrandbits`, `_randbelow`, and selection-sampling pool algorithm are reproduced exactly;
unit tests compare full permutations against CPython for `n = 17` and `n = 32`.

### Response

```json
{
  "kind": "choice",
  "effective_kind": "choice",
  "options": ["..."],
  "probabilities": [0.1, 0.9],
  "choice_index": 1,
  "choice": "...",
  "confidence": 0.9,
  "adaptation": "native",
  "protocol": "jev27-bare-v1",
  "model": "autotrust/JEV-27B-VL",
  "usage": {
    "prompt_tokens": 42,
    "completion_tokens": 1,
    "total_tokens": 43
  },
  "num_model_requests": 1
}
```

`confidence` is additive relative to the Python shim. `num_model_requests` is the number of
upstream passes (1, 3, or 4 in typical cases; more group passes for large tournaments).

## 7. `POST /v1/decisions`

OpenAI Decisions API shape.

### Request

```json
{
  "model": "ignored",
  "input": "text | [ { "role": "user", "content": "text | [parts]" } ]",
  "questions": [
    {"type": "predicate", "name": "p", "instructions": "..."},
    {"type": "choice", "name": "c", "instructions": "...", "choices": [
      {"value": "...", "description": "..."}
    ]},
    {"type": "score", "name": "s", "instructions": "...", "levels": [
      {"label": "...", "description": "..."}
    ]}
  ],
  "safety_identifier": "ignored"
}
```

Input handling (`_decisions_state`):

- A string is the state.
- A list of messages: string content is appended; for array content, `input_text` parts
  append `str(part.text)` and any `input_image` part returns 400
  `input_image content parts are not supported: this front is text-only`.
- Parts are joined with `"\n\n"`.
- Anything else is treated as empty text (the shim would raise on non-iterable input).

Question handling:

- `predicate`: options `["false","true"]`; answer
  `{"type":"predicate","name":...,"probability": p[true]}`.
- `choice`: 2 to `max_options` choices. Option text is `"{value}: {description}"` when the
  description is Python-truthy, else `str(value)`. Answer includes the original `value`,
  a probability per choice, and `confidence`.
- `score`: 2 to 6 levels (the head's score slot count). Option text follows the same rule on
  `label`/`description`. Answer `score = sum(i * p[i])`, probabilities with
  `{"value": i, "label": str(level.label), "probability": p}` and `confidence`.
- Any other type returns 400
  `unsupported question type <python repr>`, e.g. `'unknown'` or `None`.

All questions share one usage accumulator. Predicate and score run as single passes; choice
uses the default `tournament` strategy (which degrades to single for 16 or fewer options).

### Response

```json
{
  "model": "autotrust/JEV-27B-VL",
  "answers": [ ... ],
  "usage": {
    "input_tokens": 42,
    "output_tokens": 2,
    "total_tokens": 44
  }
}
```

## 8. Passthrough routes

`POST /v1/chat/completions` and `POST /v1/completions`:

- Body must be a JSON object; otherwise 400.
- `model` is overwritten with the configured vLLM id (`jev-awq`).
- All other fields are forwarded unchanged.
- Non-streaming responses preserve the upstream status, body bytes, and `content-type`.
- `stream: true` relays upstream bytes as they arrive with `text/event-stream`. The
  concurrency permit is held until the stream ends.
- Passthrough errors are not signed.

## 9. Signing

If `JEV_SIGNING_KEY` is non-empty:

- Paths starting with `/v1/decide` or `/v1/decisions` are signed, including error responses
  and `/v1/decide/info`.
- The header is `x-jev-signature: sha256=<lowercase hex>`.
- The MAC input is the exact response body byte sequence.
- `content-length` is removed and recomputed by the HTTP stack; `content-type` is preserved.
- All other paths are not signed.

## 10. Endpoints

### `GET /health`

```json
{"status":"ready","model":"autotrust/JEV-27B-VL","head":"merged-lm_head"}
```

### `GET /v1/models`

```json
{"object":"list","data":[
  {"id":"autotrust/JEV-27B-VL","object":"model","created":0,"owned_by":"autotrust"},
  {"id":"autotrust/JEV-27B","object":"model","created":0,"owned_by":"autotrust"},
  {"id":"jev-awq","object":"model","created":0,"owned_by":"autotrust"}
]}
```

### `GET /v1/decide/info`

```json
{
  "protocol": "jev27-bare-v1",
  "model": "autotrust/JEV-27B-VL",
  "max_options": 256,
  "native_choice_options": 16,
  "temperatures": {"noul": 1.0143, "choice": 1.0161, "score": 1.0036},
  "defaults": {"strategy": "tournament"}
}
```

### `GET /metrics`

Prometheus text format. Returns a comment-only body when metrics are disabled or the
recorder is not installed.

## 11. Edge cases

| Case | Behavior |
|---|---|
| Malformed JSON | 400, `invalid JSON body`. |
| Unknown `kind` | 400, `kind must be noul|score|choice`. |
| `choice` with 0, 1, or >max options | 400 with the exact count. |
| `noul`/`score` with supplied `options` | Supplied options ignored. |
| Null or scalar `state` | 400. |
| More than one image | 400. |
| `input_image` in `/v1/decisions` | 400, text-only front. |
| Unknown question type | 400, Python-repr of the value. |
| Upstream 4xx/5xx | 502, upstream status and first 600 characters in the message. |
| Upstream timeout | 504. |
| Upstream non-JSON / bad logprob shape | 502, `invalid backend response`. |
| Missing logprob for a requested token | Treated as `-1e9` before bias/temperature. |
| Ties in `choice_index` | Lowest index. |
| Fitted bias shorter than options | Missing entries `0.0`; no panic. |
| Empty probability vector | Internal 500 (should be unreachable). |
| Request body larger than `JEV_MAX_BODY_BYTES` | 413 from axum. |
| `JEV_TOKENIZER`/head mismatch | Process refuses to start. |
| `JEV_CALIBRATION_FITTED` path absent | Warning; fitted calibration disabled. |

## 12. Parity notes

- Prompt construction, label ids (verified against the real tokenizer: 256 labels,
  ids `32..57` then two-letter tokens), CPython JSON rendering, CRC32, and MT19937
  permutations match the reference.
- Softmax uses the same formula. `exp` may differ from CPython's libm by one unit in the
  last place on some platforms; probability vectors should agree to `~1e-15`.
- Upstream error status mapping, the additive `confidence` field, one-image validation, and
  the structured error envelope are intentional differences from the Python shim.
- `/v1/systemone` is intentionally absent.
- The server never starts or stops vLLM and never touches the production ports `8001` and
  `8500`.
