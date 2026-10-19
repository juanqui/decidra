# jev-engine API specification

This document is the contract implemented by `src/`. It ports the behavior of
`/srv/models/jev-awq/decide_shim.py` for the `JEV-27B-VL` AWQ W4A16 artifact, served by
the in-process native CUDA engine (`jev-native`) as `jev-awq`. Where this document and the
Python shim differ, the difference is called out explicitly.

## 1. Conventions

- All request and response bodies are UTF-8 JSON.
- Model ids: the client-facing name is `autotrust/JEV-27B-VL`; `/v1/models` also lists
  `autotrust/JEV-27B` and the configured id (`JEV_MODEL`, default `jev-awq`).
- The wire protocol string is always `jev27-bare-v1`.
- Probabilities are floats in `[0, 1]` and sum to 1 up to floating-point error.
- `state` is a string, a JSON object, or an array of content parts.
- A client request that fails validation returns **400**.
- A native backend failure returns **502**; a backend timeout returns **504**.
- Malformed JSON, wrong field types, and unsupported enum values never panic; they return
  structured errors.

### Error envelope

```json
{
  "detail": "human-readable message",
  "error": {
    "message": "human-readable message",
    "type": "invalid_request_error | backend_error | backend_timeout | not_implemented_error | internal_error",
    "code": 400
  }
}
```

`detail` is included for compatibility with clients written against FastAPI's
`HTTPException`.

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

The list may contain image parts, but the native backend is text-only: any image part
returns 400 `native vision not implemented: image states are not supported`.

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

The prompt is tokenized with `add_special_tokens=false` and passed to the native engine,
which returns raw logits for the requested head ids at the last position. Image parts never
reach the engine; they are rejected earlier with 400.

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

The backend returns raw logits for the token ids in `ids`. Missing ids use `-1e9`, and
values are clamped from below at `-1e9`:

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

## 5. Native backend read-out

`Backend::logits(prompt, ids)` receives the exact text prompt and the head token ids. The
native implementation:

1. Tokenizes `prompt` with the existing `tokenizers` tokenizer and
   `add_special_tokens=false`.
2. Calls `jev_native::Engine::decision_logits(tokens, ids)` on one blocking worker.
3. Returns raw `f32` logits at the last position, positionally aligned with `ids`.

The engine keeps all packed decoder weights resident after `preload_all()`; a request does
no weight I/O. Forwards are serialized by a mutex because the CUDA context/stream is not
safe for concurrent use.

Raw logits differ from log-probabilities by `log Z`, a constant for a fixed prompt. The
head applies bias, temperature, and optional fitted calibration before softmax, so the
normalizer cancels and the resulting probabilities are identical to the old logprob path.

A malformed backend result (length mismatch, non-finite values, engine error) returns 502
with `backend request failed` or `invalid backend response`.

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
- Any image part is a 400 (`native vision not implemented`).

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
backend passes (1, 3, or 4 in typical cases; more group passes for large tournaments).

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
  `native vision not implemented: input_image content parts are not supported`.
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

## 8. `POST /v1/systemone`

xPNP hosted System One, ported from the shim's `systemone` handler. It is a batch of
independent single-pass decisions (no tournament/permute), with its own error envelope.

### Request

```json
{
  "state": "text | object | array",
  "questions": {
    "name": {
      "type": "noul | choice",
      "instructions": "string",
      "criteria": {"true": "...", "false": "..."}
    }
  }
}
```

- `questions` must be a non-empty object with at most **8** entries.
- `state` follows `_s1_split_state`, which differs from `_parts`:
  - string → text; object → CPython `json.dumps` text; array → text parts concatenated
    with `"\n"`.
  - `{"type":"text","text":...}` appends text.
  - Image parts are parsed for validation but any image returns 400
    `native vision not implemented: image states are not supported`.
- Each question must be an object with non-empty `instructions` after Python `.strip()`.
- `noul`: if `criteria` is an object and `true` or `false` is non-null, append
  `"\nAnswer true when: <true>\nAnswer false when: <false>"` to the state text. A missing
  or null criterion becomes `""`.
- `choice`: `criteria` must be an object with 2..16 entries. Keys must match
  `[a-z0-9-]{1,64}`. Options are `"key: value"` when the value is non-null, else `"key"`.
- `score` is rejected with 400 `score questions are not served by this backend`; any other
  type is rejected with 400 using the Python repr of the value.
- All questions are validated before any backend forward.

### Execution

For each question, `s1_pass` runs **once** with the same decision-head bias, per-kind
temperature, fitted calibration (if enabled), and softmax as `/v1/decide`. There is no
tournament or permute. The prompt state is `[text + "\n"]`. Questions execute
sequentially.

### Response

```json
{
  "id": "xpnp-hosted-<32 lowercase hex>",
  "model": "autotrust/JEV-27B-VL@ba3f0d58+awq-w4a16",
  "provider": "xPNP Hosted",
  "answers": {
    "safe": {"type": "noul", "noul": 0.9997},
    "team": {
      "type": "choice",
      "choice": "sales",
      "confidence": 0.9998,
      "probabilities": {"billing": 0.0001, "support": 0.0001, "sales": 0.9998}
    }
  },
  "usage": {
    "input_tokens": 42,
    "output_tokens": 0,
    "cost": 0,
    "num_model_requests": 2
  }
}
```

- `model` comes from `JEV_SERVED_IDENTITY` (default
  `autotrust/JEV-27B-VL@ba3f0d58+awq-w4a16`).
- `answers` preserves question insertion order.
- `choice` uses the lowest-index tie-break and reports per-key probabilities.
- `usage.input_tokens` sums prompt tokens across questions; `output_tokens` is always 0;
  `cost` is 0; `num_model_requests` is the number of passes.
- Errors use the xPNP envelope:
  `{"error":{"message":"...","code":400}}`. Backend failures become
  `{"error":{"message":"backend: ...","code":502|504}}` with the server-wide backend
  status policy. `/v1/systemone` is not signed (same as the shim).

## 9. Generation routes

`POST /v1/chat/completions` and `POST /v1/completions` always return **501** with:

```json
{
  "detail": "native generation not implemented: this server exposes decision endpoints only",
  "error": {
    "message": "native generation not implemented: this server exposes decision endpoints only",
    "type": "not_implemented_error",
    "code": 501
  }
}
```

The body is not parsed and nothing is proxied. These routes are not signed.

## 10. Signing

If `JEV_SIGNING_KEY` is non-empty:

- Paths starting with `/v1/decide` or `/v1/decisions` are signed, including error responses
  and `/v1/decide/info`.
- The header is `x-jev-signature: sha256=<lowercase hex>`.
- The MAC input is the exact response body byte sequence.
- `content-length` is removed and recomputed by the HTTP stack; `content-type` is preserved.
- All other paths are not signed.

## 11. Endpoints

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

## 12. Edge cases

| Case | Behavior |
|---|---|
| Malformed JSON | 400, `invalid JSON body`. |
| Unknown `kind` | 400, `kind must be noul|score|choice`. |
| `choice` with 0, 1, or >max options | 400 with the exact count. |
| `noul`/`score` with supplied `options` | Supplied options ignored. |
| Null or scalar `state` | 400. |
| Any image part | 400, `native vision not implemented`. |
| `input_image` in `/v1/decisions` | 400, `native vision not implemented`. |
| `/v1/systemone` questions missing/not object/empty | 400, `questions must be a non-empty object`. |
| `/v1/systemone` more than 8 questions | 400, `at most 8 questions per request`. |
| `/v1/systemone` `score` question | 400, `score questions are not served by this backend`. |
| `/v1/systemone` bad criterion key | 400, `choice keys must match [a-z0-9-]+, got '<key>'`. |
| `/v1/systemone` image state | 400, `native vision not implemented`. |
| Unknown question type | 400, Python-repr of the value. |
| `/v1/chat/completions`, `/v1/completions` | 501, `native generation not implemented`. |
| Native engine failure | 502, `backend request failed`. |
| Backend timeout | 504, `backend request timed out`. |
| Malformed/non-finite logits | 502, `invalid backend response`. |
| Missing logit for a requested token | Treated as `-1e9` before bias/temperature. |
| Ties in `choice_index` | Lowest index. |
| Fitted bias shorter than options | Missing entries `0.0`; no panic. |
| Empty probability vector | Internal 500 (should be unreachable). |
| Request body larger than `JEV_MAX_BODY_BYTES` | 413 from axum. |
| `JEV_TOKENIZER`/head mismatch | Process refuses to start. |
| `JEV_CALIBRATION_FITTED` path absent | Warning; fitted calibration disabled. |

## 13. Parity notes

- Prompt construction, label ids (verified against the real tokenizer: 256 labels,
  ids `32..57` then two-letter tokens), CPython JSON rendering, CRC32, and MT19937
  permutations match the reference.
- Softmax uses the same formula. `exp` may differ from CPython's libm by one unit in the
  last place on some platforms; probability vectors should agree to `~1e-15`.
- Backend error status mapping, the additive `confidence` field, text-only validation, and
  the structured error envelope are intentional differences from the Python shim.
- `/v1/systemone` ports the shim handler, including its separate error envelope; it is not signed.
- The server never starts or stops another model process and never touches the production
  ports `8001` and `8500`.
