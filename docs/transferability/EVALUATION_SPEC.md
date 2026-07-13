# Multimodel Transferability Evaluation Specification

## Scope and separation

This branch adds replay-only evaluation of existing `metrics.json`, `clean.png`,
and `adversarial.png` artifacts. Replay code is not imported by the attack runner
and never generates or modifies an attack. Live API access is disabled unless the
caller supplies an explicit opt-in flag.

## Architecture

- `multimodal_victim.py` defines normalized responses and the provider protocol.
- `api_victims.py` owns OpenAI, Gemini, and Anthropic request construction.
- `transfer_eval.py` owns config parsing, discovery, stable caching, scoring,
  retries/rate limiting, append-safe records, manifests, and summaries.
- `scripts/replay_multimodel_eval.py` is the replay-only CLI. The existing GPT
  replay command remains compatible.

Optional generation fields are emitted only when configured and supported.

## Inputs and prompts

Each selected item directory must contain all three required artifacts. Metadata
is reconstructed from `metrics.json`. Caption tasks use the configured caption
prompt; VQA and receipt/OCR tasks use the exact item question (or an explicit
fallback). Clean and adversarial images receive identical prompts and settings.

## Records and cache

One JSONL record represents one condition within one candidate/run item. A
record key includes that evaluation-sample identity so equal item IDs from
different seeds or methods remain distinct paired observations. A separate
request-cache key is SHA-256 over canonical JSON containing schema version,
provider, model, prompt, image digest, and effective request configuration. It
excludes paths, timestamps, secrets, and responses. Resume skips terminal record
keys, while identical successful requests (most commonly repeated clean images)
reuse the raw response without another API call.

Records include experiment/source/item/task identity, provider and requested and
resolved model IDs, UTC timestamp, prompt and hash, image digest and condition,
effective config, normalized response, latency/usage/retries, refusal/error, and
git metadata. JSONL appends use a lock, flush, and fsync; summaries and manifests
use same-directory temporary files plus `os.replace`.

## Scoring

Deterministic scoring is provider-neutral. Normalized exact answers,
multiple-choice labels, booleans, and boundary-aware target/source keyword
matches are derived from attack metadata. A separate judge result may be stored
later but never replaces raw output.

`conditioned_success = adversarial_target_success and not clean_target_success`.

Summaries report evaluation records, actual API requests, reused responses,
attempted and valid pairs, clean and adversarial target
rates, conditioned ASR, source suppression, refusal and API failure rates, paired
transitions, model macro averages, and paired bootstrap 95% intervals. Failed
calls remain in attempted denominators and are never silently dropped.

## Safety and cost controls

Without `--allow-real-api`, the command performs a dry run. Dry run validates
artifacts and config and estimates both paired evaluation records and
deduplicated real request count/cost without loading API keys or clients. Real
runs enforce optional request and estimated-cost ceilings before
calls, bounded exponential retries, and configured rates. Keys are referenced
only by environment-variable name and are never serialized.

For an OpenAI-compatible gateway, first verify that a model advertised by
`/models` also accepts the standard Chat Completions vision payload:

```bash
TECHUTOPIA_API_KEY=... PYTHONPATH=src .venv/bin/python \
  scripts/smoke_openai_compatible_vision.py \
  --base-url https://copilot.techutopia.cn/v1 \
  --model gpt-4o --model gpt-5-mini \
  --image-format jpeg --image-detail low --browser-headers \
  --output outputs/api_smoke/techutopia.json --allow-real-api
```

Model listing and image capability are separate checks. A model can appear in
`/models` while the gateway rejects image media or does not route that model.
The smoke report records this distinction and never serializes the API key.

On 2026-07-13, this gateway advertised the floating and dated GPT-4o routes,
but only `gpt-4o-2024-05-13` accepted the standard image smoke. The floating
`gpt-4o` route rejected the image media type and `gpt-5-mini` returned
`model_not_supported`. `configs/transferability_techutopia_available.yaml`
therefore enables only the capability-verified dated route; model availability
must be rechecked immediately before a frozen API replay.

After held-out analysis is complete, freeze candidates before API replay:

```bash
PYTHONPATH=src .venv/bin/python scripts/freeze_api_candidates.py \
  --analysis-csv ../attack-vllm-surrogate/outputs/surrogate_analysis/all_transfer_results.csv \
  --stage stage2_equal_forwards --budget-mode equal_forwards \
  --top-k 2 --minimum-seeds 3 --minimum-targets 3 \
  --output outputs/api_frozen/ensemble_candidates.json
```

The manifest embeds the analysis CSV digest and a canonical freeze hash. Replay
verifies both before discovering image pairs:

```bash
TECHUTOPIA_API_KEY=... PYTHONPATH=src .venv/bin/python \
  scripts/replay_multimodel_eval.py \
  --config configs/transferability_techutopia_available.yaml \
  --frozen-candidates outputs/api_frozen/ensemble_candidates.json \
  --result-dir outputs/api_replay/ensemble_gpt4o \
  --allow-real-api --resume --max-requests 300
```

Use `transferability_techutopia_vqa.yaml` or
`transferability_techutopia_receipt.yaml` for those task prompts. The currently
unroutable `gpt-5-mini` remains explicitly disabled.

## Outputs and compatibility

The runner writes a run manifest, append-safe JSONL, one summary JSON per model,
combined JSON/CSV, and `summary_by_candidate_model.csv`. Frozen-manifest replays
retain candidate dataset/method/rank metadata, so each candidate is reported
separately for every API model even when item IDs overlap. Existing attack
losses, artifacts, metrics, configs, and
`scripts/replay_gpt_eval.py` semantics remain untouched.

## Test plan

Tests inject mocked clients and clocks and cover config parsing, missing keys,
request parity, cache stability, resume, refusals/errors/retries, atomic output,
conditioned scoring and denominators, omission of unsupported options, API
opt-in, and legacy GPT replay helpers. No test contacts a network service.
