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

One JSONL record represents one image request. Its cache key is SHA-256 over
canonical JSON containing schema version, provider, model, prompt, image digest,
and effective request configuration. It excludes paths, timestamps, secrets, and
responses. Resume skips terminal records with matching keys.

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

Summaries report attempted and valid requests/pairs, clean and adversarial target
rates, conditioned ASR, source suppression, refusal and API failure rates, paired
transitions, model macro averages, and paired bootstrap 95% intervals. Failed
calls remain in attempted denominators and are never silently dropped.

## Safety and cost controls

Without `--allow-real-api`, the command performs a dry run. Dry run validates
artifacts and config and estimates request count/cost without loading API keys or
clients. Real runs enforce optional request and estimated-cost ceilings before
calls, bounded exponential retries, and configured rates. Keys are referenced
only by environment-variable name and are never serialized.

## Outputs and compatibility

The runner writes a run manifest, append-safe JSONL, one summary JSON per model,
and combined JSON/CSV. Existing attack losses, artifacts, metrics, configs, and
`scripts/replay_gpt_eval.py` semantics remain untouched.

## Test plan

Tests inject mocked clients and clocks and cover config parsing, missing keys,
request parity, cache stability, resume, refusals/errors/retries, atomic output,
conditioned scoring and denominators, omission of unsupported options, API
opt-in, and legacy GPT replay helpers. No test contacts a network service.
