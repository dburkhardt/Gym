# Frozen sample staging

`prepare_sample` preserves selected rows, exposes stable task IDs, and publishes a
sibling `.source.json` receipt with source/output SHA-256 hashes and ordered IDs.
Campaign preparation wrappers expose Gym's existing `is_prepared_data_current`
hook. Native cached preparation validates the receipt, selected source, and
normalization mode before reuse. Missing, interrupted, or stale canonical caches
are prepared again from the selected source; a source-pin mismatch refuses reuse.
Preparation and receipt publication are separately atomic, so an interruption
between them is detected on the next cache check.

Set `CAMPAIGN_SAMPLE_JSONL` and optionally `CAMPAIGN_SAMPLE_SHA256` to select the
frozen source. GeneHop uses its existing `GENEHOP_SAMPLE`/`GENEHOP_PREPARED_SAMPLE`
settings; Fukuyama's frozen mode uses `FUKUYAMA_FROZEN_SAMPLE`. Their hooks bind the
same source selection as their preparation functions. Fukuyama's creator-derived,
non-frozen mode retains its existing cache behavior.

An explicitly overridden prepared-input path without a receipt remains supported.
If a frozen source/hash is selected, that external file must match the source
bytes exactly. Without a selected source/hash, the override remains the operator's
explicit input and makes no campaign provenance claim. Receipt-backed overrides
are always checked; invalid overrides are refused without rewriting them or
preparing a different canonical path. Unrelated preparation modules without this
hook retain their existing behavior. These checks do not authorize inference or
establish scientific correctness of the selected tasks.
