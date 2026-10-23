# CSMeD complete-review ranking preparation

This maintained component prepares creator-compatible Cochrane EVAL review exports and verifies complete saved rankings. It does not define a Nemotron solver, publish a runnable Gym benchmark configuration, or replace CSMeD with CSMeD-FT classification. Original retrieval assets/loader alignment remain unresolved. The inspected creator baseline uses retriv sparse/dense retrieval, not an LLM classifier.

Pinned baseline revision: `81cd35498bdea8981e5310842c32398625a10e95`. Preparation executes the exact source-bound `extract_review_details` and `prepare_data` functions from `experiments/csmed_cochrane/csmed_cochrane_retrieval.py`. It requires a checksum-bound JSON export with top-level `EVAL` mapping review IDs to the creator's `dataset_details` and `data.train` structure. This is an explicit interchange format for an authorized export, not a claim that the unavailable old loader has been repaired. No frozen review subset is invented. Every supplied review and every document are retained; missing text, duplicate IDs and invalid labels refuse publication. Empty search-strategy fallback remains the creator's literal `no search query` behavior.

Template from Gym root (replace all paths/checksum):

```sh
python -m benchmarks.csmed.prepare \
  --source /path/to/pinned/experiments/csmed_cochrane/csmed_cochrane_retrieval.py \
  --eval-export /path/to/authorized-eval-export.json \
  --expected-sha256 TRUSTED_EXPORT_SHA256 \
  --destination /path/to/fresh-preparation
```

Outputs separate `review-inputs.json` (title/abstract/query/criteria and full document collection) from `qrels.json` (reference labels). Never expose qrels or host preparation receipts to a solver. The input export checksum proves supplied-byte identity, not creator provenance; the receipt says so. Existing output directories are refused. Failure during publication may leave owned partial output for inspection; validation finishes before publication begins.

Saved rankings have the shape `{run_name: {review_id: [[document_id, score], ...]}}`. Preserve descending order and ties as submitted. Require exact review coverage, exactly one entry per corpus document, finite numeric scores, binary integer qrels and both relevance classes. Degenerate classes are refused because the creator supplementary normalized measures divide by class counts; they are not silently removed or scored zero. Duplicate JSON keys are rejected before they can collapse identities.

## Verifier

Use an isolated Python3.11 runtime with `resources_servers/csmed_ranking/requirements.txt` (ranx0.3.18, the retained qualification version). Cache directories must be writable; ranx imports ir_datasets and matplotlib even though this operation does not acquire a dataset.

```sh
IR_DATASETS_HOME=/path/to/cache/ir MPLCONFIGDIR=/path/to/cache/mpl \
python -m resources_servers.csmed_ranking.grading \
  --source /path/to/pinned/experiments/csmed_cochrane/measures.py \
  --rankings /path/to/saved-rankings.json \
  --qrels /path/to/prepared/qrels.json \
  --destination /path/to/fresh-score-output
```

The verifier calls the unchanged creator `evaluate_runs`, retaining its21 ranx metrics, Bonferroni comparison settings, LaTeX report and all supplementary measures. It captures the actual returned ranx comparison object's values without recomputing a substitute metric. `metrics.json` includes named metrics, review/document denominators, input hashes and source identity; there is no pooled scalar reward. Creator supplementary metrics retain submitted insertion/tie order; ranx's own ranking/tie behavior is unchanged. Reversed/perfect fixtures compare maintained output directly against creator LaTeX and supplementary JSON. These are synthetic metric controls, not acquired EVAL data or scientific model results.

Tests: `tests/unit_tests/test_csmed_ranking.py`. Set `CSMED_TEST_PYTHON` to an existing qualified ranx0.3.18 interpreter to run the full creator parity subprocess; otherwise that dependency-specific test skips explicitly. The other controls validate missing/duplicate/unknown documents, missing reviews, nonfinite/ascending scores, boolean scores, degenerate labels, source-bound preparation and separate qrels. No installs or provider calls occur in tests.

A future native resource endpoint must bind the exact complete review population and request lineage, preserve distinct query/run strata and report metric eligibility without inventing an accuracy reward. No discovery manifest is added until that route and an explicitly selected solver protocol are qualified.

## Explicit raw-occurrence preparation

`--preserve-occurrences` retains repeated document IDs in the original ordered
`review-inputs.json` collection. It emits `corpus-occurrences.json`, mapping each
review-local zero-based occurrence ID to its separate PMID/document scoring ID
and text SHA256. Qrels remain keyed by distinct document IDs. Conflicting labels
for repeated IDs refuse publication; no last-label repair is performed. Missing
review metadata and all existing text/label checks still fail. Default preparation
continues to require unique document IDs.

This option prepares a creator corpus for a future qualified retrieval integration;
it does not qualify a solver or change the ranking verifier. The verifier still
requires one score per distinct document ID. Do not replace the collection with
the occurrence map or deduplicate before indexing: retriv0.2.3 indexes each row,
then projects ranked occurrences through `dict(zip(document_ids, scores))`.
This preserves the last encountered score and initial insertion order per ID.
The creator dependency is `retriv~=0.2.3`, so this inspected version is not proof
of the original resolved environment. Full retrieval and missing EVAL metadata
remain separate qualification gates.
