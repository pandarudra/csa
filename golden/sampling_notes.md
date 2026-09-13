# How the golden set was sampled and labeled

150 examples, labeled across two sessions on 2026-09-10 and 2026-09-13.
Methodology below was decided and implemented before labeling started; the
numbers are the actual result.

## Sampling

`sample_candidates()` in `src/label_tool.py` takes a uniform random sample
(seeded, `RANDOM_SEED=42`) of `GOLDEN_SET_TARGET_SIZE` (200) threads from
the 4,000-thread processed corpus. No stratification by intent was
attempted at sampling time, because the intents don't exist for these
threads yet -- that's what labeling produces. Stratifying by a
keyword-heuristic proxy for intent was considered and rejected as circular
and unnecessary complexity: the intent clusters found during taxonomy
discovery (see `intents.yaml`'s header comment) are large enough relative
to 200 that plain random sampling was expected to give a workable spread
across all 6 intents without deliberate balancing.

Actual resulting distribution (150 total):

```
Billing & Subscription:             30
General Complaint / Praise / Other: 28
Playback / App Technical Issue:     27
Content & Catalog:                  24
Feature Request & Product Feedback: 23
Account & Login:                    18
```

Every intent cleared 18 examples -- no bucket is so thin that per-intent
metrics are pure noise, though Account & Login (the smallest, 18 total, 14
of those landing in the test split) still means each individual mistake on
it swings that intent's precision/recall by ~7 points. Worth remembering
when reading `results/metrics.json`'s per-intent numbers.

## Labeling

Hybrid seed + assisted, per `golden/labeling_rubric.md`:

1. The first `GOLDEN_SEED_LABEL_COUNT` (45) sampled threads were labeled by
   a human with no model assistance -- these seed labels anchor the
   taxonomy in real judgment and become few-shot context for step 2.
2. For the remaining threads, `src/label_tool.py` asks the LLM to draft a
   suggested (intent, action, reason) using the seed labels + rubric as
   context. Every suggestion is displayed and the human explicitly accepts
   (Enter) or edits (typing a different choice) every field before it is
   written to `golden_set.csv` -- see `LabelSource` in `src/schemas.py` for
   exactly which of `human`, `llm_suggested_human_confirmed`, or
   `llm_suggested_human_edited` was recorded for each row.

Actual label-source breakdown (150 total):

```
human:                          48   (the seed set -- slightly above the
                                      planned 45, since a few extra were
                                      labeled before the assisted phase
                                      config change landed, see below)
llm_suggested_human_confirmed:  76
llm_suggested_human_edited:     26
```

Edit rate on assisted labels: 26 / 102 = 25.5% -- about one in four
LLM suggestions needed a real correction (a different intent, a flipped
action, or a rewritten reason), not just a rubber-stamp accept. That rate
is evidence the human review step was doing real work, not formality.

**Process note:** `GOLDEN_SET_TARGET_SIZE` was originally set to 200,
lowered to 150 (the assignment's stated minimum) partway through labeling
once 77 examples were already done, after weighing labeling time against
the marginal value of the extra 50. `sample_candidates()`'s seeded shuffle
doesn't depend on `target_size` (it truncates the same shuffled order to a
different length), so every already-labeled thread_id remained valid under
the smaller target -- verified directly before continuing (see
`report/DECISIONS.md`).

## Dev/test split

`src/baselines.split_dev_test()` splits the final golden set into a dev
slice (`GOLDEN_DEV_FRACTION=0.25`, stratified by intent) used for
prompt/threshold tuning and for training both baseline classifiers, and a
held-out test slice used for every reported metric. See
`report/DECISIONS.md` for why the dev slice does double duty as baseline
training data, and why that's disclosed as a limitation rather than hidden.
