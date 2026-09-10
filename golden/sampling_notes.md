# How the golden set was sampled and labeled

**Status: placeholder.** This file describes the intended methodology,
already implemented in `src/label_tool.py`. The actual numbers below get
filled in once the real labeling session (`python -m src.label_tool`) is
complete -- see the TODO markers.

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

TODO after labeling: report the actual resulting intent distribution here,
e.g.:

```
Account & Login:                    NN
Billing & Subscription:             NN
Playback / App Technical Issue:     NN
Content & Catalog:                  NN
Feature Request & Product Feedback: NN
General Complaint / Praise / Other: NN
```

If any intent ended up with fewer than ~10 examples, note it here -- that
bucket's per-intent metrics in `results/metrics.json` should be read with
appropriate skepticism (small-sample noise), and this is exactly the kind
of thing that belongs in report/REPORT.md's "misleading headline number"
section.

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

TODO after labeling: report the label-source breakdown, e.g.:

```
human:                          45   (the seed set)
llm_suggested_human_confirmed:  NN
llm_suggested_human_edited:     NN
```

A high edit rate on assisted labels would say the LLM's suggestions
needed real correction, not just rubber-stamping -- also worth surfacing
honestly in the report rather than glossing over.

## Dev/test split

`src/baselines.split_dev_test()` splits the final golden set into a dev
slice (`GOLDEN_DEV_FRACTION=0.25`, stratified by intent) used for
prompt/threshold tuning and for training both baseline classifiers, and a
held-out test slice used for every reported metric. See
`report/DECISIONS.md` for why the dev slice does double duty as baseline
training data, and why that's disclosed as a limitation rather than hidden.
