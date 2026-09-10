# Decision log

Non-obvious calls made while building this, and why. Chronological within
each area.

1. **Brand: SpotifyCares, not a bigger brand like AmazonHelp/AppleSupport.**
   AmazonHelp (170k brand tweets) and AppleSupport (107k) are dominated by
   generic "please DM us" privacy redirects with little substantive public
   troubleshooting content to ground replies in. SpotifyCares (43k) is
   smaller but its public replies frequently contain real troubleshooting
   steps ("try restarting", "which device/OS/version"), which is exactly
   the grounding material a retrieval-based reply drafter needs.

2. **LLM provider: NVIDIA NIM, not Gemini.** Originally planned around
   Gemini. Live testing against the configured Gemini API key found its
   free-tier `generateContent` quota is **5 requests/minute** -- unworkable
   for an eval harness needing a few hundred chat calls per run (would put
   a 150-200 example run at 1.5-3+ hours, not the assignment's 15-minute
   reproduction budget). NVIDIA's build.nvidia.com NIM catalog exposes
   hosted open models through a standard OpenAI-compatible API with a far
   more usable free tier (measured: sequential calls average ~1-2s with no
   throttling; concurrency above ~5-8 starts producing 429s/timeouts).
   Switched `src/llm_client.py` to the `openai` SDK pointed at NIM's base
   URL rather than `google-genai`.

3. **Pinned model names, not a "-latest" alias.** `nvidia/nemotron-3.5-lightning-30b-a3b`
   (chat) and `nvidia/nemotron-3-embed-1b` (embeddings), both verified live
   against the configured key on 2026-09-10. Most of NIM's ~80-model public
   catalog returned 404 "not found for account" when actually invoked with
   this free-tier key, despite appearing in `client.models.list()` --
   availability had to be verified empirically, not assumed from the catalog.

4. **`chat_template_kwargs.thinking` disabled for every pipeline call, not
   just some.** The chat model is a reasoning model that defaults to
   emitting a "thinking process" preamble before its actual answer, which
   both wastes tokens/latency and (with a small `max_tokens`) can truncate
   before ever producing the real answer. Disabling it via NIM's
   `extra_body={"chat_template_kwargs": {"thinking": false}}` (not
   Gemini-style `thinking_budget=0`, which this model family rejects
   outright) gives clean, fast, direct answers. Reply drafting was
   initially left with thinking enabled for "better" replies, but measured
   latency/timeout risk on the free endpoint (one call took 170s across
   retries) wasn't worth it against the 15-minute reproduction budget, so
   it was switched to match classify/escalate.

5. **JSON structured output is prompted with an example instance, not the
   formal JSON Schema.** Passing `pydantic_model.model_json_schema()`
   directly (with its `properties`/`required`/`title` wrapper) into the
   prompt made the model echo the schema's own document structure back
   instead of producing a conforming value. `_example_instance()` in
   `llm_client.py` instead builds a placeholder-filled example dict (same
   keys, illustrative values) from the pydantic model's fields, which the
   model reliably instantiates correctly.

6. **LLM-judge rubric explicitly says "be critical, use the full range."**
   An initial version of the reply-quality judge prompt scored every
   dimension as a flat 3/5 regardless of actual reply quality -- a known
   central-tendency bias in LLM judges, confirmed by testing a deliberately
   nonsense reply (unrelated to the customer's problem) and getting the
   same all-3s result. Adding the explicit instruction fixed it (a
   plausible-but-mediocre reply scored ~2.8/5 average, an irrelevant one
   ~1.7/5 in side-by-side testing).

7. **Oversized reconstructed "threads" (>12 turns) are dropped, not
   truncated.** Grouping SpotifyCares replies by shared root ancestor
   occasionally merges dozens of *unrelated* customers who all replied to
   the same viral broadcast tweet into one "thread" (one observed case had
   308 turns from many different people). These aren't real 1:1
   conversations, so they're filtered out entirely rather than kept and
   truncated, which would silently present a fragment of cross-talk as a
   coherent conversation. Measured: 195 of ~28,300 reconstructed threads
   exceed the cap (well under 1%).

8. **Thread reconstruction unions ancestor chains rather than re-walking
   downward from the root.** TWCS only gives a reliable single-parent
   pointer (`in_response_to_tweet_id`); `response_tweet_id` is redundant
   and can branch (multiple replies to one tweet). Walking *up* from every
   SpotifyCares reply that shares a root, then unioning the resulting
   chains, avoids having to guess which branch is the "real" conversation
   when a thread forks -- every tweet included is one we know belongs
   because we arrived at it from an actual SpotifyCares reply.

9. **Unrecognized/out-of-taxonomy intents fail safe to `escalate`, not
   `auto_handle`.** Both the LLM classifier's out-of-taxonomy fallback and
   the simple baseline's rule-based escalation default an unrecognized
   label to escalation. Silently auto-handling something the system
   doesn't confidently recognize is the more expensive mistake (see
   REPORT.md's discussion of false-auto-handle vs. false-escalation cost).

10. **Retrieval corpus intent tags come from the cheap TF-IDF+LogReg
    baseline classifier, not an LLM call per historical thread.** The
    retrieval index covers ~4,000 threads; classifying all of them with the
    LLM just to support intent-filtered retrieval would cost thousands of
    extra API calls for no benefit the cheap local classifier doesn't
    already provide for this purpose. The live pipeline still uses the LLM
    classifier for the actual incoming message being handled.

11. **Golden set labeling is hybrid (human seed + LLM-assisted, human-
    confirmed), not fully manual or fully automated.** Fully manual
    labeling of 150-250 examples is the most assignment-faithful option but
    costs hours; fully LLM-labeled-and-spot-checked is fast but weak to
    defend as "hand-labelled... by you" when asked to justify individual
    labels live. The hybrid keeps every final label human-confirmed (with
    provenance recorded in `LabelSource`) while bounding the time cost.

12. **The golden dev split does double duty: prompt/threshold tuning *and*
    baseline classifier training data.** With only ~45-50 examples in the
    dev slice, the TF-IDF+LogReg simple baseline is trained on a genuinely
    small dataset -- this is disclosed as a limitation (see REPORT.md)
    rather than presented as a fully independent, well-resourced baseline.
    The alternative (a separate labeled training set) wasn't in scope for a
    150-250-example golden set built by hand.

13. **Retrieval is a flat, L2-normalized NumPy matrix, not a vector
    database.** The corpus is ~4,000 vectors (2048-dim) -- a few tens of MB,
    searchable with one matrix-vector product in milliseconds. A vector DB
    (FAISS, Chroma, etc.) would add a dependency and operational surface
    for no measurable benefit at this scale.

14. **The simple baseline's retrieval uses TF-IDF cosine similarity, not
    the same embeddings as the full system.** The point of a "simple, no-LLM"
    baseline is to be a genuinely different, cheaper method, not the full
    system's retrieval with a worse classifier bolted on -- otherwise the
    comparison mostly measures the classifier, not the retrieval approach.

15. **Reply drafting forces structured JSON output (`{"reply": ...}`), not
    free text.** A free-text version of the reply-drafting prompt --
    otherwise identical -- reliably echoed the customer's own message back
    verbatim instead of answering it, reproducing across repeated runs.
    Root cause: the grounding block showed 4 repetitions of a "Customer: X
    / SpotifyCares replied: Y" two-line format immediately before "now
    write the next one," which the model pattern-completed by continuing
    with the new customer's own text rather than treating the instructions
    after it as the actual task. Two fixes together resolved it: rewriting
    the grounding block as numbered prose instead of a repeated dialogue
    template, and switching the call from free-text to the same
    JSON-schema-constrained output used for classification and escalation
    -- forcing a `{"reply": ...}` shape breaks the "just continue the
    pattern" completion mode. This reduced but did not eliminate the
    failure (a short, simple message -- "can you add the new Drake album
    please" -- still got echoed back verbatim on a later test), so
    `pipeline._looks_like_echo()` backs the prompt fix with a runtime
    check: normalized-text similarity between the draft and the customer's
    own message above 0.8 triggers one corrective retry with an added
    "that was not a reply" instruction before falling back to a generic
    acknowledgement. Belt-and-suspenders rather than trusting either layer
    alone -- a customer-facing reply that just parrots their own complaint
    back at them is a real correctness bug, not a cosmetic one.

16. **No frontend, no live account actions.** Out of scope for what the
    assignment is testing (classify/draft/decide + prove it's trustworthy).
    The agent never claims to have taken an action (a refund, a
    cancellation) it cannot actually perform -- see the reply-drafting
    prompt's explicit constraint against this in `src/pipeline.py`.
