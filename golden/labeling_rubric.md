# Golden set labeling rubric

Used for both the human seed labels and as context for the LLM-assisted
suggestions on the rest of the set (see `src/label_tool.py`). Every label in
`golden_set.csv`, seed or assisted, is a value a human explicitly confirmed
or edited -- see `sampling_notes.md` for how that worked in practice.

## Intent

Pick exactly one intent from `intents.yaml`, based on what the customer is
actually asking for in their opening message (not the eventual resolution).
If a message plausibly fits two intents, prefer the one that determines
what a real reply would need to say or do. If nothing fits, use
"General Complaint / Praise / Other" rather than forcing a bad match.

## Escalate vs. auto-handle

Start from the intent's `always_escalate` default in `intents.yaml`, then
override it if the specific message warrants it:

- **Escalate even if the intent defaults to auto-handle** when the message
  shows: real anger/urgency (caps, profanity, "unacceptable", repeated
  attempts already visible in the thread), a request that implies a
  specific promise a public reply can't safely make (a refund amount, a
  timeline, compensation), or content ambiguous enough that a wrong public
  guess would make things worse.
- **Auto-handle even if the intent defaults to escalate** essentially never
  applies for Account & Login or Billing & Subscription in this taxonomy --
  both always require verifying account ownership first, which a reply
  can't do. If you find a genuine counterexample while labeling, flag it in
  `sampling_notes.md` rather than silently deviating from the rubric.
- Otherwise, follow the intent default.

## Reason

One sentence, stating the concrete fact that drove the decision (e.g. "asks
for a refund, needs identity verification" or "generic playback bug with a
standard fix, no account action needed") -- not a restatement of the
action ("this should be escalated").
