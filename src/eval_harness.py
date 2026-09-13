"""Computes every metric this project reports: intent classification,
escalation decisions, and LLM-as-judge reply quality (with human-agreement
evidence). Orchestrates running the trivial baseline, the simple baseline,
and the full agent over the same held-out golden test split, and writes
everything to results/ so the report's numbers come from files, not from
someone typing a remembered number into REPORT.md.
"""
from __future__ import annotations

import json
import logging
import random
from collections.abc import Callable, Sequence
from pathlib import Path

from scipy.stats import spearmanr
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from . import config
from .baselines import SimpleBaseline, TrivialBaseline, load_golden_examples, split_dev_test
from .llm_client import NvidiaClient, NvidiaRequestError, run_concurrently
from .pipeline import SupportAgent
from .schemas import AgentAction, AgentResult, ConversationThread, GoldenExample, JudgeScore
from .utils import read_jsonl, set_global_seed

logger = logging.getLogger(__name__)

_JUDGE_DIMENSIONS = list(JudgeScore.model_fields.keys())  # excludes the computed "overall" field


def compute_intent_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> dict:
    """Accuracy, macro F1, per-intent precision/recall/F1, and a confusion matrix."""
    labels = sorted(set(y_true) | set(y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true) if y_true else 0.0
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": accuracy,
        "macro_f1": report["macro avg"]["f1-score"],
        "labels": labels,
        "per_intent": {
            label: {"precision": p, "recall": r, "f1": f, "support": int(s)}
            for label, p, r, f, s in zip(labels, precision, recall, f1, support)
        },
        "confusion_matrix": matrix.tolist(),
    }


def compute_escalation_metrics(y_true: Sequence[AgentAction], y_pred: Sequence[AgentAction]) -> dict:
    """Precision/recall/F1 treating ESCALATE as the positive class, plus the
    two error counts broken out separately -- they are not equally costly.
    A false auto-handle (should have escalated, didn't) reaches a customer
    with an unverified or overconfident reply; a false escalation just
    costs a human agent's time. See report/REPORT.md for how that
    asymmetry is discussed.
    """
    true_binary = [1 if a == AgentAction.ESCALATE else 0 for a in y_true]
    pred_binary = [1 if a == AgentAction.ESCALATE else 0 for a in y_pred]
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_binary, pred_binary, labels=[0, 1], zero_division=0
    )
    false_auto_handle = sum(t == 1 and p == 0 for t, p in zip(true_binary, pred_binary))
    false_escalation = sum(t == 0 and p == 1 for t, p in zip(true_binary, pred_binary))
    return {
        "precision_escalate": precision[1],
        "recall_escalate": recall[1],
        "f1_escalate": f1[1],
        "false_auto_handle_count": false_auto_handle,
        "false_escalation_count": false_escalation,
        "n": len(y_true),
    }


def judge_reply(customer_message: str, historical_evidence: str, reply: str, client: NvidiaClient) -> JudgeScore:
    """LLM-as-judge scoring on the 6-dimension rubric, 1-5 each.

    Explicitly instructed to be critical and use the full range: an
    earlier ungrounded prompt ("score this reply") defaulted to scoring
    every dimension as 3 regardless of reply quality -- a known central-
    tendency failure mode for LLM judges, not a bug in this harness (see
    DECISIONS.md).
    """
    prompt = (
        "Score this SpotifyCares support reply on 6 dimensions, each 1 (terrible) to 5 (excellent). "
        "Be critical and use the full range -- do not default to the middle.\n"
        "- groundedness: does it stick to what the historical evidence actually supports?\n"
        "- correctness: is it factually reasonable for a Spotify support context?\n"
        "- helpfulness: does it actually move the customer's problem forward?\n"
        "- brand_voice: friendly, concise, on-brand support tone?\n"
        "- actionability: does the customer know what to do next?\n"
        "- no_unsupported_claims: does it avoid inventing policies, refund amounts, timelines, or "
        "claiming an action was taken?\n\n"
        f"Historical evidence available to the agent:\n{historical_evidence}\n\n"
        f"Customer message: {customer_message!r}\n\n"
        f"Reply to score: {reply!r}"
    )
    return client.generate_json(prompt, JudgeScore, system_instruction="You are a strict, critical QA reviewer.")


def _format_evidence(result: AgentResult) -> str:
    if not result.retrieved_examples:
        return "(none retrieved)"
    return "\n".join(
        f"- similar case: {e.customer_problem!r} -> {e.brand_reply!r}" for e in result.retrieved_examples
    )


def compute_judge_agreement(human_scores: list[JudgeScore], llm_scores: list[JudgeScore]) -> dict:
    """Compare human and LLM-judge scores on the same examples.

    Reports, per dimension and overall: Spearman correlation, Cohen's
    kappa (computed on scores rounded to the nearest integer 1-5, since
    kappa is defined over discrete categories -- the "overall" field is a
    float average and would otherwise never exactly match between raters),
    exact agreement, and agreement within one point.
    """
    if len(human_scores) != len(llm_scores):
        raise ValueError("human_scores and llm_scores must be paired 1:1")

    def _agreement_for(human_values: list[float], llm_values: list[float]) -> dict:
        human_rounded = [int(round(v)) for v in human_values]
        llm_rounded = [int(round(v)) for v in llm_values]
        exact = sum(h == l for h, l in zip(human_rounded, llm_rounded)) / len(human_rounded)
        within_one = sum(abs(h - l) <= 1 for h, l in zip(human_rounded, llm_rounded)) / len(human_rounded)
        correlation, _p_value = spearmanr(human_values, llm_values) if len(human_values) > 1 else (float("nan"), None)
        kappa = cohen_kappa_score(human_rounded, llm_rounded) if len(set(human_rounded + llm_rounded)) > 1 else 1.0
        return {
            "spearman_correlation": correlation,
            "cohen_kappa": kappa,
            "exact_agreement": exact,
            "within_one_point_agreement": within_one,
        }

    result = {
        dimension: _agreement_for(
            [getattr(h, dimension) for h in human_scores], [getattr(l, dimension) for l in llm_scores]
        )
        for dimension in _JUDGE_DIMENSIONS
    }
    result["overall"] = _agreement_for([h.overall for h in human_scores], [l.overall for l in llm_scores])
    return result


def _run_full_agent(agent: SupportAgent, examples: list[GoldenExample]) -> list[AgentResult]:
    def call(example: GoldenExample) -> AgentResult:
        try:
            return agent.handle(example.customer_text)
        except NvidiaRequestError as exc:
            logger.error("Pipeline failed for thread %s: %s", example.thread_id, exc)
            return AgentResult(
                intent="General Complaint / Praise / Other",
                intent_confidence=0.0,
                reply="",
                action=AgentAction.ESCALATE,
                escalation_reason=f"pipeline error, escalating to be safe: {exc}",
                retrieved_examples=[],
            )

    return run_concurrently(call, examples)


def evaluate_system(
    name: str, run_fn: Callable[[str], AgentResult], examples: list[GoldenExample]
) -> dict:
    results = [run_fn(e.customer_text) for e in examples]
    intent_metrics = compute_intent_metrics([e.intent for e in examples], [r.intent for r in results])
    escalation_metrics = compute_escalation_metrics([e.action for e in examples], [r.action for r in results])
    return {"name": name, "intent": intent_metrics, "escalation": escalation_metrics, "results": results}


def find_failure_cases(examples: list[GoldenExample], results: list[AgentResult], limit: int = 20) -> list[dict]:
    """Examples where the full agent's intent or escalation call disagreed
    with the golden label -- raw material for the report's failure analysis."""
    failures = []
    for example, result in zip(examples, results):
        intent_wrong = example.intent != result.intent
        action_wrong = example.action != result.action
        if intent_wrong or action_wrong:
            failures.append(
                {
                    "thread_id": example.thread_id,
                    "customer_text": example.customer_text,
                    "true_intent": example.intent,
                    "predicted_intent": result.intent,
                    "true_action": example.action.value,
                    "predicted_action": result.action.value,
                    "escalation_reason": result.escalation_reason,
                    "reply": result.reply,
                }
            )
    return failures[:limit]


_HUMAN_JUDGE_SAMPLE_SIZE = 40
_JUDGE_RESULTS_PATH = config.RESULTS_DIR / "judge_results.jsonl"
_HUMAN_JUDGE_PATH = config.RESULTS_DIR / "human_judge_scores.jsonl"
_JUDGE_AGREEMENT_PATH = config.RESULTS_DIR / "judge_agreement.json"


def _read_jsonl_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sample_for_human_judge(judge_records: list[dict], sample_size: int, seed: int) -> list[dict]:
    """Sample from the LLM judge's own run (same thread_id, same reply) so
    the human and the LLM are rating the identical outputs -- not
    re-running the pipeline, which could draft a different reply."""
    rng = random.Random(seed)
    pool = judge_records[:]
    rng.shuffle(pool)
    return pool[: min(sample_size, len(pool))]


def _prompt_score(dimension: str) -> int:
    while True:
        raw = input(f"  {dimension} (1-5): ").strip()
        if raw.isdigit() and 1 <= int(raw) <= 5:
            return int(raw)
        print("  enter a number 1-5")


def human_judge_cli(sample_size: int = _HUMAN_JUDGE_SAMPLE_SIZE) -> None:
    """Interactive scoring of a sample of the full agent's actual replies
    (from a prior `python -m src.eval_harness` run) against the same 6-
    dimension rubric the LLM judge used. Required for the assignment's
    mandatory judge/human agreement evidence -- see compute_judge_agreement.
    Resumable: thread_ids already in human_judge_scores.jsonl are skipped.
    """
    if not _JUDGE_RESULTS_PATH.exists():
        raise FileNotFoundError(f"{_JUDGE_RESULTS_PATH} not found; run `python -m src.eval_harness` first")
    judge_records = _read_jsonl_records(_JUDGE_RESULTS_PATH)
    sample = sample_for_human_judge(judge_records, sample_size, config.RANDOM_SEED)

    already_scored = {r["thread_id"] for r in _read_jsonl_records(_HUMAN_JUDGE_PATH)} if _HUMAN_JUDGE_PATH.exists() else set()
    remaining = [r for r in sample if r["thread_id"] not in already_scored]
    print(f"{len(already_scored)} already scored, {len(remaining)} remaining ({len(sample)} total sample).")
    print("Score each dimension 1 (terrible) to 5 (excellent). Ctrl+C to stop and save.\n")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with _HUMAN_JUDGE_PATH.open("a", encoding="utf-8") as out:
        for i, record in enumerate(remaining):
            print(f"\n--- {i + 1}/{len(remaining)} ---")
            print(f"Customer: {record['customer_text']}")
            print(f"Reply: {record['reply']}")
            try:
                scores = {dim: _prompt_score(dim) for dim in _JUDGE_DIMENSIONS}
            except (EOFError, KeyboardInterrupt):
                print("\nStopping; progress saved.")
                return
            row = {"thread_id": record["thread_id"], **scores}
            out.write(json.dumps(row) + "\n")
            out.flush()
    print(f"\nDone -- see {_HUMAN_JUDGE_PATH}")


def compute_and_save_judge_agreement() -> dict:
    """Match human and LLM judge scores by thread_id and compute agreement.

    Only examples present in both files are compared -- a human session
    stopped partway through simply yields a smaller, still-valid
    agreement sample rather than an error.
    """
    llm_by_id = {r["thread_id"]: JudgeScore(**{k: r[k] for k in _JUDGE_DIMENSIONS}) for r in _read_jsonl_records(_JUDGE_RESULTS_PATH)}
    human_by_id = {r["thread_id"]: JudgeScore(**{k: r[k] for k in _JUDGE_DIMENSIONS}) for r in _read_jsonl_records(_HUMAN_JUDGE_PATH)}
    common_ids = [tid for tid in human_by_id if tid in llm_by_id]
    if not common_ids:
        raise ValueError("No overlapping thread_ids between human and LLM judge scores")

    agreement = compute_judge_agreement([human_by_id[tid] for tid in common_ids], [llm_by_id[tid] for tid in common_ids])
    agreement["n_compared"] = len(common_ids)
    _JUDGE_AGREEMENT_PATH.write_text(json.dumps(agreement, indent=2))
    logger.info("Judge/human agreement (n=%d) written to %s", len(common_ids), _JUDGE_AGREEMENT_PATH)
    return agreement


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_global_seed(config.RANDOM_SEED)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    golden = load_golden_examples()
    dev, test = split_dev_test(golden, config.GOLDEN_DEV_FRACTION, config.RANDOM_SEED)
    logger.info("Golden set: %d dev, %d test", len(dev), len(test))

    threads = list(read_jsonl(config.PROCESSED_DATA_PATH, ConversationThread))

    trivial = TrivialBaseline.train(dev)
    simple = SimpleBaseline.train(dev, threads, config.RANDOM_SEED)
    agent = SupportAgent.load()

    trivial_eval = evaluate_system("trivial_baseline", trivial.run, test)
    logger.info("Trivial baseline: intent_acc=%.3f", trivial_eval["intent"]["accuracy"])

    simple_eval = evaluate_system("simple_baseline", simple.run, test)
    logger.info("Simple baseline: intent_acc=%.3f", simple_eval["intent"]["accuracy"])

    full_results = _run_full_agent(agent, test)
    full_intent = compute_intent_metrics([e.intent for e in test], [r.intent for r in full_results])
    full_escalation = compute_escalation_metrics([e.action for e in test], [r.action for r in full_results])
    logger.info("Full agent: intent_acc=%.3f", full_intent["accuracy"])

    client = NvidiaClient()
    _min_score = JudgeScore(
        groundedness=1, correctness=1, helpfulness=1, brand_voice=1, actionability=1, no_unsupported_claims=1
    )

    def judge_one(pair: tuple[GoldenExample, AgentResult]) -> JudgeScore:
        example, result = pair
        try:
            return judge_reply(example.customer_text, _format_evidence(result), result.reply, client)
        except NvidiaRequestError as exc:
            logger.error("Judge scoring failed for thread %s: %s", example.thread_id, exc)
            return _min_score

    # Concurrent, like _run_full_agent -- a sequential loop over the whole
    # test set here was the actual reason an earlier run risked blowing the
    # 15-minute reproduction budget (this call happens once per test
    # example, on top of the full agent's own per-example calls).
    judge_scores = run_concurrently(judge_one, list(zip(test, full_results)))
    mean_overall = sum(s.overall for s in judge_scores) / len(judge_scores) if judge_scores else 0.0
    logger.info("Mean judge score (full agent replies): %.2f / 5", mean_overall)

    failures = find_failure_cases(test, full_results)

    metrics = {
        "trivial_baseline": {"intent": trivial_eval["intent"], "escalation": trivial_eval["escalation"]},
        "simple_baseline": {"intent": simple_eval["intent"], "escalation": simple_eval["escalation"]},
        "full_agent": {
            "intent": full_intent,
            "escalation": full_escalation,
            "mean_judge_score": mean_overall,
        },
        "dev_size": len(dev),
        "test_size": len(test),
    }
    (config.RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (config.RESULTS_DIR / "failure_cases.json").write_text(json.dumps(failures, indent=2))
    judge_records = [
        {"thread_id": e.thread_id, "customer_text": e.customer_text, "reply": r.reply, **s.model_dump()}
        for e, r, s in zip(test, full_results, judge_scores)
    ]
    with (config.RESULTS_DIR / "judge_results.jsonl").open("w") as f:
        for record in judge_records:
            f.write(json.dumps(record) + "\n")

    logger.info("Wrote metrics.json, failure_cases.json, judge_results.jsonl to %s", config.RESULTS_DIR)


def _self_check() -> None:
    """Metric-calculation self-check against a hand-built confusion matrix
    and a hand-built judge-agreement case -- catches a broken metric
    function before it silently corrupts every reported number."""
    y_true = ["a", "a", "b", "b"]
    y_pred = ["a", "b", "b", "b"]
    metrics = compute_intent_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 0.75, metrics["accuracy"]

    esc_true = [AgentAction.ESCALATE, AgentAction.ESCALATE, AgentAction.AUTO_HANDLE]
    esc_pred = [AgentAction.AUTO_HANDLE, AgentAction.ESCALATE, AgentAction.AUTO_HANDLE]
    esc_metrics = compute_escalation_metrics(esc_true, esc_pred)
    assert esc_metrics["false_auto_handle_count"] == 1
    assert esc_metrics["false_escalation_count"] == 0

    perfect_scores = [JudgeScore(groundedness=5, correctness=5, helpfulness=5, brand_voice=5, actionability=5, no_unsupported_claims=5)] * 3
    agreement = compute_judge_agreement(perfect_scores, perfect_scores)
    assert agreement["overall"]["exact_agreement"] == 1.0
    assert agreement["overall"]["cohen_kappa"] == 1.0

    print("eval_harness self-check passed")


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    command = sys.argv[1] if len(sys.argv) > 1 else None
    if command == "--self-check":
        _self_check()
    elif command == "--human-judge":
        human_judge_cli()
    elif command == "--judge-agreement":
        print(json.dumps(compute_and_save_judge_agreement(), indent=2))
    else:
        main()
