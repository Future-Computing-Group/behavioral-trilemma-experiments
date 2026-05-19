"""Pure functions for the multi-model competence probe (Path B A.2).

No I/O, no Ollama, no argparse: the driver lives in a later task.
Tier classification is by category-name suffix; separation thresholds
are passed as arguments (never read from config here).
"""

_EASY_SUFFIXES = ("_easy", "_common", "_simple")
_HARD_SUFFIXES = ("_hard", "_obscure", "_algorithmic")


def classify_tier(category: str) -> str:
    """Return "easy" or "hard" by the fixed suffix rule.

    Unknown suffix fails loud (no default bucket).
    """
    if category.endswith(_EASY_SUFFIXES):
        return "easy"
    if category.endswith(_HARD_SUFFIXES):
        return "hard"
    raise ValueError(f"unknown category suffix: {category!r}")


def _tier_mean(per_task_acc, task_category, tier):
    """Mean accuracy over tasks whose category classifies as `tier`.

    Empty tier set fails loud.
    """
    vals = [
        per_task_acc[tid]
        for tid, cat in task_category.items()
        if classify_tier(cat) == tier
    ]
    if not vals:
        raise ValueError(f"empty task set for tier: {tier!r}")
    return sum(vals) / len(vals)


def compute_separation(
    per_task_acc: dict[str, float],
    task_category: dict[str, str],
    easy_min_acc: float = 0.70,
    binding_max_acc: float = 0.45,
    min_spread: float = 0.20,
) -> dict:
    """Evaluate the three-clause separation criterion.

    Clauses are independent and evaluated in order 1 -> 2 -> 3; `reason`
    is the first failing clause's tag, or "separates" if all pass.
    """
    easy_acc = _tier_mean(per_task_acc, task_category, "easy")
    binding_acc = _tier_mean(per_task_acc, task_category, "hard")

    clause1 = easy_acc >= easy_min_acc
    clause2 = binding_acc <= binding_max_acc
    clause3 = (easy_acc - binding_acc) >= min_spread

    if not clause1:
        reason = "easy_below_min"
    elif not clause2:
        reason = "no_binding_tasks"
    elif not clause3:
        reason = "insufficient_spread"
    else:
        reason = "separates"

    return {
        "easy_acc": easy_acc,
        "binding_acc": binding_acc,
        "h_ceiling": easy_acc,
        "separates": clause1 and clause2 and clause3,
        "reason": reason,
    }
