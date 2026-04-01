"""Parse confidence and answer from model completions."""
import re


_CONF_PATTERN = re.compile(
    r"(?i)confidence\s*:\s*(-?[0-9]*\.?[0-9]+)\s*(%)?", re.IGNORECASE
)
_ANS_PATTERN = re.compile(
    r"(?i)answer\s*:\s*(.*)", re.IGNORECASE | re.DOTALL
)


def parse_response(text: str) -> tuple[float | None, str | None]:
    """Extract (confidence, answer) from a model completion.

    Returns (None, ...) if confidence is missing, (..., None) if answer is missing.
    Confidence is clamped to [0, 1]. Percentages (e.g. "85%") are converted.
    """
    # Parse confidence
    conf_match = _CONF_PATTERN.search(text)
    if conf_match:
        r = float(conf_match.group(1))
        if conf_match.group(2):  # percentage
            r /= 100.0
        r = max(0.0, min(1.0, r))
    else:
        r = None

    # Parse answer: everything after "ANSWER:" to end of string
    ans_match = _ANS_PATTERN.search(text)
    if ans_match:
        a = ans_match.group(1).strip()
        a = a if a else None
    else:
        a = None

    return r, a
