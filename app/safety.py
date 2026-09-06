"""Crisis hand-off copy and regex gate. Node 6 reads FALLBACK_COPY only — not Chroma, not the model."""

from __future__ import annotations

import re

SOS_PHONE = "1767"
IMH_PHONE = "6389 2222"
EMERGENCY_SCDF = "995"
EMERGENCY_POLICE = "999"

FALLBACK_COPY = (
    "If you are in immediate danger in Singapore, call 995 (SCDF) or 999 (police). "
    "For suicidal ideation or acute crisis: Samaritans of Singapore (SOS) 1767; "
    "IMH Emergency / helpline 6389 2222. "
    "This navigator cannot provide therapy or diagnosis."
)

# First-person self-harm only. Bare "suicide"/"die" must not skip the SLM
# (e.g. "how do I help my friend who is suicidal").
_CRISIS_RE = re.compile(
    r"(?is)"
    r"(?:\bkill myself\b|"
    r"\bend my life\b|"
    r"\bwant to die\b|"
    r"\bi(?:'m| am)? going to die\b|"
    r"\bi (?:will |going to |gonna )?end it all\b)"
)


def crisis_regex_match(text: str) -> bool:
    """True for obvious first-person self-harm. Subtle / third-person crisis goes to the SLM."""
    normalised = " ".join(text.lower().split())
    return bool(_CRISIS_RE.search(normalised))
