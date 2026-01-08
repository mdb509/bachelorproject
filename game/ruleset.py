from __future__ import annotations
from typing import Dict, List, Optional

# Configuration: colors, code length, duplicates allowed, etc.
DEFAULT_RULES = {
    "name": "classic",  # Identifier for this ruleset
    "code_length": 4,  # Number of pegs in the code
    "num_colors": 6,  # Available colors (see color set below)
    "allow_duplicates": True,  # Can the code contain repeated colors?
    "max_attempts": 10,  # Number of guesses per game
    "feedback": {
        "black_pegs": True,  # Use black pegs for correct color + position
        "white_pegs": True,  # Use white pegs for correct color, wrong position
    },
    "colors": [
        "R", "G", "B", "Y", "O", "P",
    ],
    "display": {
        "emoji_map": {
            "R": "🔴",
            "G": "🟢",
            "B": "🔵",
            "Y": "🟡",
            "O": "🟠",
            "P": "🟣",
            "BK": "⚫",
            "W": "⚪",
        }
    },
}

# 21 colors for extended rulesets
# Existing: R,G,B,Y,O,P,T,C,M,K,N,L,A,S,V,I,F,H,D,E (total 21)
COLOR_POOL: List[str] = [
    "R",  # Red
    "G",  # Green
    "B",  # Blue
    "Y",  # Yellow
    "O",  # Orange
    "P",  # Purple
    "T",  # Brown/Tan
    "C",  # Cyan
    "M",  # Magenta
    "K",  # Black
    "N",  # Navy
    "L",  # Lime
    "A",  # Amber
    "S",  # Silver
    "V",  # Violet
    "I",  # Indigo
    "F",  # Fuchsia
    "H",  # Chartreuse
    "D",  # Gold
    "E",  # Teal
]

# Unified emoji map used by make_ruleset (can be overridden by passing display/colors explicitly)
EMOJI_MAP_21 = {
    # originals
    "R": "🔴",
    "G": "🟢",
    "B": "🔵",
    "Y": "🟡",
    "O": "🟠",
    "P": "🟣",
    "T": "🟤",  # brown circle
    "C": "🔷",  # blue diamond (cyan-ish)
    "M": "🟥",  # red square (magenta-ish)
    "K": "⚫",  # black
    "N": "🔹",  # small blue diamond (navy-ish)
    "L": "🟩",  # green square (lime-ish)
    "A": "🟨",  # yellow square (amber-ish)
    "S": "⬜",  # white square (silver/gray-ish; depends on theme)
    "V": "💜",  # violet heart
    "I": "🟦",  # blue square (indigo-ish)
    "F": "🟪",  # purple square (fuchsia-ish)
    "H": "💚",  # green heart (highly distinguishable)
    "D": "⭐",  # gold-ish star
    "E": "🟧",  # orange square (teal not available; still distinct)

    # feedback pegs
    "BK": "⚫",
    "W": "⚪",
}

def make_ruleset(
    *,
    code_length: int,
    num_colors: int,
    allow_duplicates: bool = True,
    max_attempts: Optional[int] = None,
    colors: Optional[List[str]] = None,
    name: Optional[str] = None,
) -> Dict:
    """
    Create a ruleset configuration dictionary.
    Args:
        code_length (int): The length of the secret code.
        num_colors (int): The number of different colors available.
        allow_duplicates (bool): Whether duplicates are allowed in the code.
        max_attempts (Optional[int]): Maximum number of attempts allowed.
        colors (Optional[List[str]]): List of color identifiers to use.
        name (Optional[str]): Name of the ruleset.
    Returns:
        Dict: The ruleset configuration dictionary.
    """
    # Validate and set colors
    if colors is None:
        if num_colors > len(COLOR_POOL):
            raise ValueError(f"num_colors={num_colors} > COLOR_POOL={len(COLOR_POOL)}")
        colors = COLOR_POOL[:num_colors]
    else:
        if len(colors) != num_colors:
            raise ValueError("len(colors) needs to be == num_colors")

    # Set default max_attempts if not provided
    if max_attempts is None:
        max_attempts = 10 if code_length <= 4 else 20

    # Build emoji map (include all known mappings; renderer can just index by token)
    emoji_map = dict(EMOJI_MAP_21)

    return {
        "name": name or f"n{code_length}_m{num_colors}",
        "code_length": code_length,
        "num_colors": num_colors,
        "allow_duplicates": allow_duplicates,
        "max_attempts": max_attempts,
        "feedback": {"black_pegs": True, "white_pegs": True},
        "colors": colors,
        "display": {"emoji_map": emoji_map},
    }
