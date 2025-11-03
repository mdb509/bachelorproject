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
        "ordered_feedback": False,  # If True, feedback pegs correspond to positions (rare variant)
    },
    "colors": [
        "R",
        "G",
        "B",
        "Y",
        "O",
        "P",
    ],  # Default color set ( Red, Green, Blue, Yellow, Orange, Purple)
    "display": {
        "emoji_map": {  # Optional — for CLI or GUI rendering
            "R": "🔴",
            "G": "🟢",
            "B": "🔵",
            "Y": "🟡",
            "O": "🟠",
            "P": "🟣",
        }
    },
    "variant_flags": {
        "reverse_mode": False,  # If True, computer guesses your code
        "no_white_feedback": False,  # Simplified "hard" variant
        "time_limit": None,  # Optional timer (seconds) for each guess
    },
}
