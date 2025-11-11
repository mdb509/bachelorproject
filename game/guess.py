from .ruleset import DEFAULT_RULES


class Guess:
    """Represents a single player guess in the Mastermind game."""

    def __init__(self, sequence, rules=None):
        """Initialize a Guess instance."""

        # --- Input normalization ---
        if isinstance(sequence, str):
            self.sequence = [c.upper() for c in sequence.replace(" ", "")]
        elif sequence is None:
            self.sequence = []
        else:
            self.sequence = [c.upper() for c in sequence]

        # --- Attribute setup ---
        self.rules = rules or DEFAULT_RULES
        self.black_pegs = None
        self.white_pegs = None
        self.is_valid = False

        # --- Validation ---
        if self.sequence:
            self.is_valid = self.validate(strict=False)

    def validate(self, strict: bool = True):
        """Check if the guess follows the rules (length, valid colors, duplicates)."""

        def fail(msg: str) -> bool:
            """Handle failure depending on strict mode."""
            if strict:
                raise ValueError(msg)
            return False

        # Length check
        if len(self.sequence) != self.rules["code_length"]:
            return fail(
                f"Code length must be {self.rules['code_length']}, "
                f"but got {len(self.sequence)}."
            )

        # Duplicate check
        if not self.rules.get("allow_duplicates", True) and len(
            set(self.sequence)
        ) != len(self.sequence):
            return fail("Duplicates are not allowed in this ruleset.")

        # Color check
        for color in self.sequence:
            if color not in self.rules["colors"]:
                allowed = ", ".join(self.rules["colors"])
                return fail(f"Invalid color '{color}'. Allowed: {allowed}.")

        return True

    def apply_feedback(self, feedback: tuple[int, int]):
        """Store feedback values after evaluation by the Board/Code."""
        self.black_pegs = feedback[0]
        self.white_pegs = feedback[1]

    def get_feedback(self):
        """Return the stored feedback as a tuple (black_pegs, white_pegs)."""
        return (self.black_pegs, self.white_pegs)

    def get_guess(self):
        """Return the stored guess."""
        return self.sequence

    def as_string(self):
        """Return a string representation of the guess (e.g. 'RGBY')."""
        return "".join(self.sequence) if self.sequence else "EMPTY"

    def __repr__(self):
        """Return a developer-friendly representation of the Guess."""
        guess_str = self.as_string()
        rules_name = self.rules.get("name", "unnamed") if self.rules else "none"
        return f"<Guess sequence={guess_str} length={len(self.sequence)} rules={rules_name}>"

    def __str__(self):
        "Returns the guess sequence."
        return self.as_string()
