import random
from .ruleset import DEFAULT_RULES


class Code:
    """Represents the secret code for the Mastermind game."""

    def __init__(self, sequence=None, rules=None):
        """
        Initialize a Code instance.

        Args:
            code (list[str] or None): The list of color symbols representing the code.
            rules (dict or None): Reference to the ruleset (defines length, colors, duplicates, etc.).
        """
        # Use default rules if
        self.rules = rules or DEFAULT_RULES
        # --- Input normalization ---
        if isinstance(sequence, str):
            self.sequence = [c.upper() for c in sequence.replace(" ", "")]
        elif sequence is None:
            self.sequence = []
        else:
            self.sequence = [c.upper() for c in sequence]

        self.is_valid = False
        if self.sequence:
            self.is_valid = self.validate()

    def generate_random(self):
        """Generate a random valid code according to the rules."""
        colors = self.rules["colors"]
        length = self.rules["code_length"]
        allow_dup = self.rules["allow_duplicates"]

        if allow_dup:
            self.sequence = random.choices(colors, k=length)
        else:
            self.sequence = random.sample(colors, k=length)

        # Safety check
        if not self.validate():
            raise ValueError("Generated code violates rules constraints.")

    def validate(self, strict: bool = True) -> bool:
        """Validate the current code (length, colors, duplicates)."""

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

    def compare_with(self, guess=None) -> tuple[int, int]:
        """
        Compare this code with a Guess object.

        Returns:
            tuple[int, int]: (black_pegs, white_pegs)
        """
        black = 0
        white = 0

        remaining_code = self.sequence[:]
        remaining_guess = guess.sequence[:]

        # count blacks
        for i in range(len(self.sequence)):
            if self.sequence[i] == guess.sequence[i]:
                black += 1
                remaining_guess[i] = None
                remaining_code[i] = None

        # count white
        for color in remaining_guess:
            if color and color in remaining_code:
                white += 1
                remaining_code[remaining_code.index(color)] = None

        return (black, white)

    def get_length(self):
        """Return the code length."""

        return self.rules["code_length"]

    def get_available_colors(self):
        """Return the list of allowed colors from the rules."""

        return self.rules["colors"]

    def as_string(self):
        """Return a string representation of the code (e.g. 'RGBY')."""
        return "".join(self.sequence) if self.sequence else "EMPTY"

    def __eq__(self, other):
        """Check equality between two Code objects."""
        if isinstance(other, Code):
            return self.sequence == other.sequence
        if isinstance(other, list):
            return self.sequence == other
        return False

    def __repr__(self):
        """Return a developer-friendly representation of the Code."""
        code_str = self.as_string()
        rules_name = self.rules.get("name", "unnamed") if self.rules else "none"
        return (
            f"<Code sequence={code_str} length={len(self.sequence)} rules={rules_name}>"
        )

    def __str__(self):
        "Returns the guess sequence."
        return self.as_string()
