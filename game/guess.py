from .ruleset import DEFAULT_RULES


class Guess:
    """
        Represents a single player guess in the Mastermind game.
    Attributes:
        sequence (list[str]): The guessed sequence of colors.
        rules (dict): The ruleset for validation.
        black_pegs (int | None): Number of correct colors in correct positions.
        white_pegs (int | None): Number of correct colors in wrong positions.
        is_valid (bool): Whether the guess is valid according to the rules."""

    def __init__(self, sequence: list[str] | str | None, rules=None):
        """
        Initialize a Guess instance.
        Args:
            sequence (list[str] | str | None): The guessed sequence.
            rules (dict, optional): The ruleset for validation. Defaults to DEFAULT_RULES.
        """

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
        """
        Check if the guess follows the rules
        (length, valid colors, duplicates).

        Args:
            strict (bool): If True, raise ValueError on invalid guess.
        Returns:
            bool: True if valid, False otherwise.
        """

        def fail(msg: str) -> bool:
            """
            Handle failure depending on strict mode.
            Args:
                msg (str): The error message.
            Returns:
                bool: Always False.
            """
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
        """
        Store feedback values after evaluation by the Board/Code.
        Args:
            feedback (tuple[int, int]): (black_pegs, white_pegs)
        """
        self.black_pegs = feedback[0]
        self.white_pegs = feedback[1]

    def get_feedback(self):
        """
        Return the stored feedback as a tuple (black_pegs, white_pegs).
        Returns:
            tuple[int, int]: The feedback tuple.
        """
        return (self.black_pegs, self.white_pegs)

    def get_guess(self):
        """
        Return the stored guess.

        Returns:
            list[str]: The guess sequence."""
        return self.sequence

    def as_string(self):
        """
        Return a string representation of the guess (e.g. 'RGBY').
        Returns:
            str: The guess as a string."""
        return "".join(self.sequence) if self.sequence else "EMPTY"

    def __str__(self):
        """
        String representation of the Guess instance.
        Returns:
            str: The guess as a string."""
        return self.as_string()
