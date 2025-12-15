import random
from .ruleset import DEFAULT_RULES


class Code:
    """
        Represents the secret code for the Mastermind game.
    Attributes:
        sequence (list[str]): The sequence of colors representing the code.
        rules (dict): The ruleset for validation.
        is_valid (bool): Whether the code is valid according to the rules."""

    def __init__(self, sequence=None, rules=None):
        """
        Initialize a Code instance.

        Args:
            sequence (list or None): The list of color symbols representing
            the code.
            rules (dict or None): Reference to the ruleset (defines length,
            colors, duplicates, etc.).
        """

        self.rules = rules or DEFAULT_RULES
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
        """
        Generate a random valid code according to the rules.
        """

        colors = self.rules["colors"]
        length = self.rules["code_length"]
        allow_dup = self.rules["allow_duplicates"]

        # Generate the secret code depending on whether duplicates are allowed.
        if allow_dup:
            self.sequence = random.choices(colors, k=length)
        else:
            self.sequence = random.sample(colors, k=length)

        # Validate the generated code
        if not self.validate():
            raise ValueError("Generated code violates rules constraints.")

    def validate(self, strict: bool = True) -> bool:
        """
        Validate the current code (length, colors, duplicates).

        Args:
            strict (bool): If True, raise ValueError with an explanatory
            message when validation fails. If False, return False on failure.

        Returns:
            bool: True if the code sequence is valid; False if invalid and
            strict is False.
        """

        def fail(msg: str) -> bool:
            """
            Handle failure depending on strict mode.

            Args:
                msg (str): Error Message, which will be printed.

            Returns:
                bool: Returns always false. Is used, when validation fails.
            """

            if strict:
                raise ValueError(msg)
            return False

        # Validates, if code sequence length is as declared in the rules,
        if len(self.sequence) != self.rules["code_length"]:
            return fail(
                f"Code length must be {self.rules['code_length']}, "
                f"but got {len(self.sequence)}."
            )

        # Validates if code sequence has no duplicates, when its not allowed.
        if not self.rules.get("allow_duplicates", True) and len(
            set(self.sequence)
        ) != len(self.sequence):
            return fail("Duplicates are not allowed in this ruleset.")

        # Validates if code sequence only contains colors as in the rules.
        for color in self.sequence:
            if color not in self.rules["colors"]:
                allowed = ", ".join(self.rules["colors"])
                return fail(f"Invalid color '{color}'. Allowed: {allowed}.")

        return True

    def compare_with(self, guess=None) -> tuple[int, int]:
        """
        Compare this secret code with a Guess object and compute
        Mastermind-style feedback.

        Args:
            guess (Guess): A Guess instance whose .sequence will be compared
            against this code.

        Returns:
            tuple[int, int]: (black, white)
            black — number of pegs with correct color in the correct position,
            white — number of pegs with correct color but in the wrong
            position.

        Notes:
            Positions counted as black are excluded from white-counting to
            avoid double-counting.
        """

        black = 0
        white = 0

        remaining_code = self.sequence[:]
        remaining_guess = guess.sequence[:]

        # Compare secrete code with guess. Count the color and position.
        for i in range(len(self.sequence)):
            if self.sequence[i] == guess.sequence[i]:
                black += 1
                remaining_guess[i] = None
                remaining_code[i] = None

        # Compare secrete code sequence with guess. Count the color.
        for color in remaining_guess:
            if color and color in remaining_code:
                white += 1
                remaining_code[remaining_code.index(color)] = None

        return (black, white)

    def get_length(self):
        """
        Return the length of the secrete code sequence from the rules.

        Returns:
            int: The length of the secret code sequence.
        """

        return self.rules["code_length"]

    def get_available_colors(self):
        """
        Return the list of allowed colors from the rules.

        Returns:
            list[str]: The list of allowed color symbols."""

        return self.rules["colors"]

    def as_string(self):
        """
        Return a string representation of the code (e.g. 'RGBY').
        Returns:
            str: The code as a string.
        """
        return "".join(self.sequence) if self.sequence else "EMPTY"

    def __eq__(self, other):
        """
        Check equality between this Code and another object.

        Args:
            other (Code or list): A Code instance or a list to compare against.

        Returns:
            bool: True if the sequences are equal, False otherwise.
        """

        if isinstance(other, Code):
            return self.sequence == other.sequence
        if isinstance(other, list):
            return self.sequence == other
        return False

    def __str__(self):
        """
        Returns the code sequence.
        Returns:
            str: The code as a string.
        """
        return self.as_string()
