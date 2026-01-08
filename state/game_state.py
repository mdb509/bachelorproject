# state/game_state.py
from game.ruleset import DEFAULT_RULES
from game.guess import Guess
from game.secret_code import Code


class GameState:
    """
    Container for Mastermind game snapshot

    Attributes:
        rules (dict): The ruleset for the game.
        guesses (list[Guess]): List of past guesses.
        current_attempts (int): The number of attempts made so far.
        is_over (bool): Whether the game is over.
        is_won (bool): Whether the game has been won.
        secret_code (list[str] | None): The secret code sequence, if revealed.
    """

    def __init__(
        self,
        rules=None,
        guesses=None | list[Guess],
        current_attempts=0,
        is_over=False,
        is_won=False,
        code=Code().generate_random(),
    ):
        """Initialize a new GameState instance."""
        self.rules = rules or DEFAULT_RULES
        self.guesses = guesses if guesses is not None else []
        self.current_attempts = current_attempts
        self.is_over = is_over
        self.is_won = is_won
        self.secret_code = code

    def to_dict(self, reveal_code=False):
        """Convert the game state to a dictionary representation."""
        return {
            "rules": self.rules,
            "guesses": [
                {"guess": g.get_guess(), "feedback": g.get_feedback()}
                for g in self.guesses
            ],
            "current_attempts": self.current_attempts,
            "is_over": self.is_over,
            "is_won": self.is_won,
            "secret_code": self.secret_code,
        }

    @classmethod
    def from_dict(cls, data):
        """
        Create a GameState instance from a dictionary.

        Attributes:
            data (dict): The dictionary containing game state data.
        Returns:
            GameState: The constructed GameState instance."""
        guesses = []
        for g in data["guesses"]:
            guess_obj = Guess(g["guess"])
            guess_obj.apply_feedback(g["feedback"])
            guesses.append(guess_obj)
        return cls(
            rules=data["rules"],
            guesses=guesses,
            current_attempts=data["current_attempts"],
            is_over=data["is_over"],
            is_won=data["is_won"],
            code=data["secret_code"],
        )
