# state/game_state.py


class GameState:
    """Container for Mastermind game snapshot"""

    def __init__(
        self, rules, guesses, current_attempts, is_over, is_won, code=None
    ):
        self.rules = rules
        self.guesses = guesses
        self.current_attempts = current_attempts
        self.is_over = is_over
        self.is_won = is_won
        self.secret_code = code

    def to_dict(self, reveal_code=False):
        # Return the gamestate as dictionary for i.e. json
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
        # Load the gamestate from a dictionary
        return cls(
            rules=data["rules"],
            guesses=data["guesses"],
            current_attempts=data["current_attempts"],
            is_over=data["is_over"],
            is_won=data["is_won"],
            code=data["secret_code"],
        )
