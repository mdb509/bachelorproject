from .secret_code import Code
from .guess import Guess
from .ruleset import DEFAULT_RULES
from state.game_state import GameState
from state.persistence import *

import random as rnd


class Board:
    """Main game board class — manages gameplay, secret code, and guess history."""

    def __init__(self, rules=None):
        """Initialize the board with a given ruleset."""
        self.rules = rules or DEFAULT_RULES
        self.secret_code = Code(rules=self.rules)
        self.guesses = []
        self.current_attempt = 0
        self.max_attempts = self.rules.get("max_attempts", 10)
        self.is_over = False
        self.is_won = False

    def initialize_game(self):
        """Set up a new game: generate a secret code and reset state."""
        self.secret_code = Code(rules=self.rules)
        self.secret_code.generate_random()
        self.guesses = []
        self.current_attempt = 0
        self.is_over = False
        self.is_won = False

    def make_guess(self, guess_input):
        """Create a Guess object from user input, evaluate it, and update state."""

        # Create new guess
        new_guess = Guess(guess_input, rules=self.rules)

        # Check if new guess is falis
        if not new_guess.is_valid:
            print("Invalid input. Try again.")
            return None

        # Calculate feedback
        feedback = self.secret_code.compare_with(new_guess)
        new_guess.apply_feedback(feedback)

        # Save feedback
        self.guesses.append(new_guess)

        # Increase attempts
        self.current_attempt += 1

        # Validate win/lose
        self.check_game_over()

    def get_feedback_history(self):
        """Return the full history of guesses and feedback."""
        result = []
        for guess in self.guesses:
            result.append((guess.get_guess(), (guess.get_feedback())))

        return result

    def check_game_over(self):
        """Check if the game is finished (win or all attempts used)."""
        if self.remaining_attempts() <= 0:
            self.is_over = True
            return

        last_guess = self.guesses[-1]
        black = last_guess.get_feedback()[0]
        if black == self.rules["code_length"]:
            self.is_over = True
            self.is_won = True
            return

    def reveal_code(self):
        """Return the secret code (used at the end of the game)."""
        return self.secret_code.as_string()

    def reset(self):
        """Reset the board for a new game with the same rules."""
        self.initialize_game()

    def remaining_attempts(self):
        """Return how many guesses are left."""
        return max(0, self.max_attempts - self.current_attempt)

    def get_current_state(self):
        """Return a GameState snapshot for saving or analysis."""
        return GameState(
            rules=self.rules,
            guesses=[g for g in self.guesses],
            current_attempts=self.current_attempt,
            is_over=self.is_over,
            is_won=self.is_won,
            code=self.secret_code.as_string(),
        )

    def save(self, filename="game_state.json"):
        save_state(self.get_current_state(), filename)

    @classmethod
    def from_file(cls, filename):
        state = load_state(filename)
        board = cls(rules=state.rules)
        board.guesses = []
        for g in state.guesses:
            guess_obj = Guess(g["guess"], rules=board.rules)
            guess_obj.apply_feedback((g["feedback"][0], g["feedback"][1]))
            board.guesses.append(guess_obj)
        board.current_attempt = state.current_attempts
        board.is_over = state.is_over
        board.is_won = state.is_won
        board.secret_code = Code(state.secret_code, rules=board.rules)
        return board

    def render(self, width=8):
        """Render a text-based representation of the board (for CLI)."""

        colors = self.rules["display"]["emoji_map"]
        title = "| +++++++++++++ Mastermind ++++++++++++ |"
        colums = "| ++++ Guesses ++++ | ++++ Feedback +++ |"
        line = "+----" * width + "+"

        # build the gameboard
        print(line)
        print(title)
        print(line)
        print(colums)
        print(line)
        guess = self.guesses[-1] if len(self.guesses) > 0 else None
        for guess in self.guesses:
            attempt_line = ""
            for c in guess.get_guess():
                attempt_line += "| " + colors[c] + " "
            black, white = guess.get_feedback()
            for i in range(black):
                attempt_line += "| " + colors["BK"] + " "
            for i in range(white):
                attempt_line += "| " + colors["W"] + " "
            for i in range(max(0, 4 - black - white)):
                attempt_line += "|    "
            print(attempt_line + "|")
            print(line)
