from secret_code import Code
from guess import Guess
from ruleset import DEFAULT_RULES
import random as rnd


class Board:
    """Main game board class — manages gameplay, secret code, and guess history."""

    def __init__(self, rules=None):
        """Initialize the board with a given ruleset."""
        self.rules = rules or DEFAULT_RULES
        self.secret_code = None
        self.guesses = []
        self.current_attempt = 0
        self.max_attempts = rules.get("max_attempts", 10)
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
            print("Invalid guess. Try again.")
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

    def get_current_state(self):
        """Return a GameState snapshot for saving or analysis."""
        pass

    def reveal_code(self):
        """Return the secret code (used at the end of the game)."""
        return self.secret_code.as_string()

    def reset(self):
        """Reset the board for a new game with the same rules."""
        self.initialize_game()

    def remaining_attempts(self):
        """Return how many guesses are left."""
        return max(0, self.max_attempts - self.current_attempt)

    def render(self):
        """Render a text-based representation of the board (for CLI)."""
        pass


if __name__ == "__main__":
    b = Board(rules=DEFAULT_RULES)
    won = False
    for i in range(10000):
        b.initialize_game()
        for j in range(10):
            b.make_guess(
                [
                    rnd.choice(b.rules["colors"]),
                    rnd.choice(b.rules["colors"]),
                    rnd.choice(b.rules["colors"]),
                    rnd.choice(b.rules["colors"]),
                ]
            )
            if b.is_over:
                print(b.get_feedback_history())
                print(b.reveal_code())
            if b.is_won:
                print("WOOOOOOOOONN")
                print(i)
                won = True
                break
        if won:
            break
