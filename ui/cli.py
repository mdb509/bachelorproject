# # Command-line interface (text-based play)

from game.board import Board
from game.ruleset import DEFAULT_RULES


def gameloop():
    print("=== Mastermind CLI ===")
    print(
        "Type colors as letters (e.g. RGBY). Type 'exit' to quit, 'save' to store, 'load' to resume.\n"
    )

    b = Board(rules=DEFAULT_RULES)
    b.initialize_game()

    while not b.is_over:
        print(f"\nAttempts left: {b.remaining_attempts()}")
        print(f"Available colors: {', '.join(b.rules['colors'])}")
        user_input = input("Enter your guess: ").strip().upper()

        # handle special commands
        if user_input == "EXIT":
            print("Exiting game.")
            break
        elif user_input == "SAVE":
            b.save()
            print("Game saved.")
            continue
        elif "LOAD" in user_input:
            try:
                b = Board.from_file(user_input.split(" ")[1].lower())
                print("Game loaded.")
                continue
            except Exception as e:
                print(f"Error loading save: {e}")
                continue

        # Make the guess
        try:
            b.make_guess(list(user_input))
        except ValueError as e:
            print(f"Invalid input: {e}")
            continue

        # Render current board
        b.render()

        # Check win/loss
        if b.is_won:
            print("\nCongratulations, you cracked the code!")
            print(f"The secret code was: {b.reveal_code()}")
            break
        elif b.is_over:
            print("\nNo more attempts left.")
            print(f"The secret code was: {b.reveal_code()}")
            break

    print("\n=== Game Over ===")
