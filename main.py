from __future__ import annotations

import os
import tempfile
import threading
import time
from itertools import product

from game.board import Board
from game.ruleset import DEFAULT_RULES
from solver.constraint_builder import Cnf
from solver.dinmacs import Dinmacs
from solver.solver_manager import MinimaxConfig, MinimaxSolver
from solver.solver_interface import SatSolverInterface
from state.persistence import save_state


# Config
code_length = DEFAULT_RULES["code_length"]
num_colors = DEFAULT_RULES["num_colors"]
colors = DEFAULT_RULES["colors"]
backend = "dualiza"  # "dualiza" | "ganak" | "bc_enum"
rounds_to_play = 10

# Create symbol variables
# e.g. R1, R2, ..., Y4
symbol_variable = [
    c + str(j) for j in range(1, code_length + 1) for c in colors
]
# Create dualizer variables
# e.g. x1, x2, ..., x24
dualier_variable = [
    "x" + str(i) for i in range(1, (code_length * num_colors) + 1)
]
# Create decode map
# e.g. x1 -> R1, x2 -> R2, ..., x24 -> P4
decode_variable_map = {
    dualier_variable[i]: symbol_variable[i]
    for i in range(code_length * num_colors)
}

# Dualiza arguments
projected_variables = ",".join(
    str(i) for i in range(1, (code_length * num_colors) + 1)
)
# Dualiza counting arguments
dualiza_counting = ["-c", "-r", projected_variables]
# Dualiza minimal prep arguments
dualiza_minimal_prep_args = ["-c",
    "-r", projected_variables,
    "--project=1",
    "--dual=0",
    "--discount=0",
    "--block=0",
    "--elim=0",
    "--subsume=0",
    "--sublearned=0",]

# unweighted Ganak arguments
default_ganak_args = ["--mode", "0"]
# approximate unweighted Ganak arguments
approx_ganak_args = ["--mode", "0", "--appmct", "1"]
# disable probabilistic techniques ganak
no_prob_ganak_args = ["--mode", "0", "--prob", "0"]
# minimal prep args ganak
minimal_prep_ganak_args = ["--mode", "0",
                            "--arjun", "0",
                            "--puura", "0",
                            "--td", "false",
                            "--bce", "false",
                            "--vivif", "0",
                            "--sbvasteps", "0",
                            "--extraoracle", "false"
        ]

tool_args = dualiza_minimal_prep_args

# All possible feedbacks
feedbacks = [
    (b, w)
    for b in range(code_length + 1)
    for w in range(code_length + 1 - b)
    if not (b == 3 and w == 1)
]

# All possible code combinations
all_combinations = list(product(colors, repeat=code_length))



if __name__ == "__main__":
    # Use /dev/shm if available for faster temp file operations
    tmp_root = "/dev/shm" if os.path.isdir("/dev/shm") else None

    # Statistics
    needed_attempts = []
    times = []
    counter = 0
    # Main game loop
    while  counter < 1:
        # temp directory for Dinmacs and solver files
        with tempfile.TemporaryDirectory(prefix="cnfs_", dir=tmp_root) as tmpdir:
            # Auto-play multiple games and collect statistics
            counter += 1
            start_time = time.perf_counter()

            # Initialize game board
            board = Board()
            # board.initialize_game(all_combinations[counter - 1])
            board.initialize_game()
            print(board.reveal_code())  # debug

            # Save initial state
            game_state = board.get_current_state()
            save_state(game_state, "auto_save.json")

            # Initialize encoder, Dinmacs, solver API, and Minimax solver
            encoder = Cnf(game_state=game_state)
            dinmacs = Dinmacs()
            dinmacs_lock = threading.Lock()
            solver_api = SatSolverInterface()
            solver_api.build_tool("dualiza")
            # solver_api.build_tool("bc_enum")
            mm = MinimaxSolver(
                config=MinimaxConfig(backend=backend),
                solver_api=solver_api,
                dinmacs=dinmacs,
                dinmacs_lock=dinmacs_lock,
                tmpdir=tmpdir,
                colors=colors,
                code_length=code_length,
                num_colors=num_colors,
                decode_variable_map=decode_variable_map,
                tool_args=tool_args,
                feedbacks=feedbacks,
            )

            # Game loop
            for round_idx in range(DEFAULT_RULES["max_attempts"]):
                print(f"\n--- Iteration {round_idx + 1} ---\n")

                base_clauses = encoder.build_constraints_for_history()
                # Encode base clauses with Dinmacs
                with dinmacs_lock:
                    encoded_base_clauses = dinmacs.encode_clauses(
                        base_clauses
                    )

                # Choose the next guess using Minimax strategy
                best_guess = mm.choose_guess(
                    encoder=encoder,
                    encoded_base_clauses=encoded_base_clauses,
                    progress=True,
                )

                # Make the guess on the board
                board.make_guess(list(best_guess))

                # Render the board
                board.render()

                # Save current state and set up encoder for next iteration
                game_state = board.get_current_state()
                save_state(game_state, "auto_save.json")
                encoder = Cnf(game_state=game_state)

                # Check for win or game over
                if board.is_won or board.is_over:
                    break

            # Game ended, record time taken
            end_time = time.perf_counter()
            times.append(end_time - start_time)
            print(f"\nTotal time taken: {end_time - start_time:.2f} seconds.")
            print(f"The secret code was: {board.reveal_code()}")

            # Print game result
            if board.is_won:
                print("\nCongratulations, you cracked the code!")
            elif board.is_over:
                print("\nGame over! You've used all attempts.")

            needed_attempts.append(board.current_attempt)
        
        # Print overall statistics
        avg_time = sum(times) / len(times)
        print(
            f"\nAverage time over {len(times)} games: {avg_time:.2f} seconds."
        )
        max_time = max(times)
        print(f"Max time over {len(times)} games: {max_time:.2f} seconds.")
        min_time = min(times)
        print(f"Min time over {len(times)} games: {min_time:.2f} seconds.")
        avg_attempts = sum(needed_attempts) / len(needed_attempts)
        print(f"Average attempts over {len(needed_attempts)} games: {avg_attempts:.2f} attempts.")
        max_attempts = max(needed_attempts)
        print(f"Max attempts over {len(needed_attempts)} games: {max_attempts} attempts.")
        min_attempts = min(needed_attempts)
        print(f"Min attempts over {len(needed_attempts)} games: {min_attempts} attempts.")
