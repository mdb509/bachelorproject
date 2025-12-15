from __future__ import annotations

import os
import tempfile
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from pathlib import Path

from solver.constraint_builder import Cnf
from solver.dinmacs import Dinmacs
from solver.sat_solver_interface import SatSolverInterface
from game.board import Board
from game.guess import Guess
from state.persistence import save_state


# -----------------------------
# Helpers
# -----------------------------
symbol_variable = [c+str(j) for j in range(1, 5) for c in ["R", "G", "B", "Y", "O", "P"] ]
dualier_variable = ["x"+str(i) for i in range(1, 25)]
decode_variable_map = { dualier_variable[i]: symbol_variable[i] for i in range(24) }



def parse_dualiza_count(stdout: str) -> int:
    """
    Count solutions from Dualiza output.
    Per line we have a string of variables, collect them. 
    count lines.
    Return tuple of list variable and count.
    """

    count = 0
    solutions = []
    lines = stdout.splitlines()

    if lines[0].startswith("s UNSATISFIABLE"):
        return solutions, 0
    elif lines[0].startswith("ALL"):
        lines = lines[1:]  # skip first line
        for line in lines:
            line = line.strip()
            solutions.append(tuple([decode_variable_map[var] for var in line.split(" ") if not var.startswith("!")]))
            count += 1
    elif lines[0].startswith("NUMBER"):
        count = int(lines[1].strip())
    return solutions, count

def evaluate_guess(
    guess_sequence: tuple[str, str, str, str],
    feedbacks: list[tuple[int, int]],
    encoded_base_clauses,
    encoder: Cnf,
    dinmacs: Dinmacs,
    dinmacs_lock: threading.Lock,
    solver_interface: SatSolverInterface,
    tmpdir: str,
    dualiza_args: list[str],
    guess_index: int,
) -> tuple[tuple[str, str, str, str], int, tuple[int, int] | None]:
    """
    Evaluate one guess in minimax style:
    For each possible feedback, compute #SAT (restricted), then take the maximum.
    Returns: (guess_sequence, worst_case_count, worst_case_feedback).
    """

    # One CNF file per thread => no file write races
    tid = threading.get_ident()
    cnf_path = Path(tmpdir) / f"thread_{tid}.cnf"

    worst_case = 0
    worst_fb: tuple[int, int] | None = None

    for fb in feedbacks:
        # IMPORTANT: create a fresh Guess per feedback to avoid accumulating state
        g = Guess(list(guess_sequence))
        g.apply_feedback(feedback=fb)

        # Build symbolic constraints for this (guess, feedback)
        claus = encoder.build_constraints(g, guess_index=guess_index)

        # Encode + write CNF.
        # We lock because Dinmacs may have internal mutable name->id mapping.
        with dinmacs_lock:
            encoded_delta = dinmacs.encode_clauses(claus)
            solv_clauses = encoded_base_clauses + encoded_delta
            dinmacs.build_dinmacs(str(cnf_path), guess_index=guess_index, clauses=solv_clauses)

        # Run Dualiza (#SAT)
        out = solver_interface.run_dualiza(str(cnf_path), extra_args=dualiza_args)
        solutions, count = parse_dualiza_count(out)

        # Track worst case
        if count > worst_case:
            worst_case = count
            worst_fb = fb
            best_solutions = solutions

    return guess_sequence, worst_case, worst_fb, best_solutions


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    # Prefer RAM-backed temp directory on Linux for faster CNF writes
    tmp_root = "/dev/shm" if os.path.isdir("/dev/shm") else None

    with tempfile.TemporaryDirectory(prefix="cnfs_", dir=tmp_root) as tmpdir:
        start_time = time.perf_counter()
        board = Board()
        board.initialize_game()
        print(board.reveal_code())  # (debug) reveals secret

        game_state = board.get_current_state()
        save_state(game_state, "auto_save.json")

        encoder = Cnf(game_state=game_state)
        dinmacs = Dinmacs()

        solver_interface = SatSolverInterface()
        solver_interface.build_dualiza()  # build once (skip later when stable)

        # NOTE: (0,0) is a valid Mastermind feedback.
        # You currently exclude it; keeping your behavior, but consider including it.
        feedbacks = [
            (b, w)
            for b in range(5)
            for w in range(5 - b)
            if not (b == 3 and w == 1)   # impossible for 4 pegs
        ]

        colors = ["R", "G", "B", "Y", "O", "P"]
        all_combinations = list(product(colors, repeat=4))  # tuples of length 4

        # Dualiza args (flat list!)
        dualiza_args = [
            "-c",
            "-r",
            "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24",
        ]

        # Lock to protect Dinmacs internal mapping if it is mutable
        dinmacs_lock = threading.Lock()

        # Adjust this to taste (often best: number of physical cores or cpu_count()-1)
        max_workers = max(1, (os.cpu_count() or 4) - 1)

        for round_idx in range(10):
            print(f"\n--- Iteration {round_idx + 1} ---\n")

            # Build base clauses from game history and encode once per iteration
            base_clauses = encoder.build_constraints_for_history()
            with dinmacs_lock:
                encoded_base_clauses = dinmacs.encode_clauses(base_clauses)

            min_max_assignments = float("inf")
            best_guess: tuple[str, str, str, str] | None = None
            best_guess_worst_fb: tuple[int, int] | None = None

            start = time.perf_counter()
            last_report = start
            done = 0
            total = len(all_combinations)

            # Parallelize across guesses
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [
                    pool.submit(
                        evaluate_guess,
                        guess_sequence=gs,
                        feedbacks=feedbacks,
                        encoded_base_clauses=encoded_base_clauses,
                        encoder=encoder,
                        dinmacs=dinmacs,
                        dinmacs_lock=dinmacs_lock,
                        solver_interface=solver_interface,
                        tmpdir=tmpdir,
                        dualiza_args=dualiza_args,
                        guess_index=0,  # keep your current convention (consider using round_idx if your encoding needs it)
                    )
                    for gs in all_combinations
                ]

                for fut in as_completed(futures):
                    gs, worst_case, worst_fb, best_solutions = fut.result()
                    done += 1

                    # Progress log every ~2 seconds
                    now = time.perf_counter()
                    if now - last_report >= 2.0:
                        rate = done / max(1e-9, (now - start))
                        print(f"\r\033[KProgress: {done}/{total} guesses ({rate:.1f} guesses/sec)", end="", flush=True)
                        # \033[K = clear to end of line (funktioniert in den meisten Terminals)
                        last_report = now

                    # Minimize the maximum (#SAT) over feedbacks
                    if 0 < worst_case < min_max_assignments:
                        min_max_assignments = worst_case
                        best_guess = gs
                        best_guess_worst_fb = worst_fb
                        min_max_solutions = best_solutions
                        print(
                            f"Best guess :{best_guess}\n"
                            f"with fb {best_guess_worst_fb}\n"
                            f"with solutions: {min_max_assignments}\n"
                            )

            print(
                f"Selected guess: {best_guess} | minimax worst-case #SAT: {min_max_assignments} | worst fb (for report): {best_guess_worst_fb}"
            )

            if best_guess is None:
                print("No valid guess found (all worst_case counts were 0?). Stopping.")
                break

            # IMPORTANT: the board determines the *real* feedback, not the minimax worst feedback
            if min_max_assignments == 1:
                board.make_guess(list(best_guess))

                solution = min_max_solutions.pop()
                dualiza_args[0] = "-e"  # get actual solution
                gs, wc, wfb, best_solutions = evaluate_guess(
                    guess_sequence=best_guess,
                    feedbacks=feedbacks,
                    encoded_base_clauses=encoded_base_clauses,
                    encoder=encoder,
                    dinmacs=dinmacs,
                    dinmacs_lock=dinmacs_lock,
                    solver_interface=solver_interface,
                    tmpdir=tmpdir,
                    dualiza_args=dualiza_args,
                    guess_index=0,
                )
                dualiza_args[0] = "-c"  # back to count modus
                if solution in best_solutions:   
                    print(f"Unique solution found: {solution}.")
                    guess = "".join(str(s[0]) for s in solution)
                    board.make_guess(list(guess))
            else:
                board.make_guess(list(best_guess))

            board.render()
            game_state = board.get_current_state()
            save_state(game_state, f"auto_save{round_idx + 1}.json")

            # Update encoder with new history
            encoder = Cnf(game_state=game_state)
           
            if board.is_won and board.is_over:
                print("\nCongratulations, you cracked the code!")
                print(f"The secret code was: {board.reveal_code()}")
                break
            elif board.is_over:
                print("\nGame over! You've used all attempts.")
                print(f"The secret code was: {board.reveal_code()}")
                break
        end_time = time.perf_counter()
        print(f"\nTotal time taken: {end_time - start_time:.2f} seconds.")
