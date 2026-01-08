# main.py
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import numpy as np
import random
from itertools import product
from typing import Any


from game.board import Board
from game.ruleset import *
from state.persistence import load_state, save_state
from solver.constraint_builder import Cnf
from solver.dinmacs import Dinmacs
from solver.solver_manager import MinimaxConfig, MinimaxSolver
from solver.solver_interface import SatSolverInterface


def _load_dataset(path: str) -> dict[str, Any]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    return {
        "meta": {
            "description": "Benchmark dataset for Minimax solver over various (n,m).", 
        },
        "runs": {},
    }



def _write_json(path: str, obj: dict[str, Any]) -> None:
    """
    Write JSON object to file.
    Args:
        path (str): File path.
        obj (dict): JSON-serializable object.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)

def _pad_turn_rows(turn_rows: list[np.ndarray], *, fill: float = np.nan) -> np.ndarray:
    """
    Pad turn rows into a 2D numpy array.
    Rows are padded with `fill` to the length of the longest row.
    Columns represent turns, rows represent games.

    Args:
        turn_rows (list[np.ndarray]): List of 1D numpy arrays representing turn times per game.
        fill (float, optional): Fill value for padding. Defaults to np.nan.
    Returns:
        np.ndarray: 2D numpy array with shape (num_games, max_turns).
    """
    if not turn_rows:
        return np.empty((0, 0), dtype=np.float64)

    max_len = max((r.size for r in turn_rows), default=0)
    mat = np.full((len(turn_rows), max_len), fill, dtype=np.float64)

    for i, r in enumerate(turn_rows):
        if r.size:
            mat[i, : r.size] = r
    return mat

def _table_to_columns(turn_headers: list[str], turn_table: list[list[float | None]]) -> dict[str, list[float | None]]:
    """ 
    Convert a turn table into a columnar dictionary.
    Args:
        turn_headers (list[str]): List of column headers.
        turn_table (list[list[float | None]]): 2D list representing the table.
    Returns:
        dict[str, list[float | None]]: Dictionary mapping headers to column data.
    """
    if not turn_headers:
        return {}
    cols = {h: [] for h in turn_headers}
    for row in turn_table:
        for j, h in enumerate(turn_headers):
            cols[h].append(row[j])
    return cols

def _nan_matrix_to_none_list(mat: np.ndarray) -> list[list[float | None]]:
    """
    Convert a float matrix with NaNs into nested lists with None.
    Args:
        mat (np.ndarray): 2D numpy array with float values and NaNs.
    Returns:
        list[list[float | None]]: Nested list with None in place of NaNs.
    """
    out = mat.tolist()
    for row in out:
        for j, v in enumerate(row):
            # NaN check (v != v) works for floats
            if isinstance(v, float) and v != v:
                row[j] = None
    return out

def benchmark_all_codes(
    n: int,
    m: int,
    solver: str,
    *,
    timeout_s: float = 600.0,
) -> dict[str, Any]:
    """
    Benchmark the Minimax solver for given (n,m).

    Benchmarks total times, attempts, per-turn times, and win rates
    over all possible codes of length n with m colors.

    Args:
        n (int): Code length.
        m (int): Number of colors.
        solver (str): Solver backend to use ("dualiza" | "ganak" | "bc_enum").
        timeout_s (float): Soft timeout for the whole (n,m) run.
    Returns:
        dict: Results for this (n,m) run.
    """
    # Prepare rules and variables
    rules = make_ruleset(code_length=n, num_colors=m)
    code_length = rules["code_length"]
    num_colors = rules["num_colors"]
    colors = rules["colors"][:num_colors]
    rounds_to_play = rules["max_attempts"]

    # Create symbol variables: e.g. R1, G1, ..., R2, ...
    symbol_variable = [c + str(j) for j in range(1, code_length + 1) for c in colors]

    # Create dualizer variables: x1..x_(n*m)
    dualier_variable = ["x" + str(i) for i in range(1, (code_length * num_colors) + 1)]

    # x1 -> R1, ...
    decode_variable_map = {
        dualier_variable[i]: symbol_variable[i]
        for i in range(code_length * num_colors)
    }

    projected_variables = ",".join(str(i) for i in range(1, (code_length * num_colors) + 1))

    # Dualiza minimal args
    dualiza_minimal_prep_args = [
        "-c",
        "-r", projected_variables,
        "--project=1",
        "--dual=0",
        "--discount=0",
        "--block=0",
        "--elim=0",
        "--subsume=0",
        "--sublearned=0",
    ]

    # Ganak minimal-ish args (kept from your code)
    ganak_minimal_prep_args = [
        "--mode", "0",
        "--prob", "0",   
        "--arjun", "0",
        "--puura", "0",
        "--td", "false",
        "--bce", "false",
        "--vivif", "0",
        "--sbvasteps", "0",
        "--extraoracle", "false",
    ]


    # BC_Enum minimal args
    bc_enum_minimal_prep_args = ["./bcp_enum", "-c"]

    # Select tool args based on solver
    match solver:
        case "ganak":
            tool_args = ganak_minimal_prep_args
        case "dualiza":
            tool_args = dualiza_minimal_prep_args
        case "bc_enum":
            tool_args = bc_enum_minimal_prep_args
        case _:
            raise ValueError(f"Unknown solver: {solver}")

    # All possible feedbacks
    feedbacks = [
        (b, w)
        for b in range(code_length + 1)
        for w in range(code_length + 1 - b)
        if not (b == code_length - 1 and w == 1)
    ]

    # Total possible games
    num_games_total = int(m ** n)

    # Use /dev/shm if available
    tmp_root = "/dev/shm" if os.path.isdir("/dev/shm") else None

    # Dynamic per-game data
    turn_rows: list[np.ndarray] = []
    clausle_len_rows: list[np.ndarray] = []
    total_time_list: list[float] = []
    attempts_list: list[int] = []
    won_list: list[bool] = []

    # Keep track of code order for processed games
    games_processed = 0

    # Dinmacs instance + lock
    dinmacs = Dinmacs()
    dinmacs_lock = threading.Lock()

    # Minimax solver setup
    solver_api = SatSolverInterface()
    solver_api.build_tool(solver)

    # Start run timer
    run_start = time.perf_counter()
    # timeout flag
    timeout = False

    with tempfile.TemporaryDirectory(prefix="cnfs_", dir=tmp_root) as tmpdir:
        # Iterate all possible codes randomly
        for code in random.sample(list(product(colors, repeat=code_length)), k=num_games_total):
            # Initialize game board
            board = Board(rules=rules)
            board.initialize_game(code)

            # get game state and encoder
            game_state = board.get_current_state()
            encoder = Cnf(game_state=game_state)
            
            # Minimax solver instance
            minimax_solver = MinimaxSolver(
                config=MinimaxConfig(backend=solver),
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
            # Start game timer
            game_start = time.perf_counter()

            # Per-game turn times
            per_game_turn_times: list[float] = []
            # per-game clause lengths
            per_game_clause_len: list[int] = []

            print(f"\nStarting game {games_processed + 1}/{num_games_total} with code: {code}")
            print(f"Runtime:{time.perf_counter() - run_start:.02f} s")

            # per-game turns
            for _turn_idx in range(rounds_to_play):
                
                # build base clauses
                base_clauses = encoder.build_constraints_for_history()
                with dinmacs_lock:
                    encoded_base_clauses = dinmacs.encode_clauses(base_clauses)

                # run timeout at turn boundary
                if (time.perf_counter() - run_start) > timeout_s:
                    timeout = True
                    break

                # Start turn timer
                turn_start = time.perf_counter()

                
                # choose guess
                best_guess, clausle_len = minimax_solver.choose_guess(
                    encoder=encoder,
                    encoded_base_clauses=encoded_base_clauses,
                    progress=True,
                    timeout_s=timeout_s
                )

                # make guess
                if best_guess is None:
                    print("No valid guess could be found, terminating game.")
                    break

                # play best guess
                board.make_guess(list(best_guess))

                # update encoder
                game_state = board.get_current_state()
                encoder = Cnf(game_state=game_state)

                # record turn time
                per_game_turn_times.append(time.perf_counter() - turn_start)
                # record clause length
                per_game_clause_len.append(clausle_len)

                # check for win/over
                if board.is_won or board.is_over:
                    board.render()
                    break
                
            # end per-game turns
            total_game_duration = time.perf_counter() - game_start

            # store game results (also if timeout hit mid-run)
            turn_rows.append(np.asarray(per_game_turn_times, dtype=np.float64))
            clausle_len_rows.append(np.asarray(per_game_clause_len, dtype=np.int64))
            total_time_list.append(float(total_game_duration))
            attempts_list.append(len(per_game_turn_times))
            won_list.append(bool(board.is_won))

            games_processed += 1
            # limit to 1000 games for benchmarking purposes
            if games_processed >= 1000:
                print("Limiting to 1000 games for benchmarking purposes.")
                break

            # check for run timeout
            if (time.perf_counter() - run_start) > timeout_s:
                timeout = True
                break
    
    # status determination completed vs timeout
    status = "completed" if games_processed == num_games_total else "timeout"

    # numpy arrays
    total_time = np.asarray(total_time_list, dtype=np.float64)
    attempts = np.asarray(attempts_list, dtype=np.int32)
    won = np.asarray(won_list, dtype=np.bool_)

    # dynamic turn table: only up to max observed turns
    turn_mat = _pad_turn_rows(turn_rows, fill=np.nan)
    turn_headers = [f"turn_{i+1}_s" for i in range(turn_mat.shape[1])]
    turn_table = _nan_matrix_to_none_list(turn_mat)
    # dynamic clause length table: only up to max observed turns
    clausle_len_mat = _pad_turn_rows(clausle_len_rows, fill=np.nan)
    clausle_len_table = _nan_matrix_to_none_list(clausle_len_mat)

    return {
        "meta": {
            "code_length": int(n),
            "number_colors": int(m),
            "solver": str(solver),
            "max_attempts": int(rounds_to_play),
            "colors": list(colors),
            "num_games_total": int(num_games_total),
            "num_games_processed": int(games_processed),
            "run_timeout_s": float(timeout_s),
            "status": status,
        },
        "games": {
            "turn_headers": turn_headers,
            "turn_time_s_columns": _table_to_columns(turn_headers, turn_table),
            "clausle_len_table": _table_to_columns(turn_headers, clausle_len_table),
            "total_time_s": total_time.tolist(),
            "attempts": attempts.tolist(),
            "won": won.tolist(),
        },
    }, timeout


def benchmark_grid(
    fixed_n: int | None=None,
    fixed_m: int | None=None,
    n_range: range | None=None,
    m_range: range | None=None,
    solver: str="dualiza",
    out_path: str   ="benchmark_all.json",
    dataset: dict   [str, Any]={},
    run_timeout_s: float    =10.0,
):
    """
    Runs benchmarks over a parameter grid.

    As example, to benchmark all (n,m) with n in 1..8 and m=10, call with
    n_range=range(1,9) and fixed_m=10.

    Args:
        fixed_n (int | None): Fixed code length n. If None, use n_range.
        fixed_m (int | None): Fixed number of colors m. If None, use m_range.
        n_range (range | None): Range of code lengths n to benchmark. If None, use fixed_n.
        m_range (range | None): Range of number of colors m to benchmark. If None, use fixed_m.
        solver (str): Solver backend to use ("dualiza" | "ganak" | "bc_enum").
        out_path (str): Path to output JSON file.
        dataset (dict): Existing dataset to append results to.
        run_timeout_s (float): Timeout per (n,m) run in seconds.
    Returns:
        None
    """
    def is_power_of_two(v):
        """ 
        Check if v is a power of two.
        With bitwise and operations.
        Args:
            v (int): Value to check.
        Returns:
            bool: True if v is a power of two, False otherwise."""
        return v & (v - 1) == 0
    
    # Validate inputs
    if n_range is None and fixed_n is None:
        raise ValueError("Provide either fixed_n or n_range.")
    if m_range is None and fixed_m is None:
        raise ValueError("Provide either fixed_m or m_range.")

    # Turn fixed values into single-element ranges to unify logic
    n_values = [n_value for n_value in n_range if is_power_of_two(n_value)] if n_range is not None else range(fixed_n, fixed_n + 1)
    m_values = [m_value for m_value in m_range if is_power_of_two(m_value)] if m_range is not None else range(fixed_m, fixed_m + 1)

    # Iterate grid
    for n_val in n_values:
        for m_val in m_values:
            run_key = f"{n_val}x{m_val}"

            if run_key not in dataset.get("runs", {}):
                dataset["runs"][run_key] = {}
            dataset["runs"][run_key][solver], timeout = benchmark_all_codes(
                n_val,
                m_val,
                solver,
                timeout_s=run_timeout_s
            )
            _write_json(out_path, dataset)
            if timeout:
                print(f"\nTimeout reached for (n,m)=({n_val},{m_val}), stopping benchmarks.")
                return

    print("Completed benchmarks.")

   

if __name__ == "__main__":
    # Configuration
    out_path = "benchmark_all.json"
    solvers = ["dualiza", "ganak", "bc_enum"] # choose from dualiza | ganak | bc_enum
    run_timeout = 10.0  # timeout time per turn
    # Load or initialize dataset
    dataset = _load_dataset(out_path)
    dataset.setdefault("runs", {})
    # Benchmark grid settings
    # pegs = [4, 8, 10, 4]
    # colors = [range(1,17), range(1,17), range(1,17), 6]
    pegs = [4]
    colors = [2]
    # Run benchmarks
    for solver in solvers:
        print(f"\nStarting benchmarks for solver: {solver}\n")
        for n_val, m_val in zip(pegs, colors):
            print(f"\nStarting benchmarks for (n,m)=({n_val},{m_val})\n")
            benchmark_grid(
                fixed_n=n_val,
                m_range=m_val if isinstance(m_val, range) else None,
                fixed_m=m_val if isinstance(m_val, int) else None,
                solver=solver,
                out_path=out_path,
                dataset=dataset,
                run_timeout_s=run_timeout,
            )
    print("\nAll benchmarks completed.")
