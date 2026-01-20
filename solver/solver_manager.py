from __future__ import annotations

import os
import threading
import time
import random
from concurrent.futures import ThreadPoolExecutor,wait, FIRST_COMPLETED
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Literal

from game.guess import Guess
from solver.constraint_builder import Cnf
from solver.dinmacs import Dinmacs
from solver.solver_interface import SatSolverInterface


Backend = Literal["dualiza", "ganak", "bc_enum"]


# drop-in helper for progress and log line
def progress_print(msg: str) -> None:
    # overwrite same line, no newline
    print(f"\r\033[K{msg}", end="", flush=True)


def log_print(msg: str) -> None:
    # first terminate the progress line, then print normally
    print("\r\033[K", end="", flush=True)
    print(msg, flush=True)


@dataclass(frozen=True)
class MinimaxConfig:
    backend: Backend = "dualiza"
    max_workers: int = max(1, (os.cpu_count() or 4))
    # high return value to indicate stop
    stop_sentinel: int = 10**18


class MinimaxSolver:
    """
    Minimax Guess Selection:
    - pro Guess über alle Feedbacks worst-case count bestimmen
    - best_guess = min über worst-case
    - optional: early-stop, wenn best_guess sehr klein und wir ein Model ziehen können

    Attributes:
        cfg: MinimaxConfig
        solver_api: SatSolverInterface
        dinmacs: Dinmacs
        dinmacs_lock: threading.Lock
        tmpdir: str
        colors: list[str]
        code_length: int
        num_colors: int
        decode_variable_map: dict[str, str]
        tool_args: list[str]
        feedbacks: list[tuple[int, int]]
        all_combinations: list[tuple[str, ...]]

        Methods:
        choose_guess(...): Selects the best guess using the minimax strategy.
    """

    def __init__(
        self,
        *,
        config: MinimaxConfig,
        solver_api: SatSolverInterface,
        dinmacs: Dinmacs,
        dinmacs_lock: threading.Lock,
        tmpdir: str,
        colors: list[str],
        code_length: int,
        num_colors: int,
        decode_variable_map: dict[str, str],
        tool_args: list[str],
        feedbacks: list[tuple[int, int]],
    ):
        self.cfg = config
        self.solver_api = solver_api
        self.dinmacs = dinmacs
        self.dinmacs_lock = dinmacs_lock
        self.tmpdir = tmpdir
        self.colors = colors
        self.code_length = code_length
        self.num_colors = num_colors
        self.decode_variable_map = decode_variable_map
        self.tool_args = tool_args
        self.feedbacks = feedbacks
        self.all_combinations = list(product(colors, repeat=code_length))

    def _cnf_path_for_thread(self) -> Path:
        """
        Get a unique CNF file path for the current thread.
        """
        tid = threading.get_ident()
        return Path(self.tmpdir) / f"thread_{tid}.cnf"

    def _count_for_guess_feedback(
        self,
        *,
        encoder: Cnf,
        encoded_base_clauses,
        guess_sequence: tuple[str, ...],
        fb: tuple[int, int],
        stop_event: threading.Event,
    ) -> tuple[int, list[tuple[str, ...]]]:
        """
        Build CNF for (guess, fb) and return (count, solutions).

        Args:
            encoder: Cnf - The CNF encoder.
            encoded_base_clauses: The base clauses already encoded.
            guess_sequence: The guess sequence as a tuple of strings.
            fb: The feedback tuple (correct positions, correct colors).
            stop_event: threading.Event - Event to signal stopping.
        Returns:
            A tuple containing the count of solutions and a list of solution tuples.
        """
        if stop_event.is_set():
            return self.cfg.stop_sentinel, []

        cnf_path = self._cnf_path_for_thread()

        g = Guess(list(guess_sequence))
        g.apply_feedback(feedback=fb)
        claus = encoder.build_constraints(g)
        
        # build dinmacs CNF
        # lock dinmacs usage
        with self.dinmacs_lock:
            encoded_delta = self.dinmacs.encode_clauses(claus)
            solv_clauses = encoded_base_clauses + encoded_delta
            self.dinmacs.build_dinmacs(str(cnf_path), clauses=solv_clauses)
        # run solver
        stdout = self.solver_api.run_tool(
            tool_name=self.cfg.backend,
            input_file=str(cnf_path),
            extra_args=self.tool_args,
        )
        if self.cfg.backend == "dualiza":
            # dualiza: count
            cnt = self.solver_api.parse_dualiza_stdout(
                stdout, decode_variable_map=self.decode_variable_map
            )
        elif self.cfg.backend == "ganak":
            # ganak: count
            cnt = self.solver_api.parse_ganak_stdout(stdout)
        elif self.cfg.backend == "bc_enum":
            # bc enum: count
            cnt = self.solver_api.parse_bc_enum_stdout(stdout)
        return cnt, len(solv_clauses)

    def _evaluate_guess(
        self,
        *,
        encoder: Cnf,
        encoded_base_clauses,
        guess_sequence: tuple[str, ...],
        stop_event: threading.Event,
    ) -> tuple[tuple[str, ...], int, tuple[int, int] | None]:
        """
        Worker: worst-case over all feedbacks for a given guess.
        Args:
            encoder: Cnf - The CNF encoder.
            encoded_base_clauses: The base clauses already encoded.
            guess_sequence: The guess sequence as a tuple of strings.
            stop_event: threading.Event - Event to signal stopping.
        Returns:
            A tuple containing the guess sequence, worst-case count, and worst feedback.
        """

        if stop_event.is_set():
            return guess_sequence, self.cfg.stop_sentinel, None

        max_cnt = 0
        max_fb = None

        # Evaluate all feedbacks for this guess
        for fb in self.feedbacks:
            if stop_event.is_set():
                return guess_sequence, self.cfg.stop_sentinel, None

            # Count solutions for this guess and feedback
            cnt, claus_len = self._count_for_guess_feedback(
                encoder=encoder,
                encoded_base_clauses=encoded_base_clauses,
                guess_sequence=guess_sequence,
                fb=fb,
                stop_event=stop_event,
            )
            if cnt is not None:
                if cnt > max_cnt:
                    max_cnt = cnt
                    max_fb = fb

        return guess_sequence, max_cnt, max_fb, claus_len


    def choose_guess(
        self,
        *,
        encoder: Cnf,
        encoded_base_clauses,
        progress: bool = True,
        timeout_s: float | None = None,
    ) -> tuple[
        tuple[str, ...], int, tuple[int, int] | None, list[str] | None
    ]:
        """
        Choose the best guess using the minimax strategy.
        Args:
            encoder: Cnf - The CNF encoder.
            encoded_base_clauses: The base clauses already encoded.
            progress: bool - Whether to show progress output.
        Returns:
          best_guess, best_worst_case, best_worst_fb, optional_model_guess
        """
        # # check cpu usage
        # print("cpu_count:", os.cpu_count())
        # print("affinity :", len(os.sched_getaffinity(0)), sorted(os.sched_getaffinity(0)))

        # p = Path("/sys/fs/cgroup/cpu.max")
        # if p.exists():
        #     print("cpu.max  :", p.read_text().strip())

        # p = Path("/sys/fs/cgroup/cpuset.cpus.effective")
        # if p.exists():
        #     print("cpuset.effective:", p.read_text().strip())

        # setup for parallel evaluation
        stop_event = threading.Event()
        best_guess = None
        best_minmax_cnt = float("inf")
        best_minmax_fb = None

        # timing and progress
        start = time.perf_counter()
        deadline = (start + timeout_s) if timeout_s is not None else None
        
        # progress tracking
        last_report = start
        done = 0
        total = len(self.all_combinations)

        # iterator over all combinations
        it = iter(self.all_combinations)
        in_flight = set()


        def submit_one():
            """
            Submit one guess evaluation to the thread pool.
            """
            gs = next(it)
            return pool.submit(
                self._evaluate_guess,
                encoder=encoder,
                encoded_base_clauses=encoded_base_clauses,
                guess_sequence=gs,
                stop_event=stop_event,
            )

        # use ThreadPoolExecutor for parallel evaluation
        with ThreadPoolExecutor(max_workers=self.cfg.max_workers) as pool:
            # initially fill the in-flight set
            try:
                for _ in range(self.cfg.max_workers):
                    in_flight.add(submit_one())
            except StopIteration:
                pass
            claus_len = 0
            # main loop: wait for completions and submit new tasks
            while in_flight:
                # check for timeout
                now = time.perf_counter()
                if deadline is not None and now >= deadline:
                    stop_event.set()
                    break

                # wait for at least one to complete
                wait_timeout = 1.0
                if deadline is not None:
                    wait_timeout = max(0.0, min(wait_timeout, deadline - now))
                
                # wait for any future to complete
                done_set, not_done_set = wait(
                    in_flight, timeout=wait_timeout, return_when=FIRST_COMPLETED
                )

                if not done_set:
                    now2 = time.perf_counter()
                    # periodic progress report
                    if progress and now2 - last_report >= 1.0:
                        rate = done / max(1e-9, (now2 - start))
                        progress_print(
                            f"Progress: {done}/{total} guesses ({rate:.1f} guesses/sec)"
                        )
                        last_report = now2
                    continue

                # collect results as they complete
                for fut in done_set:
                    # remove from in-flight
                    in_flight.remove(fut)
                    # get result
                    gs, minmax_cnt, minmax_fb, claus_len = fut.result()
                    done += 1

                    # New best guess found
                    if minmax_cnt > 0 and minmax_fb is not None:
                        key = (minmax_cnt, -minmax_fb[0], -minmax_fb[1])

                        # with first guess, we always take it
                        if best_guess is None:
                            best_guess = gs
                            best_minmax_cnt = minmax_cnt
                            best_minmax_fb = minmax_fb

                        # else, compare with current best
                        else:
                            best_key = (
                                best_minmax_cnt,
                                -best_minmax_fb[0],
                                -best_minmax_fb[1],
                            )
                            if key < best_key:
                                best_guess = gs
                                best_minmax_cnt = minmax_cnt
                                best_minmax_fb = minmax_fb

                                # log_print(
                                #     f"new best guess : {best_guess}\n"
                                #     f"with fb        : {best_minmax_fb}\n"
                                #     f"new best key   : {key}\n"
                                # )

                    # Early stopping condition
                    if not stop_event.is_set():
                        try:
                            in_flight.add(submit_one())
                        except StopIteration:
                            pass

                # periodic progress report
                now3 = time.perf_counter()
                if progress and now3 - last_report >= 1.0:
                    rate = done / max(1e-9, (now3 - start))
                    progress_print(
                        f"Progress: {done}/{total} guesses ({rate:.1f} guesses/sec)"
                    )
                    last_report = now3
            if stop_event.is_set():
                for fut in in_flight:
                    fut.cancel()
        return best_guess, claus_len 
