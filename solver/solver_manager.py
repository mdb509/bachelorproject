from __future__ import annotations

import os
import threading
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Literal

from game.guess import Guess
from solver.constraint_builder import Cnf
from solver.dinmacs import Dinmacs
from solver.sat_solver_interface import SatSolverInterface


Backend = Literal["dualiza", "ganak"]


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
    max_workers: int = max(1, (os.cpu_count() - 1 or 4))
    # try tiebreaker models early
    early_model_threshold: int = 2
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
        dualiza_counting_args: list[str]
        dualiza_models_args: list[str]
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
        dualiza_counting_args: list[str],
        dualiza_models_args: list[str],
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

        self.dualiza_counting_args = dualiza_counting_args
        self.dualiza_models_args = dualiza_models_args
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

        # dualiza: count + optional models
        if self.cfg.backend == "dualiza":
            out = self.solver_api.run_dualiza(
                str(cnf_path), extra_args=self.dualiza_counting_args
            )
            sols, cnt = self.solver_api.parse_dualiza_stdout(
                out, decode_variable_map=self.decode_variable_map
            )
            return cnt, sols

        # ganak: count only
        out = self.solver_api.run_ganak(
            str(cnf_path), extra_args=["--mode", "0"]
        )
        cnt = self.solver_api.parse_ganak_stdout(out)
        return cnt, []

    def _evaluate_guess_worker(
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
            cnt, _ = self._count_for_guess_feedback(
                encoder=encoder,
                encoded_base_clauses=encoded_base_clauses,
                guess_sequence=guess_sequence,
                fb=fb,
                stop_event=stop_event,
            )

            if cnt > max_cnt:
                max_cnt = cnt
                max_fb = fb

        return guess_sequence, max_cnt, max_fb

    def _try_get_model_guess(
        self,
        *,
        encoder: Cnf,
        encoded_base_clauses,
        best_guess: tuple[str, ...],
        best_max_fb: tuple[int, int] | None,
        stop_event: threading.Event,
    ) -> list[str] | None:
        """
        Try to get a model guess for the best guess and feedback.
        Args:
            encoder: Cnf - The CNF encoder.
            encoded_base_clauses: The base clauses already encoded.
            best_guess: The best guess sequence as a tuple of strings.
            best_max_fb: The feedback tuple (correct positions, correct colors).
            stop_event: threading.Event - Event to signal stopping.
        Returns:
            A list of strings representing the model guess, or None if not found.
        """

        if self.cfg.backend != "dualiza":
            return None
        if stop_event.is_set():
            return None

        # unique CNF path per thread
        cnf_path = (
            Path(self.tmpdir)
            / f"model_{os.getpid()}_{threading.get_ident()}.cnf"
        )

        # try only the best feedback (if given), else all feedbacks
        fb_list = (
            [best_max_fb]
            if best_max_fb is not None
            else list(self.feedbacks)
        )

        for fb in fb_list:

            g = Guess(list(best_guess))
            g.apply_feedback(feedback=fb)

            claus = encoder.build_constraints(g)

            with self.dinmacs_lock:
                encoded_delta = self.dinmacs.encode_clauses(claus)
                solv_clauses = encoded_base_clauses + encoded_delta
                self.dinmacs.build_dinmacs(
                    str(cnf_path),
                    clauses=solv_clauses,
                )

            stdout = self.solver_api.run_dualiza(
                str(cnf_path), extra_args=self.dualiza_models_args
            )
            model_solutions, _ = self.solver_api.parse_dualiza_stdout(
                stdout, decode_variable_map=self.decode_variable_map
            )

            if model_solutions:
                # randomly pick one model
                guess_lists = [
                    [s[0] for s in model] for model in model_solutions
                ]
                return random.choice(guess_lists)

        return None

    def choose_guess(
        self,
        *,
        encoder: Cnf,
        encoded_base_clauses,
        progress: bool = True,
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
        stop_event = threading.Event()

        best_guess = None
        best_minmax_cnt = float("inf")
        best_minmax_fb = None

        start = time.perf_counter()
        last_report = start
        done = 0
        total = len(self.all_combinations)

        # use ThreadPoolExecutor for parallel evaluation
        pool = ThreadPoolExecutor(max_workers=self.cfg.max_workers)
        futures = []
        try:
            # submit all guess evaluations
            for gs in self.all_combinations:
                futures.append(
                    pool.submit(
                        self._evaluate_guess_worker,
                        encoder=encoder,
                        encoded_base_clauses=encoded_base_clauses,
                        guess_sequence=gs,
                        stop_event=stop_event,
                    )
                )
            # track if we have tried models already
            tried_models = False

            # collect results as they complete
            for fut in as_completed(futures):
                gs, minmax_cnt, minmax_fb = fut.result()
                done += 1

                # New best guess found
                if 0 < minmax_cnt < best_minmax_cnt:
                    best_guess = gs
                    best_minmax_cnt = minmax_cnt
                    best_minmax_fb = minmax_fb

                    log_print(
                        f"Best guess : {best_guess}\n"
                        f"with fb    : {best_minmax_fb}\n"
                        f"min max    : {best_minmax_cnt}\n"
                    )

                now = time.perf_counter()
                # periodic progress report
                if progress and now - last_report >= 2.0:
                    rate = done / max(1e-9, (now - start))
                    progress_print(
                        f"Progress: {done}/{total} guesses ({rate:.1f} guesses/sec)"
                    )
                    last_report = now

                # early model try
                if (
                    best_guess is not None
                    and best_minmax_cnt <= self.cfg.early_model_threshold
                    and self.cfg.backend == "dualiza"
                    and not tried_models
                ):
                    tried_models = True
                    # get model for best guess + feedback
                    model_guess = self._try_get_model_guess(
                        encoder=encoder,
                        encoded_base_clauses=encoded_base_clauses,
                        best_guess=best_guess,
                        best_max_fb=best_minmax_fb,
                        stop_event=stop_event,
                    )
                    # if we got a model guess, we can stop early
                    if model_guess is not None:
                        # signal all workers to stop
                        stop_event.set()
                        for f in futures:
                            f.cancel()
                        return (
                            best_guess,
                            int(best_minmax_cnt),
                            best_minmax_fb,
                            model_guess,
                        )
        finally:
            try:
                # ensure proper shutdown
                pool.shutdown(wait=True, cancel_futures=True)
            except TypeError:
                for f in futures:
                    f.cancel()
                pool.shutdown(wait=True)

        return best_guess, int(best_minmax_cnt), best_minmax_fb, None
