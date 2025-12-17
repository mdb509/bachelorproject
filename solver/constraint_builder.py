from __future__ import annotations

from itertools import combinations
from state.game_state import GameState
from game.guess import Guess


class Cnf:
    """
    Mastermind CNF encoder (correct + fast).

    Variables:
      - Code variables: <Color>_<Pos>  (e.g. R_1)  -- these should be the first 24 vars.
      - Global helper vars per color:
          GE_<Color>_<t>  means: "the secret code contains at least t occurrences of <Color>"
        plus small auxiliary AND vars for defining GE (only depends on code vars, not on guesses).

    Per guess constraints:
      - exactly black of the 4 "position-match" literals are true:
          code_var(guess[i], i)
      - exactly (black+white) of the 4 "overlap threshold" literals are true:
          for each color c repeated gcount times in the guess, include GE_c_1 .. GE_c_gcount

    Attributes:
        game_state: The current game state.
        guesses: List of Guess objects representing the history of guesses.
        code_length: Length of the secret code.
        num_colors: Number of possible colors.
        colors: List of color identifiers.
        code_vars: Tuple of variable names representing the code positions.
        _base_clauses: List of base clauses enforcing code variable constraints.
        _global_clauses: List of global clauses defining GE variables.
    """

    def __init__(self, game_state: GameState):
        self.game_state = game_state
        self.guesses = game_state.guesses

        self.code_length = self.game_state.rules["code_length"]
        self.num_colors = self.game_state.rules["num_colors"]
        self.colors = self.game_state.rules["colors"]

        # Code variables, should be first encountered in DIMACS
        self.code_vars: tuple[str, ...] = tuple(
            f"{c}_{i}"
            for c in self.colors
            for i in range(1, self.code_length + 1)
        )
        # Base constraints (ONLY code vars): exactly one color per position
        self._base_clauses: list[tuple[str, ...]] = []
        self._base_clauses.extend(self._only_one_color())

        # Global constraints that may introduce helper vars (GE_* and aux AND vars).
        # these are appended after base clauses so code vars get IDs 1..(colors*len)
        self._global_clauses: list[tuple[str, ...]] = []
        self._global_clauses.extend(self._define_all_ge_thresholds())

    @staticmethod
    def _neg(lit: str) -> str:
        """Negate a literal string: x -> -x, -x -> x."""
        return lit[1:] if lit.startswith("-") else "-" + lit

    def _code_var(self, color: str, pos_1based: int) -> str:
        """<color>_<pos> : code position pos has color."""
        return f"{color}_{pos_1based}"

    def _ge_var(self, color: str, t: int) -> str:
        """GE_<color>_<t> : code contains at least t occurrences of color."""
        return f"GE_{color}_{t}"

    def _and_aux(self, color: str, t: int, combo_idx: int) -> str:
        """Aux var for 'AND of a particular combination' used in GE definitions."""
        return f"A_{color}_{t}_{combo_idx}"

    def _exactly_k(
        self, lits: tuple[str, ...], k: int
    ) -> list[tuple[str, ...]]:
        """
        Naive exactly-k encoding via combinations.
        Perfect for n=4 (very small and avoids sequential-counter aux vars).

        Args:
            lits: Tuple of literals.
        Returns:
            List of clauses encoding "exactly k of lits are true".
        """
        # Basic cases
        n = len(lits)
        if k < 0 or k > n:
            return [()]
        clauses: list[tuple[str, ...]] = []

        # At most k: for every (k+1)-subset, not all can be true
        for combo in combinations(lits, k + 1):
            clauses.append(tuple(self._neg(v) for v in combo))

        # At least k: for every (n-k+1)-subset, at least one must be true
        for combo in combinations(lits, n - k + 1):
            clauses.append(tuple(combo))

        return clauses

    # Base clauses: code variable constraints
    def _only_one_color(self) -> list[tuple[str, ...]]:
        """
        Each position must have exactly one color.

        Returns:
            List of clauses encoding the constraints.
        """

        clauses: list[tuple[str, ...]] = []
        n = self.code_length
        m = self.num_colors

        # For each position, exactly one of the colors must be assigned
        for pos in range(1, n + 1):
            vars_at_pos = tuple(
                self.code_vars[(pos - 1) + n * color_idx]
                for color_idx in range(m)
            )
            clauses.extend(self._exactly_k(vars_at_pos, 1))
        return clauses

    # Global clauses: define GE_<c>_<t> vars
    def _define_all_ge_thresholds(self) -> list[tuple[str, ...]]:
        """
        Defines GE_<color>_<t> for all colors and t=1..code_length.
        These constraints depend ONLY on code vars, so they are shared across all guesses.

        Returns:
            List of clauses encoding the GE definitions.
        """
        clauses: list[tuple[str, ...]] = []
        n = self.code_length

        for c in self.colors:
            # Create code vars for this color
            x = tuple(self._code_var(c, i) for i in range(1, n + 1))

            #  GE(c,1)  definition
            ge1 = self._ge_var(c, 1)
            # x1 -> GE1, x2 -> GE1, ... xn -> GE1 : (¬xi ∨ GE1)
            for xi in x:
                clauses.append((self._neg(xi), ge1))
            # GE1 -> (x1 ∨ x2 ∨ ... ∨ xn) : (¬GE1 ∨ x1 ∨ x2 ∨ ... ∨ xn)
            clauses.append((self._neg(ge1),) + x)

            # GE(c,t) definitions for t=2..n
            for t in range(2, n + 1):
                # GE(c,t) definition
                ge = self._ge_var(c, t)
                # Create auxiliary AND vars for each t-combination of positions
                aux_vars: list[str] = []
                # For each t-subset of positions, create an aux AND var
                for idx, pos_subset in enumerate(
                    combinations(range(n), t), start=1
                ):
                    
                    a = self._and_aux(c, t, idx)
                    aux_vars.append(a)

                    # a <-> (x_p1 ∧ x_p2 ∧ ... ∧ x_pt) is encoded as:

                    # a -> (x1 ∧ x2 ∧ ... ∧ xt) == (¬a ∨ x1) ∧ (¬a ∨ x2) ... (¬a ∨ xt)
                    for p in pos_subset:
                        clauses.append((self._neg(a), x[p]))

                    # (x1 ∧ x2 ∧ ... ∧ xt) -> a  == (a ∨ ¬x1 ∨ ¬x2 ... ¬xt)
                    clauses.append(
                        (a,) + tuple(self._neg(x[p]) for p in pos_subset)
                    )

                    # aux -> GE : (¬a ∨ GE)
                    clauses.append((self._neg(a), ge))

                # GE -> OR(aux): (¬GE ∨ a1 ∨ a2 ∨ ...)
                clauses.append((self._neg(ge),) + tuple(aux_vars))

                # Monotonic helper: GE(c,t) -> GE(c,t-1)
                clauses.append((self._neg(ge), self._ge_var(c, t - 1)))

        return clauses

    # Per-guess constraints (correct Mastermind semantics)
    def build_constraints(
        self, guess: Guess, base_clauses: bool = False
    ) -> list[tuple[str, ...]]:
        """
        Build constraints for ONE guess+feedback.

        Args:
            guess: The Guess object containing the guess and feedback.
            guess_index: Index of the guess in the history (not used here).
            base_clauses: If True, include base clauses (only needed once).
        Returns:
            List of clauses encoding the constraints for this guess.

        """
        try:
            black, white = guess.get_feedback()
        except TypeError:
            return []

        overlap = black + white
        n = self.code_length

        gseq = guess.get_guess()  # list[str] length n

        clauses: list[tuple[str, ...]] = []
        if base_clauses:
            clauses.extend(self._base_clauses)
            clauses.extend(self._global_clauses)

        # Black pegs: exactly 'black' positions match
        black_lits = tuple(self._code_var(gseq[i], i + 1) for i in range(n))
        clauses.extend(self._exactly_k(black_lits, black))

        # Total overlap (black+white): sum_c min(count(code,c), count(guess,c))
        # For each color c that appears gcount times in the guess,
        # include GE(c,1) ... GE(c,gcount). Total number of these lits is always n.
        gcount = {c: 0 for c in self.colors}
        for col in gseq:
            gcount[col] += 1

        overlap_lits: list[str] = []
        for c in self.colors:
            for t in range(1, gcount[c] + 1):
                overlap_lits.append(self._ge_var(c, t))
        clauses.extend(self._exactly_k(tuple(overlap_lits), overlap))
        return clauses

    def build_constraints_for_history(self) -> list[tuple[str, ...]]:
        """
        Build constraints for the full game history.
        Ordering:
          1) base clauses (code vars only) -> ensures code vars get IDs first
          2) global GE/aux clauses
          3) per-guess clauses
        Returns:
            List of clauses encoding the constraints for the full history.
        """
        clauses: list[tuple[str, ...]] = []
        clauses.extend(self._base_clauses)
        clauses.extend(self._global_clauses)

        for guess in self.guesses:
            clauses.extend(self.build_constraints(guess))
        return clauses
