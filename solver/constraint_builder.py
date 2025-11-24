class cnf:
    """
    A helper class for Mastermind that encodes the game state as CNF formulas.

    Given a sequence of guesses and their feedback, this class generates DIMACS
    CNF clauses that capture all valid secret codes consistent with that
    history. The resulting CNFs are intended to be used as input to a SAT
    solver and form the basis for solver-driven Mastermind strategies
    (e.g., to choose optimal next guesses or to enumerate all remaining
    solutions).
    """

    def __init__(
        self,
    ):
        pass

    def combinations(self, vars: tuple[str], k: int) -> list[str]:
        """
        Generate all k-element combinations (unordered subsets) of the given
        variables.

        This is recursive combinatorics helper used for encoding
        cardinality constraints. Each combination is returned as a tuple
        of variable names.

        Args:
            variables: Tuple of propositional variables (as strings).
            k: Size of each combination (0 ≤ k ≤ len(variables)).

        Returns:
            A list of all k-element combinations, each as a tuple of variables.
            For example, combinations(("x1", "x2", "x3"), 2) yields:
            [("x1", "x2"), ("x1", "x3"), ("x2", "x3")].
        """

        # base cases
        if k == 0:
            return [()]
        if k > len(vars):
            return []
        if len(vars) == 0:
            return []

        head = vars[0]
        tail = vars[1:]

        # with head, remains k-1 from tail
        with_head = [
            (head,) + comb for comb in self.combinations(tail, k - 1)
        ]

        # without head, remains k aus tail
        without_head = self.combinations(tail, k)

        return with_head + without_head

    def k_subsets(self, vars: tuple[str], k: int) -> list[tuple[str]]:
        """
        Generate CNF clauses enforcing that **exactly** k of the given
        variables are true.

        This uses the naive cardinality encoding based on
        combinations:

        - "At least k" is encoded by adding, for every (n - k + 1)-subset S of
        `variables`, a clause that requires at least one variable in S to
        be true.
        - "At most k" is encoded by adding, for every (k + 1)-subset S
        of `variables`,a clause that forbids all variables in S from
        being true at the same time.

        Args:
            variables: Tuple of propositional variables (as strings).
            k: Number of variables that must be true.

        Returns:
            A list of CNF clauses, each clause represented as a tuple of
            literals (e.g. ("x1", "-x3", "x7")). The conjunction of
            all returned clauses is satisfied iff exactly k variables
            from `variables` are true.
        """

        clauses = []
        n = len(vars)

        # At least k variables are true
        for combo in self.combinations(vars, n - k + 1):
            clauses.append(tuple(combo))

        # At most k variables are true
        for combo in self.combinations(vars, k + 1):
            clauses.append(tuple(["-" + var for var in combo]))

        return clauses

    def sequential_cardinality_atmost(
        self, vars: tuple[str], k: int, prefix: str
    ):
        """
        Encodes the cardinality constraint sum(vars) <= k using a sequential
        counter.

        The encoding follows Sinz (2005) and introduces auxiliary variables
        of the form f"{prefix}{i}{j}" to represent partial sums.

        Args:
            vars: Tuple of Boolean variable names (as strings) to which the
                at-most constraint is applied.
            k: Maximum number of variables in vars that may be true.
            prefix: String prefix used for naming auxiliary counter variables
                (e.g. "s" → "s11", "s12", ...).

        Returns:
            list[tuple[str, ...]]: CNF clauses enforcing sum(vars) <= k.
            Each clause is a tuple of literals as strings, where a negated
            literal is prefixed with "-" (e.g. "-x1").
        """

        n = len(vars)

        # Base cases
        if k == 0:
            return [()]
        if k > n:
            return []
        if len(vars) == 0:
            return []

        # Sequentilal counter translatet into cnf clauses
        clauses = [("-" + vars[0], f"{prefix}11")]
        for i in range(2, k + 1):
            clauses.append((f"-{prefix}1" + str(i),))

        for i in range(2, n):
            clauses.append(("-" + vars[i - 1], f"{prefix}" + str(i) + "1"))
            clauses.append(
                (f"-{prefix}" + str(i - 1) + "1", f"{prefix}" + str(i) + "1")
            )
            for j in range(2, k + 1):
                clauses.append(
                    (
                        "-" + vars[i - 1],
                        f"-{prefix}" + str(i - 1) + str(j - 1),
                        f"{prefix}" + str(i) + str(j),
                    )
                )
                clauses.append(
                    (
                        f"-{prefix}" + str(i - 1) + str(j),
                        f"{prefix}" + str(i) + str(j),
                    )
                )
            clauses.append(
                ("-" + vars[i - 1], f"-{prefix}" + str(i - 1) + str(k))
            )
        clauses.append(
            ("-" + vars[n - 1], f"-{prefix}" + str(n - 1) + str(k))
        )

        return clauses

    def resolve_double_negation(self, clauses: list[tuple[str]]):
        """
        Simplifies CNF clauses by removing double negations in literals.

        Args:
            clauses (list[tuple[str]): List of clauses, each a tuple of
            literal strings (e.g. "x1", "-x1", "--x1").

        Returns:
            list[tuple[str]: A new list of clauses with all "--"
            substrings removed from literals (e.g. "--x1" → "x1").
        """

        result = []

        for claus in clauses:
            result.append(tuple([var.replace("--", "") for var in claus]))

        return result

    def variables_negation(self, vars: tuple[str]):
        """
        Negates all variables in a tuple of strings.

        Args:
            vars: Tuple of variable names as strings, e.g. ("X1", "X2", "X3").

        Returns:
            tuple[str, ...]: Tuple of negated variable names, e.g.
                ("X1", "X2", "X3") -> ("-X1", "-X2", "-X3").
        """
        return tuple(["-" + var for var in vars])


if __name__ == "__main__":
    c = cnf()
    # variables = ("A", "B", "C", "D", "E", "F")
    # clauses = c.k_subsets(variables, 3)
    # subsets = c.combinations(variables, 3)
    # print(clauses)
    # print(subsets)

    vars = (
        "m11",
        "m12",
        "m13",
        "m14",
        "m21",
        "m22",
        "m23",
        "m24",
        "m31",
        "m32",
        "m33",
        "m34",
        "m41",
        "m42",
        "m43",
        "m44",
    )

    neg_vars = c.variables_negation(vars)

    atmost_clauses = c.sequential_cardinality_atmost(vars, 3, "atmost")
    print(atmost_clauses)

    atleast_clauses = c.sequential_cardinality_atmost(
        neg_vars, 16 - 3, "atleast"
    )
    print(atleast_clauses)

    resolved_at_least_clauses = c.resolve_double_negation(atleast_clauses)
    print(resolved_at_least_clauses)

    exakt_k_clauses = c.k_subsets(vars, 3)
    print(exakt_k_clauses)
    print("length at most clauses: ", len(atmost_clauses))
    print("length at least clauses: ", len(resolved_at_least_clauses))
    print("length exackt clauses with naiv: ", len(exakt_k_clauses))
