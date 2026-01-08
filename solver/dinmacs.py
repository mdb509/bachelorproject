from typing import Sequence
import os
from game.ruleset import DEFAULT_RULES
from pathlib import Path


class Dinmacs:
    """
    A simple encoder/decoder for the DINMACS format.

    Attributes:
        name_to_id (dict[str, int]): Mapping from variable names to integer IDs.
        id_to_name (dict[int, str]): Mapping from integer IDs to variable names.
    """

    def __init__(self, rules= None):
        """
        Initialize the Dinmacs encoder/decoder.
        """
        self.name_to_id: dict[str, int] = {}
        self.id_to_name: dict[int, str] = {}
        self.rules = rules or DEFAULT_RULES
        self.code_length = self.rules["code_length"]
        self.num_colors = self.rules["num_colors"]

    def _get_id(self, name: str) -> int:
        """
        Get the integer ID for a given variable name, assigning a new ID if necessary.
        Args:
            name (str): The variable name.
        Returns:
            int: The integer ID for the variable.
        """
        vid = self.name_to_id.get(name)
        if vid is None:
            vid = len(self.name_to_id) + 1
            self.name_to_id[name] = vid
            self.id_to_name[vid] = name
        return vid

    def encode_clauses(
        self, clauses: Sequence[Sequence[str]]
    ) -> list[list[int]]:
        """
        Encode clauses from variable names to integer IDs.
        Args:
            clauses (Sequence[Sequence[str]]): The clauses to encode.
        Returns:
            list[list[int]]: The encoded clauses.
        """
        encoded: list[list[int]] = []
        for clause in clauses:
            enc_clause: list[int] = []
            for lit in clause:
                lit = lit.strip()
                neg = lit.startswith("-")
                name = lit[1:].strip() if neg else lit
                vid = self._get_id(name)
                enc_clause.append(-vid if neg else vid)
            encoded.append(enc_clause)
        return encoded

    def decode_literal(self, lit: int) -> str:
        """
        Decode a literal from integer ID to variable name.
        Args:
            lit (int): The literal to decode.
        Returns:
            str: The decoded literal.
        """
        name = self.id_to_name[abs(lit)]
        return f"-{name}" if lit < 0 else name

    def decode_clause(self, clause: Sequence[int]) -> tuple[str, ...]:
        """
        Decode a clause from integer IDs to variable names.
        Args:
            clause (Sequence[int]): The clause to decode.
        Returns:
            tuple[str, ...]: The decoded clause.
        """
        return tuple(self.decode_literal(lit) for lit in clause)

    def decode_clauses(
        self, clauses: Sequence[Sequence[int]]
    ) -> list[tuple[str, ...]]:
        """
        Decode multiple clauses from integer IDs to variable names.
        Args:
            clauses (Sequence[Sequence[int]]): The clauses to decode.
        Returns:
            list[tuple[str, ...]]: The decoded clauses.
        """
        return [self.decode_clause(c) for c in clauses]

    def build_dinmacs(self, path: str, clauses: Sequence[Sequence[int]]):
        """
        Write the DINMACS file to disk.
        Args:
            path (str): The file path to write the DINMACS file to.
            guess_index (int): The index of the guess for which the clauses are built.
            clauses (Sequence[Sequence[int]]): The clauses to write.
        """
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"c t pmc\n")
            f.write(f"p cnf {len(self.name_to_id)} {len(clauses)}\n")
            f.writelines(
                (" ".join(map(str, claus)) + " 0\n" for claus in clauses)
            )
            # f.write(
            #     "c p show "
            #     + " ".join(
            #         str(i) for i in range(1, self.num_colors * self.code_length + 1)
            #     )
            #     + " 0\n"
            # )
