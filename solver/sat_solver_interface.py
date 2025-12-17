import subprocess
from pathlib import Path


class SatSolverInterface:
    def __init__(self):
        # base paths
        self.HERE = Path(__file__).resolve().parent
        # Dualizer: assume already built in dualiza/ folder
        self.DUALIZER_DIR = (self.HERE / "../../dualiza").resolve()
        self.DUALIZER_BIN = (self.DUALIZER_DIR / "dualiza").resolve()

        # Ganak: generate nix output link in ganak/ folder
        self.GANAK_DIR = (self.HERE / "../../ganak").resolve()
        self.GANAK_OUT = (self.GANAK_DIR / "ganak-result").resolve()
        self.GANAK_BIN = (self.GANAK_OUT / "bin" / "ganak").resolve()

    def _run(self, cmd, *, cwd: Path, capture: bool):
        return subprocess.run(
            cmd,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
        )

    def build_ganak(self):
        """
        Build ganak via nix once and create a local output link ganak-result/.
        """
        # if already built, skip
        if self.GANAK_BIN.exists():
            return

        self.GANAK_DIR.mkdir(parents=True, exist_ok=True)

        cp = self._run(
            [
                "nix",
                "build",
                "github:meelgroup/ganak#ganak",
                "-o",
                str(self.GANAK_OUT),
            ],
            cwd=self.GANAK_DIR,
            capture=True,
        )
        if cp.returncode != 0:
            raise RuntimeError(
                f"Building ganak failed\n{cp.stderr}\n{cp.stdout}"
            )

        if not self.GANAK_BIN.exists():
            raise RuntimeError(
                f"Ganak build ok, but Binary not found: {self.GANAK_BIN}"
            )

    def run_ganak(
        self,
        input_file: str | Path,
        extra_args=None,
        *,
        save_to: str | None = None,
    ) -> str:
        """
        Runs the Ganak SAT solver on the given input file with optional
        extra arguments.
        Can save output to a file or return it as a string.

        Args:
            input_file: Path to the input CNF file.
            extra_args: List of extra command-line arguments for Ganak.
            save_to: Optional path to save the output. If None, returns
            output as string.
        Returns:
            Output from Ganak as a string if save_to is None.
        Raises:
            FileNotFoundError: If the Ganak binary is not found.
            RuntimeError: If Ganak execution fails.
        """
        self.build_ganak()

        if extra_args is None:
            extra_args = ["--mode", "0"]

        input_path = Path(input_file).expanduser()
        if not input_path.is_absolute():
            input_path = (Path.cwd() / input_path).resolve()

        cmd = [str(self.GANAK_BIN), *extra_args, str(input_path)]
        cp = self._run(cmd, cwd=self.GANAK_DIR, capture=True)

        if cp.returncode != 0:
            raise RuntimeError(
                f"ganak error (Exit {cp.returncode})\n"
                f"--- stderr ---\n{cp.stderr}\n--- stdout ---\n{cp.stdout}"
            )

        if save_to is not None:
            Path(save_to).write_text(cp.stdout, encoding="utf-8")

        return cp.stdout

    def build_dualiza(self):
        """
        Builds the Dualizer tool by running its configure script and make.
        Raises RuntimeError if any step fails.
        """
        cp = self._run(
            ["./configure.sh"], cwd=self.DUALIZER_DIR, capture=True
        )
        if cp.returncode != 0:
            raise RuntimeError(
                f"configure.sh fehlgeschlagen\n{cp.stderr}\n{cp.stdout}"
            )

        cp = self._run(
            ["make"], cwd=self.DUALIZER_DIR, capture=True
        )  # optional: ["make", "-j"]
        if cp.returncode != 0:
            raise RuntimeError(
                f"make fehlgeschlagen\n{cp.stderr}\n{cp.stdout}"
            )

    def run_dualiza(
        self,
        input_file: str | Path,
        extra_args=None,
        *,
        save_to: str | None = None,
    ) -> str:
        """
        Runs the Dualizer tool on the given input file with optional
        extra arguments.
        Can save output to a file or return it as a string.

        Args:
            input_file: Path to the input CNF file.
            extra_args: List of extra command-line arguments for Dualizer.
            save_to: Optional path to save the output. If None, returns
            output as string.
        Returns:
            Output from Dualizer as a string if save_to is None.
        Raises:
            FileNotFoundError: If the Dualizer binary is not found.
            RuntimeError: If Dualizer execution fails.
        """

        if not self.DUALIZER_BIN.exists():
            raise FileNotFoundError(f"Binary not found: {self.DUALIZER_BIN}")

        if extra_args is None:
            extra_args = [
                "-c",
                "-r",
                "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24",
            ]

        input_path = Path(input_file).expanduser()
        # If relative, resolve relative to *current working dir* (or choose your own rule)
        if not input_path.is_absolute():
            input_path = (Path.cwd() / input_path).resolve()

        cmd = [str(self.DUALIZER_BIN), *extra_args, str(input_path)]
        # capture=True => get stdout/stderr in CompletedProcess
        cp = self._run(cmd, cwd=self.DUALIZER_DIR, capture=True)

        if cp.returncode != 0:
            raise RuntimeError(
                f"dualiza error (Exit {cp.returncode})\n"
                f"--- stderr ---\n{cp.stderr}\n--- stdout ---\n{cp.stdout}"
            )

        if save_to is not None:
            Path(save_to).write_text(cp.stdout, encoding="utf-8")

        return cp.stdout

    def parse_guess_to_filename(
        self, guess: str, feedback: tuple[int, int]
    ) -> str:
        """
        Converts a guess string and feedback tuple into a
        standardized filename.

        Args:
            guess: The guess string (e.g., "RGBY").
            feedback: A tuple of (black_pegs, white_pegs).
        Returns:
            A filename string in the format "GUESS_(B, W).cnf".
        """
        b, w = feedback
        return f"{guess}_({b}, {w}).cnf"

    def parse_dualiza_stdout(
        self, stdout: str, decode_variable_map: dict[str, str]
    ) -> tuple[list[tuple[str, ...]], int]:
        """
        Parse Dualiza output.

        Args:
            stdout: The standard output from Dualiza.

        Returns:
            (solutions, count)
        """
        stdout = stdout.strip()
        if not stdout:
            return [], 0

        lines = stdout.splitlines()
        head = lines[0].strip()

        if head.startswith("s UNSATISFIABLE"):
            return [], 0

        if head.startswith("ALL"):
            # output format: models
            solutions: list[tuple[str, ...]] = []
            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue
                decoded = tuple(
                    decode_variable_map[var]
                    for var in line.split()
                    if var and not var.startswith("!")
                )
                solutions.append(decoded)
            return solutions, len(solutions)

        if head.startswith("NUMBER"):
            # output format: counting
            if len(lines) >= 2:
                try:
                    return [], int(lines[1].strip())
                except ValueError:
                    return [], 0
            return [], 0

    def parse_ganak_stdout(
        self, stdout: str
    ) -> tuple[list[tuple[str, ...]], int]:
        """
        Parse Ganak output.

        Args:
            stdout: The standard output from Ganak.

        Returns:
            (solutions, count)
        """
        stdout = stdout.strip()
        if not stdout:
            return 0

        lines = stdout.splitlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("s SATISFIABLE"):
                continue
            if line.startswith("s UNSATISFIABLE"):
                return 0
            if line.startswith("c s exact"):
                return int(line.split()[-1])
