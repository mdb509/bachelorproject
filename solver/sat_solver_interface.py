import subprocess
from pathlib import Path


class SatSolverInterface:
    """
    Interface for building and running the Dualizer SAT solver tool.
    Provides methods to build the tool, run it on CNF files, and parse
    filenames.
    """

    def __init__(self):
        # Current file path
        self.HERE = Path(__file__).resolve().parent
        # Dualizer directory and binary paths
        self.DUALIZER_DIR = (self.HERE / "../../dualiza").resolve()
        self.DUALIZER_BIN = (self.DUALIZER_DIR / "dualiza").resolve()
        # Input file path for testing
        self.INPUT_FILE = (self.HERE / "../cnfs/").resolve()

    def _run(self, cmd, *, cwd: Path, capture: bool):
        """
        Calls a subprocess command.
        If capture is True, captures stdout and stderr and returns them in the
        CompletedProcess.
        If capture is False, lets the subprocess print directly to the
        terminal.

        Args:
            cmd: List of command arguments.
            cwd: Working directory for the subprocess.
            capture: Whether to capture stdout and stderr.
        Returns:
            CompletedProcess instance with returncode, stdout, and stderr.
        """
        return subprocess.run(
            cmd,
            cwd=str(cwd),
            text=True,
            stdout=(
                subprocess.PIPE if capture else None
            ),  # None => live ins Terminal
            stderr=subprocess.PIPE if capture else None,
        )

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
