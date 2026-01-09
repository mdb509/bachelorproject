import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Optional, Any

@dataclass(frozen=True)
class ToolSpec:
    name: str
    work_dir: Path                  # where to run build commands from
    bin_path: Path                  # executable path
    build_args: Optional[list[str]] = None  # build commands
    run_args: Optional[list[str]] = None    # default run arguments

class SatSolverInterface:
    def __init__(self):
        # base paths
        self.HERE = Path(__file__).resolve().parent
        # Dualiza: Path for direction and binary
        self.DUALIZER_DIR = (self.HERE / "../../dualiza").resolve()
        self.DUALIZER_BIN = (self.DUALIZER_DIR / "dualiza").resolve()
        # Ganak: Path for direction and binary
        self.GANAK_DIR = (self.HERE / "../../ganak-linux-amd64").resolve()
        self.GANAK_OUT = (self.GANAK_DIR ).absolute()
        self.GANAK_BIN = (self.GANAK_OUT / "ganak").absolute()
        # BCEnum: Path for direction and binary
        self.BC_ENUM_DIR = (self.HERE / "../../master_project/blocked_clauses_enumeration").resolve()
        self.BC_ENUM_BIN = (self.BC_ENUM_DIR / "src" / "bcp_enum").resolve()
        # Cadical: Path for direction and binary
        self.CADICAL_DIR = (self.HERE / "../../master_project/cadical").resolve()

        self.tools = {
            "dualiza": ToolSpec(
                name="dualiza",
                work_dir=self.DUALIZER_DIR,
                bin_path=self.DUALIZER_BIN,
                build_args=["./configure.sh", "make"],
                run_args=[
                    "-c",
                    "-r",
                    "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24",
                ],
            ),
            "ganak": ToolSpec(
                name="ganak",
                work_dir=self.GANAK_DIR,
                bin_path=self.GANAK_BIN,
                build_args=[[
                        "nix",
                        "build",
                        "github:meelgroup/ganak#ganak",
                        "-o",
                        str(self.GANAK_OUT)]
                    ],
                run_args=["--mode", "0"],
            ),
            "bc_enum": ToolSpec(
                name="bc_enum",
                work_dir=self.BC_ENUM_DIR,
                bin_path=self.BC_ENUM_BIN,
                build_args=["make"],
                run_args=["./bcp_enum", "-c"],
            ),
            "cadical": ToolSpec(
                name="cadical",
                work_dir=self.CADICAL_DIR,
                bin_path=None,  # No single binary for cadical
                build_args=["./configure", "make"],
                run_args=None,
            ),
        }

    def _run(self, cmd, *, cwd: Path, capture: bool):
        return subprocess.run(
            cmd,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
        )

    def build_tool(self, tool_name: str):
        """
        Builds the specified tool by running its build args.
        Raises RuntimeError if any step fails.
        """
        if tool_name == "ganak":
            return
        if tool_name == "bc_enum":
            self.build_tool("cadical")
        
        print(f"Building tool: {tool_name}")
        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")
        tool = self.tools[tool_name]
        if not tool.build_args:
            print(f"No build steps defined for {tool_name}")
            return
        
        for step_args in tool.build_args:
            cp = self._run(step_args, cwd=tool.work_dir, capture=True)
            if cp.returncode != 0:
                raise RuntimeError(
                    f"Build step failed for {tool_name}\n"
                    f"Command: {' '.join(step_args)}\n"
                    f"--- stderr ---\n{cp.stderr}\n--- stdout ---\n{cp.stdout}"
                )

    def run_tool(
        self,
        tool_name: str,
        input_file: str | Path,
        extra_args=None,
        *,
        save_to: str | None = None,
    ) -> str:
        """
        Runs the specified tool on the given input file with optional
        extra arguments.
        Can save output to a file or return it as a string.

        Args:
            tool_name: Name of the tool to run.
            input_file: Path to the input CNF file.
            extra_args: List of extra command-line arguments for the tool.
            save_to: Optional path to save the output. If None, returns
            output as string.
        Returns:
            Output from the tool as a string if save_to is None.
        Raises:
            FileNotFoundError: If the tool binary is not found.
            RuntimeError: If tool execution fails.
        """
        # Get tool spec
        tool = self.tools.get(tool_name)
        # Validate tool
        if tool is None:
            raise ValueError(f"Unknown tool: {tool_name}")
        # Ensure tool is built
        if tool.bin_path is None or not tool.bin_path.exists():
            raise FileNotFoundError(f"Binary not found: {tool.bin_path}")

        input_path = Path(input_file).expanduser()
        # If relative, resolve relative to *current working dir* (or choose your own rule)
        if not input_path.is_absolute():
            input_path = (Path.cwd() / input_path).resolve()

        # Build command
        cmd = [str(tool.bin_path), *(extra_args if extra_args is not None else tool.run_args), str(input_path)]
        # capture=True => get stdout/stderr in CompletedProcess
        cp = self._run(cmd, cwd=tool.work_dir, capture=True)

        if cp.returncode != 0:
            raise RuntimeError(
                f"{tool_name} error (Exit {cp.returncode})\n"
                f"--- stderr ---\n{cp.stderr}\n--- stdout ---\n{cp.stdout}"
            )

        if save_to is not None:
            Path(save_to).write_text(cp.stdout, encoding="utf-8")

        return cp.stdout

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
            return 0

        lines = stdout.splitlines()
        head = lines[0].strip()

        if head.startswith("s UNSATISFIABLE"):
            return 0

        if head.startswith("NUMBER"):
            # output format: counting
            if len(lines) >= 2:
                try:
                    return int(lines[1].strip())
                except ValueError:
                    return 0
            return 0

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
            if line.startswith("c s exact"):
                return int(line.split()[-1])
        return 0
    
    def parse_bc_enum_stdout(
        self, stdout: str) -> tuple[list[tuple[str, ...]], int]:
        """
        Parse Dualiza output.

        Args:
            stdout: The standard output from Dualiza.

        Returns:
            (solutions, count)
        """
        stdout = stdout.strip()
        if not stdout:
            return 0

        lines = stdout.splitlines()
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith("NUMBER"):
                return int(lines[i+1].strip())

        return 0, []
        