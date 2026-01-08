import argparse
import json
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _natural_turn_sort_key(s: str):
    m = re.search(r"(\d+)", str(s))
    return int(m.group(1)) if m else s


def _parse_run_key(key: str):
    m = re.fullmatch(r"(\d+)\s*x\s*(\d+)", key.strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))

def _annotate_points(ax, xs, ys, *, fmt="{:.2f}", dx=0, dy=6, fontsize=8):
        """
        Annotate points (x, y) on ax with formatted y values.

        Args:
            ax: matplotlib Axes
            xs: list of x coordinates
            ys: list of y coordinates
            fmt: format string for y values
            dx: x offset in points
            dy: y offset in points
            fontsize: font size for annotations
        """
        
        for x, y in zip(xs, ys):
            if y is None:
                continue
            ax.annotate(
                fmt.format(y),
                (x, y),
                textcoords="offset points",
                xytext=(dx, dy),
                ha="center",
                va="center",
                fontsize=fontsize,
            )

def compute_run_stats(games: dict):
    """
    Returns:
      avg_total_time (float, np.nan if no won games)
      avg_turn_times (list[float]) average time per turn index (turn 1 at index 0), won games only
      avg_turns_per_game (float, np.nan if no won games), won games only
      n_won (int)
    """
    won = np.array(games.get("won", []), dtype=bool)
    total_time = np.array(games.get("total_time_s", []), dtype=np.float32)

    # Guard against length mismatches
    n = min(len(won), len(total_time))
    won = won[:n]
    total_time = total_time[:n]

    # avg, min, max total time (won games only)
    won_total_times = total_time[won]
    avg_total_time = float(np.mean(won_total_times)) if won_total_times.size > 0 else np.nan
    min_total_time = float(np.min(won_total_times)) if won_total_times.size > 0 else np.nan
    max_total_time = float(np.max(won_total_times)) if won_total_times.size > 0 else np.nan

    n_won = int(won_total_times.size)


    # avg, min, max turn time per turn index (won games only)
    turn_headers = list(games.get("turn_headers", []))
    turn_headers.sort(key=_natural_turn_sort_key)
    turn_cols = games.get("turn_time_s_columns", {}) or {}
    avg_turn_times = []
    min_turn_times = []
    max_turn_times = []
    for th in turn_headers:
        col = turn_cols.get(th, [])
        nn = min(len(col), len(won))
        if nn == 0:
            avg_turn_times.append(np.nan)
            continue

        vals = []
        for t, w in zip(col[:nn], won[:nn]):
            if not w or t is None:
                continue
            vals.append(float(t))
        avg_turn_times.append(float(np.mean(vals)) if vals else np.nan)
        min_turn_times.append(float(np.min(vals)) if vals else np.nan)
        max_turn_times.append(float(np.max(vals)) if vals else np.nan)

    # avg, min ,max clausle length per turn index (won games only)
    clausle_cols = games.get("clausle_len_table", {}) or {}
    avg_clausle_lengths = []
    min_clausle_lengths = []
    max_clausle_lengths = []
    for th in turn_headers:
        col = clausle_cols.get(th, [])
        nn = min(len(col), len(won))
        if nn == 0:
            avg_clausle_lengths.append(np.nan)
            continue

        vals = []
        for t, w in zip(col[:nn], won[:nn]):
            if not w or t is None:
                continue
            vals.append(float(t))
        avg_clausle_lengths.append(float(np.mean(vals)) if vals else np.nan)
        min_clausle_lengths.append(float(np.min(vals)) if vals else np.nan)
        max_clausle_lengths.append(float(np.max(vals)) if vals else np.nan)

    # avg, min ,max turns per game (won games only)
    # turns(game i) = count of non-None entries across all turn columns for that game
    if n == 0 or not turn_headers:
        avg_turns_per_game = np.nan
        min_turns_per_game = np.nan
        max_turns_per_game = np.nan
    else:
        counts = []
        for i in range(n):
            if not won[i]:
                continue
            c = 0
            for th in turn_headers:
                col = turn_cols.get(th, [])
                if i < len(col) and col[i] is not None:
                    c += 1
            counts.append(c)
        avg_turns_per_game = float(np.mean(counts)) if counts else np.nan
        min_turns_per_game = float(np.min(counts)) if counts else np.nan
        max_turns_per_game = float(np.max(counts)) if counts else np.nan


    return (
        avg_total_time, 
        min_total_time, 
        max_total_time, 
        avg_turn_times, 
        min_turn_times, 
        max_turn_times, 
        avg_clausle_lengths,
        min_clausle_lengths,
        max_clausle_lengths,
        avg_turns_per_game, 
        min_turns_per_game, 
        max_turns_per_game, 
        n_won
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="benchmark_all.json", help="Path to benchmark JSON")
    ap.add_argument("--pegs", nargs="*", type=int, default=None,
                    help="Which peg counts to plot (e.g. --pegs 4 8). Default: all found.")
    ap.add_argument("--outdir", default="./results", help="Output directory for PNGs")
    args = ap.parse_args()

    path = Path(args.file)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with path.open("r", encoding="utf-8") as f:
        runs = json.load(f).get("runs", {})

    # Solvers
    solvers = ["dualiza", "ganak", "bc_enum"]

    # Discover available (pegs, colors)
    available = {}
    for k in runs.keys():
        parsed = _parse_run_key(k)
        if parsed is None:
            continue
        p, m = parsed
        available.setdefault(p, set()).add(m)

    if not available:
        raise ValueError("No runs found with keys like '4x8' in data['runs'].")

    peg_list = sorted(available.keys()) if args.pegs is None else args.pegs

    for p in peg_list:
        if p not in available:
            print(f"[skip] No runs for {p} pegs.")
            continue

        colors_list = sorted(available[p])
        if not colors_list:
            print(f"[skip] No colors left for {p} pegs after filtering.")
            continue

        avg_total_times = {str:[]}
        min_total_times = {str:[]}
        max_total_times = {str:[]}
        avg_turns_per_game_list = {str:[]}
        min_turns_per_game_list = {str:[]}
        max_turns_per_game_list = {str:[]}
        wins_by_m = {str:{}}
        avg_turn_time = {}
        min_turn_time = {}
        max_turn_time = {}
        avg_clausle_lengths = {}
        min_clausle_lengths = {}
        max_clausle_lengths = {}

        for solver in solvers:
            avg_total_times[solver] = []
            min_total_times[solver] = []
            max_total_times[solver] = []
            avg_turns_per_game_list[solver] = []
            min_turns_per_game_list[solver] = []
            max_turns_per_game_list[solver] = []
            avg_turn_time[solver] = {}
            min_turn_time[solver] = {}
            max_turn_time[solver] = {}
            avg_clausle_lengths[solver] = {}
            min_clausle_lengths[solver] = {}
            max_clausle_lengths[solver] = {}
            wins_by_m[solver] = {}
            for m in colors_list:
                key = f"{p}x{m}"
                games = runs[key].get(solver, {}).get("games", {})

                (avg_total, 
                min_total, 
                max_total, 
                avg_turn_times, 
                min_turn_times,
                max_turn_times, 
                avg_clausle_length,
                min_clausle_length,
                max_clausle_length,
                avg_turns_per_game, 
                min_turns_per_game, 
                max_turns_per_game, 
                n_won
                ) = compute_run_stats(games)


                # Collect for plotting
                avg_total_times[solver].append(avg_total)
                min_total_times[solver].append(min_total)
                max_total_times[solver].append(max_total)
                avg_turns_per_game_list[solver].append(avg_turns_per_game)
                min_turns_per_game_list[solver].append(min_turns_per_game)
                max_turns_per_game_list[solver].append(max_turns_per_game)
                avg_turn_time[solver][m] = avg_turn_times
                min_turn_time[solver][m] = min_turn_times
                max_turn_time[solver][m] = max_turn_times
                avg_clausle_lengths[solver][m] = avg_clausle_length
                min_clausle_lengths[solver][m] = min_clausle_length
                max_clausle_lengths[solver][m] = max_clausle_length
                wins_by_m[solver][m] = n_won

        # Plot configuration:
        plt.rcParams["lines.solid_capstyle"] = "round"
        plt.rcParams["lines.solid_joinstyle"] = "round"
        plt.rcParams["lines.linewidth"] = 1.0

      
        for solver in solvers:
            # Plot 1: Average total time vs colors
            plt.figure(figsize=(10, 6))
            # Average total time line with min/max scatter and band
            plt.plot(colors_list, avg_total_times[solver], marker="o", markersize=3, label=f"Average Total Time")
            plt.scatter(colors_list, max_total_times[solver], marker="^", s=20, label=f"Max Total Time")
            plt.scatter(colors_list, min_total_times[solver], marker="v", s=20, label=f"Min Total Time")
            plt.fill_between(colors_list, min_total_times[solver], max_total_times[solver], alpha=0.2, label=f"Min–Max range")
            # Annotations for points
            _annotate_points(plt.gca(), colors_list, avg_total_times[solver], fmt="{:.2f}s", dy=8)
            _annotate_points(plt.gca(), colors_list, min_total_times[solver], fmt="{:.2f}s", dy=-8)
            _annotate_points(plt.gca(), colors_list, max_total_times[solver], fmt="{:.2f}s", dy=16)
                
            # Titles and labels
            plt.title(f"Average Total Time for {p} pegs ({solver})\n Games won per color(m): " + ", ".join([f"m={m}: {wins_by_m[solver][m]}" for m in colors_list]))
            plt.xlabel("Number of Colors (m)")
            plt.ylabel("Average Total Time (s) [won games]")
            plt.xticks(colors_list)
            plt.grid(True)
            plt.legend()
            # Save plot
            out1 = outdir / f"{p}pegs_avg_total_time_{solver}.png"
            plt.savefig(out1, dpi=200, bbox_inches="tight")


            # Plot 2: Average turns per game vs colors
            plt.figure(figsize=(10, 6))
            # Average turns per game line with min/max scatter and band
            plt.plot(colors_list, avg_turns_per_game_list[solver], marker="o", label=f"Average Turns per Game ({solver})")
            plt.scatter(colors_list, max_turns_per_game_list[solver], marker="^", s=20, label=f"Max Turns per Game ({solver})")
            plt.scatter(colors_list, min_turns_per_game_list[solver], marker="v", s=20, label=f"Min Turns per Game ({solver})")
            plt.fill_between(colors_list, min_turns_per_game_list[solver], max_turns_per_game_list[solver], alpha=0.2, label=f"Min–Max range ({solver})")
            # Annotations for points
            _annotate_points(plt.gca(), colors_list, avg_turns_per_game_list[solver], fmt="{:.2f}", dy=8)
            _annotate_points(plt.gca(), colors_list, min_turns_per_game_list[solver], fmt="{:.2f}", dy=-8)
            _annotate_points(plt.gca(), colors_list, max_turns_per_game_list[solver], fmt="{:.2f}", dy=16)
            # Titles and labels
            plt.title(f"Average Turns per Game for {p} pegs ({solver})\n Games won per color(m): " + ", ".join([f"m={m}: {wins_by_m[solver][m]}" for m in colors_list]))
            plt.xlabel("Number of Colors (m)")
            plt.ylabel("Average Turns per Game [won games]")
            plt.xticks(colors_list)
            plt.grid(True)
            plt.legend()
            # Save plot
            out_turns = outdir / f"{p}pegs_avg_turns_per_game_{solver}.png"
            plt.savefig(out_turns, dpi=200, bbox_inches="tight")
            plt.close()


            # Plot 3: Average turn time vs turn number (all colors)
            # Determine max number of turns across all m
            max_turns = max((len(v) for v in avg_turn_time[solver].values()), default=0)
            if max_turns == 0:
                print(f"[info] No turn-time data to plot for {p} pegs.")
                continue
            # Create plot
            plt.figure(figsize=(12, 8))
            x_ticks = np.arange(1, max_turns + 1)
            # Plot lines for each m
            for m in colors_list:
                y_avg = avg_turn_time.get(solver, {}).get(m, [])
                y_min = min_turn_time.get(solver, {}).get(m, [])
                y_max = max_turn_time.get(solver, {}).get(m, [])
                if not y_avg:
                    continue
                x = np.arange(1, len(y_avg) + 1)
                # Plot average line, min/max scatter, and band  
                plt.plot(x, y_avg, marker="o", label=f"m={m} (wins={wins_by_m[solver].get(m, 0)})")
                plt.scatter(x, y_min, marker="v", s=20)
                plt.scatter(x, y_max, marker="^", s=20)
                plt.fill_between(x, y_min, y_max, alpha=0.2)
                # Annotations for average points
                _annotate_points(plt.gca(), x, y_avg, fmt="{:.2f}s", dy=8)   
                _annotate_points(plt.gca(), x, y_min, fmt="{:.2f}s", dy=-8)
                _annotate_points(plt.gca(), x, y_max, fmt="{:.2f}s", dy=16)
            # Titles and labels
            plt.title(f"Average Turn Time for ({p} pegs ({solver})")
            plt.xlabel("Turn Number")
            plt.ylabel("Average Turn Time (s) [won games]")
            plt.xticks(x_ticks)
            plt.legend(title="Number of Colors (m)")
            plt.grid(True)
            out2 = outdir / f"{p}pegs_avg_turn_time_{solver}.png"
            plt.savefig(out2, dpi=200, bbox_inches="tight")
            plt.close()


            # Plot 4: clause length
            plt.figure(figsize=(12, 8))
            x_ticks = np.arange(1, max_turns + 1)
            # Plot lines for each m
            for m in colors_list:
                y_avg = avg_clausle_lengths.get(solver, {}).get(m, [])
                y_min = min_clausle_lengths.get(solver, {}).get(m, [])
                y_max = max_clausle_lengths.get(solver, {}).get(m, [])
                if not y_avg:
                    continue
                x = np.arange(1, len(y_avg) + 1)
                # Plot average line, min/max scatter, and band  
                plt.plot(x, y_avg, marker="o", label=f"m={m} (wins={wins_by_m[solver].get(m, 0)})")
                plt.scatter(x, y_min, marker="v", s=20)
                plt.scatter(x, y_max, marker="^", s=20)
                plt.fill_between(x, y_min, y_max, alpha=0.2)
                # Annotations for average points
                _annotate_points(plt.gca(), x, y_avg, fmt="{:.2f}", dy=8)   
                _annotate_points(plt.gca(), x, y_min, fmt="{:.2f}", dy=-8)
                _annotate_points(plt.gca(), x, y_max, fmt="{:.2f}", dy=16)
            # Titles and labels
            plt.title(f"Average Clause Length for {p} pegs ({solver})")
            plt.xlabel("Turn Number")
            plt.ylabel("Average Clause Length [won games]")
            plt.xticks(x_ticks)
            plt.legend(title="Number of Colors (m)")
            plt.grid(True)
            out3 = outdir / f"{p}pegs_avg_clause_length_{solver}.png"
            plt.savefig(out3, dpi=200, bbox_inches="tight")
            plt.close()


if __name__ == "__main__":
    main()
