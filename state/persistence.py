# state/persistence.py
import json
from pathlib import Path
from .game_state import GameState


def save_state(game_state: GameState, path: str):
    """Save the game state to disk as JSON."""
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(game_state.to_dict(reveal_code=True), f, indent=2)


def load_state(path: str) -> GameState:
    """Load the game state from disk."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return GameState.from_dict(data)
