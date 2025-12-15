# state/persistence.py
import json
from pathlib import Path
from .game_state import GameState


def save_state(game_state: GameState, path: str):
    """
    Save the game state to disk as JSON.
    Args:
        game_state (GameState): The game state to save.
        path (str): The file path to save the game state to.
    """
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(game_state.to_dict(reveal_code=True), f, indent=2)


def load_state(path: str) -> GameState:
    """
    Load the game state from disk.
    Args:
        path (str): The file path to load the game state from.
    Returns:
        GameState: The loaded game state."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return GameState.from_dict(data)
