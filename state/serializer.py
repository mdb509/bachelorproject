# Convert game states to readable formats (for logging or export)
import json


def to_json(data_dict: dict) -> str:
    return json.dumps(data_dict, indent=2)


def from_json(json_string: str) -> dict:
    return json.loads(json_string)
