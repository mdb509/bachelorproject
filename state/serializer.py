# Convert game states to readable formats (for logging or export)
import json


def to_json(data_dict: dict) -> str:
    """
    Convert a dictionary to a JSON string.
    Args:
        data_dict (dict): The dictionary to convert.
    Returns:
        str: The JSON string representation of the dictionary.
    """
    return json.dumps(data_dict, indent=2)


def from_json(json_string: str) -> dict:
    """
    Convert a JSON string back to a dictionary.
    Args:
        json_string (str): The JSON string to convert.
    Returns:
        dict: The resulting dictionary.
    """
    return json.loads(json_string)
