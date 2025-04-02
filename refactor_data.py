import json
import re

def extract_grade(text):
    """Extracts the grade level from a text string.

    Args:
        text: The text string to extract the grade from.

    Returns:
        The grade level as an integer, or None if no grade level is found.
    """
    grade_pattern = r"(\d+)(?:th|st|nd|rd)|(College)|(Kindergarten)"
    match = re.search(grade_pattern, text, re.IGNORECASE)
    if match:
        if match.group(3):
            return 0
        elif match.group(2):
            return 13
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None

def add_grade_level(filepath="dataset.json"):
    """Adds a 'grade_level' key to each entry in the dataset.

    Args:
        filepath: The path to the JSON dataset file.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for entry in data:
        text = entry["prompt"]
        grade_level = extract_grade(text)
        entry['grade_level'] = grade_level

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    add_grade_level()