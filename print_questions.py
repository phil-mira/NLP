import json

def print_questions_and_answers(filename="dataset.json", grade_levels=[0, 1, 2, 3]):
    """
    Prints the chosen questions and answers from a JSON dataset to a text file,
    filtering by specified grade levels.

    Args:
        filename (str): The name of the JSON file to read from.
        grade_levels (list): A list of integers representing the grade levels to include.
    """

    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: The file '{filename}' contains invalid JSON.")
        return
    
    count = 1
    output_filename = "questions_and_answers.txt"
    with open(output_filename, 'w') as outfile:
        for item in data:
            if 'grade_level' in item and item['grade_level'] in grade_levels:
                prompt = item.get('prompt', 'No prompt found')
                chosen = item.get('chosen', 'No chosen answer found')

                outfile.write(f"Question Number: {count}\n")
                outfile.write(f"Grade Level: {item['grade_level']}\n")
                outfile.write(f"Question: {prompt}\n")
                outfile.write(f"Answer: {chosen}\n")
                outfile.write("Score:\n")
                outfile.write("\n")

                count += 1

    print(f"Questions and answers for grade levels {grade_levels} have been written to '{output_filename}'.")

if __name__ == "__main__":
    print_questions_and_answers()