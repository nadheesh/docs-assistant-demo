import os
from builtins import FileNotFoundError

FILE_PATH = "secret"
OPENAI_API_KEY = "OPENAI_API_KEY"


def set_keys():
    # Check if the environment variable exists
    if OPENAI_API_KEY in os.environ:
        return
    else:
        # Set the environment variable from a file
        try:
            with open(FILE_PATH, 'r') as file:
                value = file.read().strip()
                os.environ[OPENAI_API_KEY] = value
        except FileNotFoundError:
            print(f"File not found: {FILE_PATH}")
        except Exception as e:
            print(f"Error setting environment variable")


def load_files(path):
    files = []
    for filename in os.listdir(path):
        child_path = os.path.join(path, filename)
        if os.path.isdir(child_path):
            files += load_files(child_path)
        elif filename.endswith(".md"):
            files.append((filename, open(os.path.join(path, filename))))
    return files