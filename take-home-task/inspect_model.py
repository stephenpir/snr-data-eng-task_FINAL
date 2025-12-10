import os
import struct
import pickle
from typing import Dict, Any, Optional

# --- Configuration ---
# Build an absolute path to the model file to ensure the script can be run from anywhere.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "artifacts", "model.joblib")
print(f"--- Inspecting model file at: {MODEL_PATH} ---") # This will now print the absolute path

def get_joblib_metadata(filename: str) -> Optional[Dict[str, Any]]:
    """
    Reads the header of a joblib file to extract and return its metadata.

    Args:
        filename: The path to the .joblib file.

    Returns:
        A dictionary containing the file's metadata, or None if it cannot be read.
    """
    print(f"--- Attempting to read metadata from: {filename} ---")
    try:
        with open(filename, 'rb') as f:
            # Joblib files start with a magic number. We check for a few common ones.
            magic_bytes = f.read(4)
            if magic_bytes not in (b'\x80\x02j\x01', b'JL\x01\x01'):
                print(f"Error: This does not appear to be a recognized joblib file.")
                print(f"Magic bytes found: {magic_bytes.hex()}")
                return None

            # The next 4 bytes are the header length (little-endian unsigned int)
            header_len_bytes = f.read(4)
            header_len = struct.unpack('<I', header_len_bytes)[0]

            # Read the header, which is a pickled dictionary
            header_bytes = f.read(header_len)
            
            # Unpickle the header to get the metadata dictionary
            metadata = pickle.loads(header_bytes)
            return metadata

    except FileNotFoundError:
        print(f"Error: Model file not found at '{filename}'.")
        print("Please ensure the path is correct and the file exists.")
        return None
    except (pickle.UnpicklingError, struct.error, EOFError) as e:
        print(f"Error: Could not parse the joblib file. It might be corrupted or in an old format.")
        print(f"       Details: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
         print(f"FATAL: The file '{MODEL_PATH}' does not exist.")
         print("Please make sure you have trained a model and saved it to the 'artifacts' directory.")
    else:
        metadata = get_joblib_metadata(MODEL_PATH)

        if metadata:
            print("\n✅ Successfully extracted model metadata:")
            for key, value in metadata.items():
                print(f"  - {key}: {value}")
            
            if 'python_version' in metadata:
                py_version = metadata['python_version']
                print(f"\nConclusion: This model was created with Python version: {py_version}")
            else:
                print("\nNote: The 'python_version' key was not found in the model's metadata.")
        else:
            print("\n❌ Failed to extract model metadata.")
