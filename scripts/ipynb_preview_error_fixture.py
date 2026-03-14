"""
    How To Use:
    1. Save this script as ipynb_preview_error_fixture.py
    2. Run the script from the command line:
        python ./scripts/ipynb_preview_error_fixture.py '..\notebooks\hydrogen_leakage_detection.ipynb'
"""

import json
import sys

def clean_notebook(nb_path, output_path=None):
    """
    Removes metadata.widgets from a Jupyter notebook to fix GitHub rendering issues.
    
    Args:
        nb_path (str): Path to the input .ipynb notebook
        output_path (str, optional): Path to save the cleaned notebook. 
                                     If None, overwrites the original notebook.
    """
    output_path = output_path or nb_path

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    if "metadata" in nb and "widgets" in nb["metadata"]:
        print("Removing 'metadata.widgets'...")
        nb["metadata"].pop("widgets")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print(f"Notebook saved to {output_path}. GitHub should render it now.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python clean_notebook.py path_to_notebook.ipynb [output_path.ipynb]")
    else:
        clean_notebook(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)