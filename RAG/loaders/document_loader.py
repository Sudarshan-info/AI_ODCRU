from config.settings import RAW_DATA_PATH


def load_documents(path=RAW_DATA_PATH):
    """
    Load a text document from disk.

    Args:
        path (str): Path to the document file. Defaults to RAW_DATA_PATH from config.

    Returns:
        List[str]: List containing the full text of the document.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        # Wrap in list so downstream chunking works consistently
        return [text]

    except FileNotFoundError:
        print(f"Error: File not found at {path}")
        return []
    except Exception as e:
        print(f"Error loading document: {e}")
        return []
