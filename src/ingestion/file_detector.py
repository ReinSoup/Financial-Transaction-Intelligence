#FILE TYPE VALIDATION GATE

#when project recieves files, we need to validate the file type before processing it. This is to ensure that we are only processing files that are in the correct format and can be ingested by our system.

from pathlib import Path

from src.utils.config import SUPPORTED_EXTENSIONS


def detect_file_type(file_path: str):

    ext = Path(file_path).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:

        raise ValueError(
            f"Unsupported file type: {ext}"
        )

    return ext


