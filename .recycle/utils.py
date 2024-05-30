import csv
import logging
from typing import List, Any

log = logging.getLogger(__name__)


def read_rrf_file(file_path: str, delimiter: str = "|") -> List[List[Any]]:
    """
    Reads an RRF (Rich Record Format) file and returns its contents as a list of rows.

    Args:
        file_path (str): Path to the RRF file.
        delimiter (str): Delimiter used in the file (default is '|').

    Returns:
        List of rows, where each row is a list of values.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            reader = csv.reader(file, delimiter=delimiter)
            return list(reader)
    except Exception as e:
        log.error(f"Error reading file {file_path}: {e}")
        raise ValueError(f"Error reading file {file_path}: {e}")
