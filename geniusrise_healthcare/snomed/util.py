import logging
import zipfile
from typing import Tuple

log = logging.getLogger(__name__)


def unzip_snomed_ct(zip_path: str, extract_path: str) -> None:
    log.info(f"Unzipping SNOMED-CT dataset from {zip_path} to {extract_path}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)


def extract_and_remove_semantic_tag(fsn: str) -> Tuple[str, str]:
    semantic_tag = fsn.split("(")[-1].rstrip(")").strip()
    fsn_without_tag = fsn.rsplit("(", 1)[0].strip()
    semantic_tag = "" if semantic_tag == fsn_without_tag else semantic_tag
    return semantic_tag, fsn_without_tag
