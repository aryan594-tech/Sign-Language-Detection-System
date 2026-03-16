import os
import sys
import base64
import yaml
from pathlib import Path


def read_yaml_file(file_path: str) -> dict:
    """Read a YAML file and return its contents as a dictionary."""
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise Exception(f"Error reading YAML file: {e}") from e


def write_yaml_file(file_path: str, content: dict) -> None:
    """Write a dictionary to a YAML file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as yaml_file:
            yaml.dump(content, yaml_file, default_flow_style=False)
    except Exception as e:
        raise Exception(f"Error writing YAML file: {e}") from e


def decodeImage(imgstring: str, fileName: str) -> None:
    """Decode a base64-encoded image string and save it to a file."""
    imgdata = base64.b64decode(imgstring)
    with open(fileName, "wb") as f:
        f.write(imgdata)


def encodeImageIntoBase64(croppedImagePath: str) -> str:
    """Encode an image file into a base64 string."""
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_yolov5_root() -> Path:
    """Return the absolute path to the bundled yolov5 directory."""
    return Path(__file__).resolve().parents[2] / "yolov5"
