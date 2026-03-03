# save frames, load templates, paths

# Alle Datei-/Pfad-Operationen: YAML-Konfig laden, Templates laden, Debug-Frames speichern, Assets auflisten.

"""
io.py

This module contains filesystem and I/O helpers for the project:
- loading YAML configuration files
- resolving project-relative asset paths
- ensuring directories exist
- saving debug images
- loading template images from disk

Inputs:
- Paths (pathlib.Path) to config files, assets, or output folders.
- NumPy arrays representing images (OpenCV format).

Outputs:
- Python dictionaries (from YAML config)
- Template image collections (e.g., dict[label] -> list[np.ndarray])
- Debug images saved to disk
"""


"""
Zu implementierende Funktionen

project_root() -> Path

load_yaml(path: Path) -> dict

resolve_asset_path(*parts) -> Path

ensure_dir(path: Path) -> None

save_debug_image(path: Path, image: np.ndarray) -> None

list_images(folder: Path, exts=(".png",".jpg",".jpeg")) -> list[Path]

load_templates(folder: Path) -> dict[str, list[np.ndarray]] (label -> templates)

pathlib:
https://docs.python.org/3/library/pathlib.html

PyYAML (safe_load):
https://pyyaml.org/wiki/PyYAMLDocumentation

NumPy array basics (for images):
https://numpy.org/doc/stable/user/quickstart.html




"""



