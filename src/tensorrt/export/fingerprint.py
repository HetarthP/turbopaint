"""Content fingerprinting for ONNX protobufs and external tensor data."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def onnx_bundle_sha256(onnx_path: Path, onnx_module: Any) -> str:
    """Hash the ONNX protobuf and every external-data file it references."""
    model = onnx_module.load(str(onnx_path), load_external_data=False)
    locations: set[str] = set()
    for tensor in model.graph.initializer:
        for entry in tensor.external_data:
            if entry.key == "location":
                locations.add(entry.value)

    digest = hashlib.sha256()
    for path in [onnx_path, *(onnx_path.parent / location for location in sorted(locations))]:
        if not path.is_file():
            raise FileNotFoundError(f"ONNX bundle file not found: {path}")
        relative_name = path.name.encode("utf-8")
        digest.update(len(relative_name).to_bytes(8, "big"))
        digest.update(relative_name)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()

