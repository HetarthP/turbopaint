from pathlib import Path

import pytest

from src.tensorrt.export.contracts import COMPONENT_CONTRACTS, contract_as_dict
from src.tensorrt.export.unet import build_parser as build_unet_parser
from src.tensorrt.export.vae_decoder import build_parser as build_vae_parser
from src.tensorrt.export.fingerprint import onnx_bundle_sha256


def test_unet_contract_matches_sdxl_512() -> None:
    contract = COMPONENT_CONTRACTS["unet"]
    assert contract["inputs"][0].shape == (1, 4, 64, 64)
    assert contract["inputs"][2].shape == (1, 77, 2048)
    assert contract["outputs"][0].shape == (1, 4, 64, 64)
    assert contract["inputs"][0].dtype == "float16"
    assert contract["inputs"][1].dtype == "float32"


def test_vae_contract_preserves_force_upcast_decode() -> None:
    contract = COMPONENT_CONTRACTS["vae_decoder"]
    assert contract["inputs"][0].shape == (1, 4, 64, 64)
    assert contract["outputs"][0].shape == (1, 3, 512, 512)
    assert contract["inputs"][0].dtype == "float32"


def test_contract_serialization_and_unknown_component() -> None:
    assert contract_as_dict("unet")["inputs"][0]["shape"] == [1, 4, 64, 64]
    with pytest.raises(ValueError, match="Unknown component"):
        contract_as_dict("unknown")


def test_export_cli_defaults() -> None:
    unet = build_unet_parser().parse_args([])
    vae = build_vae_parser().parse_args([])
    assert unet.output == Path("artifacts/onnx/unet/model.onnx")
    assert vae.output == Path("artifacts/onnx/vae_decoder/model.onnx")
    assert unet.seed == vae.seed == 0


def test_onnx_bundle_hash_includes_external_data(tmp_path) -> None:
    model_path = tmp_path / "model.onnx"
    data_path = tmp_path / "weights.data"
    model_path.write_bytes(b"graph")
    data_path.write_bytes(b"weights-a")

    class Entry:
        key = "location"
        value = "weights.data"

    class Tensor:
        external_data = [Entry()]

    class Graph:
        initializer = [Tensor()]

    class Model:
        graph = Graph()

    class Onnx:
        @staticmethod
        def load(path, load_external_data=False):
            return Model()

    first = onnx_bundle_sha256(model_path, Onnx())
    data_path.write_bytes(b"weights-b")
    assert onnx_bundle_sha256(model_path, Onnx()) != first
