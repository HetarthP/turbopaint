from pathlib import Path

import pytest

from src.tensorrt.export.contracts import COMPONENT_CONTRACTS, contract_as_dict
from src.tensorrt.export.unet import build_parser as build_unet_parser
from src.tensorrt.export.vae_decoder import build_parser as build_vae_parser


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
