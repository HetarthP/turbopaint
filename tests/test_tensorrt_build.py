from pathlib import Path

import pytest

from src.tensorrt.build.engine import (
    BuildRequest,
    parser_errors,
    precision_for,
    validate_network_contract,
)
from src.tensorrt.build.unet import main as unet_main
from src.tensorrt.build.vae_decoder import main as vae_main


class Tensor:
    def __init__(self, name, shape, dtype):
        self.name = name
        self.shape = shape
        self.dtype = dtype


class Network:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.num_inputs = len(inputs)
        self.num_outputs = len(outputs)

    def get_input(self, index):
        return self.inputs[index]

    def get_output(self, index):
        return self.outputs[index]


def valid_unet_network() -> Network:
    return Network(
        [
            Tensor("sample", (1, 4, 64, 64), "float16"),
            Tensor("timestep", (1,), "float32"),
            Tensor("encoder_hidden_states", (1, 77, 2048), "float16"),
            Tensor("text_embeds", (1, 1280), "float16"),
            Tensor("time_ids", (1, 6), "float16"),
        ],
        [Tensor("noise_pred", (1, 4, 64, 64), "float16")],
    )


def test_precision_policy() -> None:
    assert precision_for("unet") == "fp16"
    assert precision_for("vae_decoder") == "fp32"
    with pytest.raises(ValueError):
        precision_for("unknown")


def test_build_request_defaults_and_validation() -> None:
    request = BuildRequest("unet", Path("input.onnx"), Path("output.plan"))
    assert request.workspace_gib == 4.0
    assert request.force is False
    with pytest.raises(ValueError, match="workspace_gib"):
        BuildRequest("unet", Path("x"), Path("y"), workspace_gib=0)


def test_static_network_contract_accepts_expected_unet() -> None:
    validate_network_contract(valid_unet_network(), "unet")


def test_static_network_contract_reports_shape_and_dtype() -> None:
    network = valid_unet_network()
    network.inputs[0].shape = (2, 4, 64, 64)
    network.inputs[1].dtype = "float16"
    with pytest.raises(RuntimeError) as error:
        validate_network_contract(network, "unet")
    assert "sample shape" in str(error.value)
    assert "timestep dtype" in str(error.value)


def test_parser_errors_preserve_every_message() -> None:
    class Parser:
        num_errors = 2

        @staticmethod
        def get_error(index):
            return f"error-{index}"

    assert parser_errors(Parser()) == ["error-0", "error-1"]


def test_component_clis_expose_help_without_tensorrt() -> None:
    with pytest.raises(SystemExit) as unet_exit:
        unet_main(["--help"])
    with pytest.raises(SystemExit) as vae_exit:
        vae_main(["--help"])
    assert unet_exit.value.code == vae_exit.value.code == 0

