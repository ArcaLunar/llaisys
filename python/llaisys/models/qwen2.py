import json
from typing import Sequence
from ..libllaisys import LIB_LLAISYS, LlaisysQwen2Meta
from ..libllaisys import DeviceType, DataType

from pathlib import Path
import safetensors


DEFAULT_MODEL_PATH = "./data"


class Qwen2:

    def __init__(
        self,
        model_path=DEFAULT_MODEL_PATH,
        device: DeviceType = DeviceType.CPU,
    ):
        self.device = device
        self._backend = None

        model_path = Path(model_path)
        self.__load_config(model_path / "config.json")

        for file in sorted(model_path.glob("*.safetensors")):
            data_ = safetensors.safe_open(file, framework="numpy", device="cpu")
            for name_ in data_.keys():
                ## TODO: load the model weights
                pass

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = 128,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):

        # TODO: Implement generate function

        return []

    def __load_config(self, config_path: Path):
        with open(config_path, "r") as f:
            config = json.load(f)

        meta = LlaisysQwen2Meta()
        match config.get("torch_dtype", ""):
            case "bfloat16":
                meta.dtype = DataType.BF16
            case "float16":
                meta.dtype = DataType.F16
            case "float32":
                meta.dtype = DataType.F32
            case _:
                raise ValueError(
                    f"Unsupported data type: {config.get('torch_dtype', '')}"
                )
        meta.nlayer = config.get("num_hidden_layers", 0)
        meta.nh = config.get("num_attention_heads", 0)
        meta.hs = config.get("hidden_size", 0)
        meta.nkvh = config.get("num_key_value_heads", 0)
        meta.dh = config.get("head_dim", int(meta.hs / meta.nh) if meta.nh else 0)
        meta.di = config.get("intermediate_size", 0)
        meta.maxseq = config.get("max_position_embeddings", 0)
        meta.voc = config.get("vocab_size", 0)
        meta.epsilon = config.get("layer_norm_epsilon", 1e-5)
        meta.theta = config.get("rope_theta", 1000000.0)
        meta.end_token = config.get("eos_token_id", 0)

        print(
            meta.dtype,
            meta.nlayer,
            meta.hs,
            meta.nh,
            meta.nkvh,
            meta.dh,
            meta.di,
            meta.maxseq,
            meta.voc,
            meta.epsilon,
            meta.theta,
            meta.end_token,
        )

    def __load_weights(self, weights_folder: Path):
        name_mapping: dict[str, tuple[str, int | None]] = {}
