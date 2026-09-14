"""Read the retrieval projection from a safetensors shard prefix."""

import json
import struct
from pathlib import Path
from typing import Any


def load_projection(path: Path, torch: Any) -> dict[str, Any]:
    payload = path.read_bytes()
    header_length = struct.unpack("<Q", payload[:8])[0]
    data_start = 8 + header_length
    header = json.loads(payload[8:data_start])
    dtype_map = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
    tensors = {}
    for key in ("custom_text_proj.weight", "custom_text_proj.bias"):
        spec = header[key]
        start, end = spec["data_offsets"]
        raw = bytearray(payload[data_start + start : data_start + end])
        tensors[key] = torch.frombuffer(raw, dtype=dtype_map[spec["dtype"]]).reshape(
            spec["shape"]
        )
    return tensors
