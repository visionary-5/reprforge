"""Serializable codecs for reusable post-vision index-compilation states."""

from __future__ import annotations

from typing import Any


CODECS = ("bf16", "int8_token", "int4_group64", "pca512_bf16", "pca256_bf16")


def fit_pca(values: Any, rank: int, niter: int, torch: Any) -> dict[str, Any]:
    """Fit one query-free shared channel basis to sampled visual states."""
    mean = values.float().mean(dim=0)
    centered = values.float() - mean
    _, _, basis = torch.pca_lowrank(centered, q=rank, center=False, niter=niter)
    return {"mean": mean.to(torch.bfloat16).cpu(), "basis": basis.to(torch.bfloat16).cpu()}


def encode(values: Any, codec: str, torch: Any, pca: dict[str, Any] | None = None) -> dict[str, Any]:
    """Encode a [batch, visual_tokens, channels] BF16 tensor on its current device."""
    if codec == "bf16":
        return {"values": values.to(torch.bfloat16).cpu()}
    if codec == "int8_token":
        scale = values.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
        quantized = torch.round(values.float() / scale).clamp(-127, 127).to(torch.int8)
        return {"quantized": quantized.cpu(), "scale": scale.to(torch.bfloat16).cpu()}
    if codec == "int4_group64":
        if values.shape[-1] % 64:
            raise ValueError("int4_group64 requires channel dimension divisible by 64")
        grouped = values.float().reshape(*values.shape[:-1], values.shape[-1] // 64, 64)
        scale = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 7.0
        quantized = torch.round(grouped / scale).clamp(-7, 7).to(torch.int16) + 8
        packed = (quantized[..., 0::2] | (quantized[..., 1::2] << 4)).to(torch.uint8)
        return {
            "packed": packed.cpu(),
            "scale": scale.squeeze(-1).to(torch.bfloat16).cpu(),
            "channels": values.shape[-1],
        }
    if codec.startswith("pca"):
        if pca is None:
            raise ValueError("PCA codec requires a fitted basis")
        rank = int(codec.removeprefix("pca").split("_", 1)[0])
        mean = pca["mean"].to(values.device).float()
        basis = pca["basis"][:, :rank].to(values.device).float()
        coefficients = (values.float() - mean) @ basis
        return {"coefficients": coefficients.to(torch.bfloat16).cpu()}
    raise ValueError(codec)


def decode(
    payload: dict[str, Any],
    codec: str,
    device: str,
    torch: Any,
    pca: dict[str, Any] | None = None,
) -> Any:
    """Decode a serialized payload into BF16 visual states on ``device``."""
    if codec == "bf16":
        return payload["values"].to(device)
    if codec == "int8_token":
        return (
            payload["quantized"].to(device).float()
            * payload["scale"].to(device).float()
        ).to(torch.bfloat16)
    if codec == "int4_group64":
        packed = payload["packed"].to(device)
        low = (packed & 15).to(torch.int16) - 8
        high = (packed >> 4).to(torch.int16) - 8
        quantized = torch.stack((low, high), dim=-1).flatten(-2).float()
        scale = payload["scale"].to(device).float().unsqueeze(-1)
        return (quantized * scale).reshape(
            *quantized.shape[:-2], int(payload["channels"])
        ).to(torch.bfloat16)
    if codec.startswith("pca"):
        if pca is None:
            raise ValueError("PCA codec requires a fitted basis")
        rank = int(codec.removeprefix("pca").split("_", 1)[0])
        coefficients = payload["coefficients"].to(device).float()
        basis = pca["basis"][:, :rank].to(device).float()
        mean = pca["mean"].to(device).float()
        return (coefficients @ basis.T + mean).to(torch.bfloat16)
    raise ValueError(codec)
