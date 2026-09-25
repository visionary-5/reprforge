"""Architecture-neutral capture and resume at a traced cut.

Given the modules that the interface tracer identified as the visual branch,
``capture`` records their outputs during an ordinary source forward pass and
``resume`` replaces their forward methods during a target forward pass so
that the retained outputs are returned and the visual computation is skipped.
The page-level check lives at the boundary itself: the stub hashes the tensor
arguments the target hands to the visual branch and refuses to return a state
that was produced from different inputs.

Model loading and seed selection remain integration responsibilities. The
caller must validate model dependencies before requesting replay; this module
checks boundary inputs unless a fixed-input contract is explicitly supplied.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
from collections.abc import Callable
from typing import Any

import torch
from torch.utils._pytree import tree_flatten


def tensor_digest(tensors: list[torch.Tensor]) -> str:
    h = hashlib.sha256()
    for t in tensors:
        t = t.detach()
        h.update(f"{list(t.shape)}:{t.dtype}".encode())
        h.update(t.contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def describe(obj: Any) -> Any:
    """Serializable description of an output structure (tensors reduced to shape
    and dtype).
    """

    if isinstance(obj, torch.Tensor):
        return {
            "__tensor__": True,
            "shape": list(obj.shape),
            "dtype": str(obj.dtype).replace("torch.", ""),
        }
    if isinstance(obj, (list, tuple)):
        kind = (
            "tuple"
            if isinstance(obj, tuple) and type(obj) is tuple
            else "list"
            if isinstance(obj, list)
            else "namedtuple"
        )
        return {
            "__seq__": kind,
            "type": f"{type(obj).__module__}.{type(obj).__qualname__}",
            "items": [describe(x) for x in obj],
        }
    if isinstance(obj, dict):
        return {
            "__map__": f"{type(obj).__module__}.{type(obj).__qualname__}",
            "items": {k: describe(v) for k, v in obj.items()},
        }
    if obj is None or isinstance(obj, (int, float, str, bool)):
        return {"__const__": obj}
    raise TypeError(f"cannot describe output of type {type(obj)}")


def _resolve(path: str):
    module, _, name = path.rpartition(".")
    return getattr(importlib.import_module(module), name)


def materialize(desc: Any, device: str, tensors: list[torch.Tensor] | None):
    """Rebuild an output structure; tensor slots take retained tensors in order, or
    empties when ``tensors`` is None.
    """

    def build(d):
        if "__tensor__" in d:
            value = tensors.pop(0) if tensors is not None else None
            if value is None:
                return torch.empty(
                    d["shape"], dtype=getattr(torch, d["dtype"]), device=device
                )
            return value.to(device)
        if "__seq__" in d:
            items = [build(x) for x in d["items"]]
            if d["__seq__"] == "tuple":
                return tuple(items)
            if d["__seq__"] == "list":
                return items
            return _resolve(d["type"])(*items)
        if "__map__" in d:
            items = {k: build(v) for k, v in d["items"].items()}
            cls = _resolve(d["__map__"])
            return cls(**items) if cls is not dict else items
        return d["__const__"]

    return build(desc)


@dataclasses.dataclass
class CutSpec:
    cut_modules: tuple[
        str, ...
    ]  # outermost visual modules in execution order; all are skipped at resume
    frontier_modules: tuple[
        str, ...
    ]  # subset whose outputs the language side consumes; their outputs are retained
    retained_leaves: dict[str, tuple[int, ...]] = dataclasses.field(
        default_factory=dict
    )  # module -> output leaf positions to retain

    @classmethod
    def from_report(cls, report, execution_order: list[str] | None = None) -> CutSpec:
        cut = list(report.visual_modules)
        if execution_order:
            position = {n: i for i, n in enumerate(execution_order)}
            cut.sort(key=lambda n: position.get(n, len(position)))
        frontier = [m for m in report.cut_modules if m]
        for f in frontier:
            if not any(f == c or f.startswith(c + ".") for c in cut):
                raise ValueError(
                    f"frontier module {f} is not inside the visual branch {cut}"
                )
        leaves: dict[str, set[int]] = {}
        for ft in report.frontier:
            if ft.source_module is not None and ft.source_leaf_index is not None:
                leaves.setdefault(ft.source_module, set()).add(ft.source_leaf_index)
        return cls(
            tuple(cut),
            tuple(frontier),
            {m: tuple(sorted(v)) for m, v in leaves.items()},
        )


def _module(model: torch.nn.Module, name: str) -> torch.nn.Module:
    return model.get_submodule(name)


def capture(
    model: torch.nn.Module, spec: CutSpec, run: Callable[[], Any]
) -> dict[str, Any]:
    """Run the source forward and retain the visual branch outputs together with
    input digests.
    """

    records: dict[str, dict[str, Any]] = {}
    handles = []

    def make_hook(name: str):
        def hook(module, args, kwargs, output):
            leaves, _ = tree_flatten((args, kwargs))
            tensors = [x for x in leaves if isinstance(x, torch.Tensor)]
            rec = {
                "input_digest": tensor_digest(tensors),
                "output_desc": describe(output),
                "calls": records.get(name, {}).get("calls", 0) + 1,
            }
            if name in spec.frontier_modules:
                out_leaves, _ = tree_flatten(output)
                tensor_leaves = [x for x in out_leaves if isinstance(x, torch.Tensor)]
                keep = set(spec.retained_leaves.get(name, range(len(out_leaves))))
                # positions are counted over all leaves (tensors and non-tensors) as in
                # describe()
                tensor_positions = [
                    i for i, x in enumerate(out_leaves) if isinstance(x, torch.Tensor)
                ]
                rec["output"] = [
                    x.detach().cpu().clone() if pos in keep else None
                    for pos, x in zip(tensor_positions, tensor_leaves, strict=False)
                ]
                rec["retained_leaves"] = sorted(keep & set(tensor_positions))
            records[name] = rec

        return hook

    for name in spec.cut_modules:
        handles.append(
            _module(model, name).register_forward_hook(
                make_hook(name), with_kwargs=True
            )
        )
    try:
        result = run()
    finally:
        for h in handles:
            h.remove()
    for name in spec.cut_modules:
        if name not in records:
            raise RuntimeError(f"cut module {name} was not executed")
        if records[name]["calls"] != 1:
            raise RuntimeError(
                f"cut module {name} executed {records[name]['calls']} times; "
                "one call per page is required"
            )
    return {"spec": dataclasses.asdict(spec), "modules": records, "result": result}


class ResumeStub:
    """Replace one visual-branch module.

    Inputs from outside the branch are checked against the retained digest
    (the page-level check). Inputs produced by another stub are placeholders
    and are not checked.
    """

    def __init__(
        self,
        name: str,
        record: dict[str, Any],
        device: str,
        check_inputs: bool,
        produced,
    ):
        self.name = name
        self.record = record
        self.device = device
        self.check_inputs = check_inputs
        self.produced = (
            produced  # shared WeakTensorKeyDictionary of tensors emitted by any stub
        )
        self.calls = 0
        self.checked = False

    def __call__(self, *args, **kwargs):
        self.calls += 1
        leaves, _ = tree_flatten((args, kwargs))
        tensors = [x for x in leaves if isinstance(x, torch.Tensor)]
        external = not any(x in self.produced for x in tensors)
        if self.check_inputs and external:
            digest = tensor_digest(tensors)
            self.checked = True
            if digest != self.record["input_digest"]:
                raise PageLevelMismatch(self.name, self.record["input_digest"], digest)
        output = self.record.get("output")
        result = materialize(
            self.record["output_desc"],
            self.device,
            list(output) if output is not None else None,
        )
        out_leaves, _ = tree_flatten(result)
        for x in out_leaves:
            if isinstance(x, torch.Tensor):
                self.produced[x] = self.name
        return result


class PageLevelMismatch(RuntimeError):
    def __init__(self, module: str, expected: str, observed: str):
        super().__init__(
            f"visual-branch input to {module} differs from the retained state "
            f"({expected[:12]} vs {observed[:12]})"
        )
        self.module = module


def resume(
    model: torch.nn.Module,
    state: dict[str, Any],
    run: Callable[[], Any],
    *,
    device: str = "cuda:0",
    check_inputs: bool = True,
) -> Any:
    """Run the target forward with the visual branch replaced by the retained state."""

    from torch.utils.weak import WeakTensorKeyDictionary

    stubs: dict[str, ResumeStub] = {}
    originals: dict[str, Any] = {}
    produced = WeakTensorKeyDictionary()
    for name, record in state["modules"].items():
        module = _module(model, name)
        originals[name] = module.forward
        stubs[name] = ResumeStub(name, record, device, check_inputs, produced)
        module.forward = stubs[name]
    try:
        result = run()
    finally:
        for name, forward in originals.items():
            _module(model, name).forward = forward
    for name, stub in stubs.items():
        if stub.calls != 1:
            raise RuntimeError(f"stubbed module {name} was called {stub.calls} times")
    if check_inputs and not any(stub.checked for stub in stubs.values()):
        raise RuntimeError(
            "no visual-branch entry received external inputs; "
            "page-level check could not run"
        )
    return result


def dependency_digest(model: torch.nn.Module, names: list[str]) -> dict[str, str]:
    """Digest of the named parameters and buffers as loaded; the model-level check
    compares these.
    """

    lookup = dict(model.named_parameters())
    lookup.update(dict(model.named_buffers()))
    out = {}
    for name in names:
        t = lookup[name].detach()
        h = hashlib.sha256()
        h.update(f"{list(t.shape)}:{t.dtype}".encode())
        h.update(t.contiguous().cpu().view(torch.uint8).numpy().tobytes())
        out[name] = h.hexdigest()
    return out
