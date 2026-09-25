"""Execution-trace discovery of the consumed visual interface.

The interface a retriever's later stages consume from its visual branch is
discovered by running one page through the model under a dispatch-level
taint tracer. Every tensor produced during the forward pass inherits a taint
set from its inputs: ``visual`` for anything derived from the pixel input,
``text`` for anything derived from token identifiers or text embeddings, and
``structural`` for cheap discrete context such as the visual grid or the
attention mask. Parameters and buffers are tracked by name while a tensor is
still purely visual.

A *frontier tensor* is a visual-only tensor that is consumed by an operation
whose result also carries text taint: that is exactly where the language side
reads the visual branch. For this executed path, the frontier tensors describe
the consumed visual interface. The parameters that feed frontier tensors are the
*pre-cut dependency set*; everything else is target-owned computation that a
resume must re-run.

The tracer is architecture-neutral. It requires no knowledge of module names,
only the identity of the seed inputs.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten
from torch.utils.weak import WeakTensorKeyDictionary

VISUAL = "visual"
TEXT = "text"
STRUCTURAL = "structural"
TAG_ORDER = (VISUAL, TEXT, STRUCTURAL)


@dataclasses.dataclass(frozen=True)
class Taint:
    tags: frozenset[str]
    params: frozenset[str]  # tracked only while tags contain no text
    producer: str  # innermost module executing when the tensor was produced
    op: str

    @property
    def visual_only(self) -> bool:
        return VISUAL in self.tags and TEXT not in self.tags

    @property
    def text(self) -> bool:
        return TEXT in self.tags


@dataclasses.dataclass
class FrontierTensor:
    """A visual-only tensor read by the language side."""

    shape: tuple[int, ...]
    dtype: str
    producer_module: str
    producer_op: str
    consumer_module: str
    consumer_op: str
    consumer_output_tags: tuple[str, ...]
    param_dependencies: tuple[str, ...]
    source_module: (
        str | None
    )  # outermost module whose output this tensor derives from by param-free ops
    source_leaf_index: (
        int | None
    )  # position of that output leaf in the module's flattened output
    postprocess_ops: tuple[
        str, ...
    ]  # param-free ops between source_module output and this tensor
    structural_inputs: tuple[
        str, ...
    ]  # shapes of structural-only tensors consumed by the same op

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class InterfaceReport:
    frontier: list[FrontierTensor]
    pre_cut_params: tuple[str, ...]
    post_cut_params: tuple[str, ...]
    structural_params: tuple[
        str, ...
    ]  # params/buffers used only on structural-only paths (target-recomputable context)
    unused_params: tuple[str, ...]
    cut_modules: tuple[
        str, ...
    ]  # outermost modules whose outputs carry the whole frontier
    visual_modules: tuple[
        str, ...
    ]  # outermost modules whose parameters are entirely pre-cut
    op_count: int
    seeds: dict[str, list[list[int]]]
    execution_order: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "frontier": [f.to_dict() for f in self.frontier],
            "pre_cut_params": list(self.pre_cut_params),
            "post_cut_params": list(self.post_cut_params),
            "structural_params": list(self.structural_params),
            "unused_params": list(self.unused_params),
            "cut_modules": list(self.cut_modules),
            "visual_modules": list(self.visual_modules),
            "op_count": self.op_count,
            "seeds": self.seeds,
            "execution_order": list(self.execution_order),
        }

    def contract_digest(self) -> str:
        """Stable digest of the discovered interface (frontier shapes, cut modules,
        dependency names).
        """

        payload = {
            "frontier": [
                (
                    f.source_module,
                    f.consumer_module,
                    f.consumer_op,
                    list(f.shape),
                    f.dtype,
                )
                for f in self.frontier
            ],
            "pre_cut_params": list(self.pre_cut_params),
            "cut_modules": list(self.cut_modules),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class _ModuleStack:
    """Forward hooks that maintain the executing module path and record module
    outputs.
    """

    def __init__(self, model: torch.nn.Module):
        self.names: dict[int, str] = {id(m): n for n, m in model.named_modules()}
        self.stack: list[str] = ["<root>"]
        self.order: list[str] = []  # first-execution order of modules
        self.outputs: WeakTensorKeyDictionary = WeakTensorKeyDictionary()
        self.handles = []
        for _name, module in model.named_modules():
            self.handles.append(module.register_forward_pre_hook(self._enter))
            self.handles.append(module.register_forward_hook(self._exit))

    def _enter(self, module, args):
        name = self.names[id(module)]
        if name not in self.order:
            self.order.append(name)
        self.stack.append(name)

    def _exit(self, module, args, output):
        name = self.stack.pop()
        leaves, _ = tree_flatten(output)
        for index, leaf in enumerate(leaves):
            if isinstance(leaf, torch.Tensor):
                self.outputs.setdefault(leaf, []).append((name, index))

    @property
    def current(self) -> str:
        return self.stack[-1]

    def remove(self):
        for h in self.handles:
            h.remove()


class InterfaceTracer(TorchDispatchMode):
    def __init__(
        self, model: torch.nn.Module, seeds: Mapping[str, Iterable[torch.Tensor]]
    ):
        super().__init__()
        self.model = model
        self.taints: WeakTensorKeyDictionary = WeakTensorKeyDictionary()
        self.param_names: WeakTensorKeyDictionary = WeakTensorKeyDictionary()
        for name, p in model.named_parameters():
            self.param_names[p] = name
        for name, b in model.named_buffers():
            self.param_names[b] = name
        self.seeds = {
            tag: [tuple(t.shape) for t in tensors] for tag, tensors in seeds.items()
        }
        for tag, tensors in seeds.items():
            for t in tensors:
                self.taints[t] = Taint(
                    frozenset({tag}), frozenset(), "<input>", "<seed>"
                )
        self.stack = _ModuleStack(model)
        self.frontier_records: list[tuple] = []
        self.frontier_seen: WeakTensorKeyDictionary = WeakTensorKeyDictionary()
        self.param_usage: dict[
            str, set[str]
        ] = {}  # param name -> set of output-tag signatures
        # eager provenance for visual-only tensors: tensor -> (source module, param-free
        # ops since its output)
        self.sources: WeakTensorKeyDictionary = WeakTensorKeyDictionary()
        self.op_count = 0

    # -- helpers -----------------------------------------------------------------------
    def _taint_of(self, t: torch.Tensor) -> Taint | None:
        taint = self.taints.get(t)
        if taint is not None:
            return taint
        name = self.param_names.get(t)
        if name is not None:
            return Taint(frozenset(), frozenset({name}), "<param>", "<param>")
        return None

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        self.op_count += 1
        leaves, _ = tree_flatten((args, kwargs))
        inputs = [
            (leaf, self._taint_of(leaf))
            for leaf in leaves
            if isinstance(leaf, torch.Tensor)
        ]
        tags: set[str] = set()
        params: set[str] = set()
        param_only_inputs: set[str] = set()
        for _leaf, taint in inputs:
            if taint is None:
                continue
            tags |= taint.tags
            if not taint.tags:
                # a parameter, or a tensor derived from parameters alone (e.g. ``1 +
                # weight``): its
                # usage is classified by the taint of the result it feeds
                param_only_inputs |= taint.params
            params |= taint.params
        if not tags and not params:
            return out
        op = str(func.overloadpacket) if hasattr(func, "overloadpacket") else str(func)
        module = self.stack.current
        out_tags = frozenset(tags)
        signature = "+".join(t for t in TAG_ORDER if t in out_tags) or "const"
        for name in param_only_inputs:
            self.param_usage.setdefault(name, set()).add(signature)
        # frontier detection: visual-only input feeding a text-tainted result
        if TEXT in out_tags:
            for leaf, taint in inputs:
                if (
                    taint is not None
                    and taint.visual_only
                    and leaf not in self.frontier_seen
                ):
                    structural = [
                        f"{list(t.shape)}:{str(t.dtype).replace('torch.', '')}"
                        for t, tn in inputs
                        if tn is not None and tn.tags == frozenset({STRUCTURAL})
                    ]
                    self.frontier_seen[leaf] = True
                    # resolve the source module now: only modules that have already
                    # returned are candidates
                    self.frontier_records.append(
                        (leaf, module, op, out_tags, structural, self._source_of(leaf))
                    )
        track_params = TEXT not in out_tags
        new_taint = Taint(
            out_tags, frozenset(params) if track_params else frozenset(), module, op
        )
        out_leaves, _ = tree_flatten(out)
        source = None
        if track_params and VISUAL in out_tags and not param_only_inputs:
            visual_inputs = [
                leaf
                for leaf, taint in inputs
                if taint is not None and VISUAL in taint.tags
            ]
            if len(visual_inputs) == 1:
                origin = visual_inputs[0]
                names = self.stack.outputs.get(origin)
                if names:
                    module_name, leaf_index = self._outermost(names)
                    source = (module_name, leaf_index, (op,))
                else:
                    prior = self.sources.get(origin)
                    if prior is not None:
                        source = (prior[0], prior[1], prior[2] + (op,))
        for leaf in out_leaves:
            if isinstance(leaf, torch.Tensor):
                self.taints[leaf] = new_taint
                if source is not None:
                    self.sources[leaf] = source
        return out

    # -- analysis ----------------------------------------------------------------------
    def _source_of(self, tensor: torch.Tensor):
        """Nearest module output from which ``tensor`` derives by param-free
        single-input ops.
        """

        names = self.stack.outputs.get(tensor)
        if names:
            module_name, leaf_index = self._outermost(names)
            return module_name, leaf_index, ()
        source = self.sources.get(tensor)
        if source is not None:
            return source
        return None, None, ()

    def _outermost(self, names: list[tuple[str, int]]) -> tuple[str, int]:
        # the shortest dotted path is the outermost module
        return min(names, key=lambda n: (n[0].count("."), len(n[0])))

    def report(self) -> InterfaceReport:
        frontier: list[FrontierTensor] = []
        pre_cut: set[str] = set()
        for (
            tensor,
            consumer_module,
            consumer_op,
            out_tags,
            structural,
            source,
        ) in self.frontier_records:
            taint = self.taints.get(tensor)
            source_module, source_leaf, post_ops = source
            pre_cut |= set(taint.params)
            frontier.append(
                FrontierTensor(
                    shape=tuple(tensor.shape),
                    dtype=str(tensor.dtype).replace("torch.", ""),
                    producer_module=taint.producer,
                    producer_op=taint.op,
                    consumer_module=consumer_module,
                    consumer_op=consumer_op,
                    consumer_output_tags=tuple(t for t in TAG_ORDER if t in out_tags),
                    param_dependencies=tuple(sorted(taint.params)),
                    source_module=source_module,
                    source_leaf_index=source_leaf,
                    postprocess_ops=post_ops,
                    structural_inputs=tuple(structural),
                )
            )
        all_params = list(self.param_names.values())
        used = set(self.param_usage)
        structural_params = {
            n
            for n, sigs in self.param_usage.items()
            if sigs <= {STRUCTURAL, "const"} and n not in pre_cut
        }
        post_cut = {
            n
            for n, sigs in self.param_usage.items()
            if any(TEXT in sig for sig in sigs)
        } - pre_cut
        visual_unconsumed = used - pre_cut - post_cut - structural_params
        unused = (set(all_params) - used) | visual_unconsumed
        # outermost modules whose parameters are entirely pre-cut
        visual_modules = self._entire_modules(pre_cut)
        for f in frontier:
            if f.source_module is not None and not any(
                f.source_module == v or f.source_module.startswith(v + ".")
                for v in visual_modules
            ):
                f.source_module, f.source_leaf_index, f.postprocess_ops = None, None, ()
        cut_modules = tuple(
            sorted({f.source_module for f in frontier if f.source_module})
        )
        return InterfaceReport(
            frontier=frontier,
            pre_cut_params=tuple(sorted(pre_cut)),
            post_cut_params=tuple(sorted(post_cut)),
            structural_params=tuple(sorted(structural_params)),
            unused_params=tuple(sorted(unused)),
            cut_modules=cut_modules,
            visual_modules=visual_modules,
            op_count=self.op_count,
            seeds=self.seeds,
            execution_order=tuple(self.stack.order),
        )

    def _entire_modules(self, pre_cut: set[str]) -> tuple[str, ...]:
        result = []
        for name, module in self.model.named_modules():
            if not name:
                continue
            owned = [n for n, _ in module.named_parameters(prefix=name)] + [
                n for n, _ in module.named_buffers(prefix=name)
            ]
            if owned and all(n in pre_cut for n in owned):
                if not any(name.startswith(r + ".") for r in result):
                    result.append(name)
        return tuple(result)

    def close(self):
        self.stack.remove()


def discover_interface(
    model: torch.nn.Module,
    forward: Any,
    *,
    visual: Iterable[torch.Tensor],
    text: Iterable[torch.Tensor],
    structural: Iterable[torch.Tensor] = (),
) -> InterfaceReport:
    """Run ``forward()`` once under the tracer and return the discovered interface.

    ``forward`` is a zero-argument callable that executes the model on inputs that
    include the seed tensors passed here (the same tensor objects).
    """

    tracer = InterfaceTracer(
        model, {VISUAL: list(visual), TEXT: list(text), STRUCTURAL: list(structural)}
    )
    try:
        with torch.no_grad(), tracer:
            forward()
        return tracer.report()
    finally:
        tracer.close()
