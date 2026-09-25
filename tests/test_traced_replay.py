"""CPU checks of executed-path tracing and generic replay, without model downloads."""

import copy

import pytest

torch = pytest.importorskip("torch")
from reprforge import hooked  # noqa: E402
from reprforge.tracing import discover_interface  # noqa: E402


class Visual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(3))
        self.register_buffer("offset", torch.ones(3), persistent=False)
        self.calls = 0

    def forward(self, pixels):
        self.calls += 1
        main = pixels @ self.weight
        deep = main + self.offset
        return main, deep


class Retriever(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = Visual()
        self.embedding = torch.nn.Embedding(4, 3)
        self.projection = torch.nn.Linear(3, 2)

    def forward(self, pixels, ids):
        main, deep = self.visual(pixels)
        text = self.embedding(ids)
        return self.projection(text + main + deep)


def fixture():
    torch.manual_seed(7)
    model = Retriever().eval()
    pixels = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    ids = torch.tensor([1, 2])
    report = discover_interface(
        model, lambda: model(pixels, ids), visual=[pixels], text=[ids]
    )
    spec = hooked.CutSpec.from_report(report, list(report.execution_order))
    with torch.inference_mode():
        state = hooked.capture(model, spec, lambda: model(pixels, ids))
    return model, pixels, ids, report, state


def test_trace_includes_nonpersistent_buffer_and_both_visual_outputs():
    model, _, _, report, state = fixture()
    assert "visual.offset" not in model.state_dict()
    assert "visual.offset" in report.pre_cut_params
    assert "embedding.weight" not in report.pre_cut_params
    assert set(state["modules"]["visual"]["retained_leaves"]) == {0, 1}


def test_replay_uses_changed_target_text_and_suffix_without_running_vision():
    source, pixels, ids, report, state = fixture()
    target = copy.deepcopy(source)
    with torch.no_grad():
        target.embedding.weight.add_(0.25)
        target.projection.bias.add_(0.125)
    names = list(report.pre_cut_params)
    assert hooked.dependency_digest(source, names) == hooked.dependency_digest(
        target, names
    )
    with torch.inference_mode():
        full = target(pixels, ids)
        calls = target.visual.calls
        replay = hooked.resume(target, state, lambda: target(pixels, ids), device="cpu")
    assert target.visual.calls == calls
    assert torch.equal(replay, full)
    assert not torch.equal(full, state["result"])


def test_loaded_nonpersistent_buffer_change_invalidates_dependency_digest():
    source, _, _, report, _ = fixture()
    target = copy.deepcopy(source)
    target.visual.offset.add_(1)
    names = list(report.pre_cut_params)
    assert hooked.dependency_digest(source, names) != hooked.dependency_digest(
        target, names
    )


def test_input_mismatch_rejects_and_restores_target_module():
    model, pixels, ids, _, state = fixture()
    original = model.visual.forward
    with torch.inference_mode(), pytest.raises(hooked.PageLevelMismatch):
        hooked.resume(model, state, lambda: model(pixels + 1, ids), device="cpu")
    assert model.visual.forward == original
    assert torch.isfinite(model(pixels, ids)).all()


def test_replay_restores_modules_when_target_raises():
    model, _, _, _, state = fixture()
    original = model.visual.forward

    def fail():
        raise ValueError("target failure")

    with pytest.raises(ValueError, match="target failure"):
        hooked.resume(model, state, fail, device="cpu")
    assert model.visual.forward == original


def test_trace_removes_hooks_when_forward_raises():
    model = Retriever()
    pixels = torch.ones(2, 3)
    ids = torch.tensor([1, 2])

    def fail():
        model(pixels, ids)
        raise ValueError("forward failure")

    with pytest.raises(ValueError, match="forward failure"):
        discover_interface(model, fail, visual=[pixels], text=[ids])
    assert all(
        not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()
    )
