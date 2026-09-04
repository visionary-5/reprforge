import pytest

from reprforge import COMPONENTS, VersionManifest


def manifest(**overrides: str) -> VersionManifest:
    base = {name: f"{name}-v1" for name in COMPONENTS}
    base.update(overrides)
    return VersionManifest(**base)


def test_changed_components_is_the_exact_delta() -> None:
    old = manifest()
    new = manifest(adapter="adapter-v2", projection="projection-v2")

    assert old.changed_components(new) == frozenset({"adapter", "projection"})
    assert new.changed_components(old) == frozenset({"adapter", "projection"})
    assert old.changed_components(old) == frozenset()


def test_processor_change_is_a_first_class_component() -> None:
    """A release that lowers max_pixels changes the processor contract even if
    the adapter is otherwise decoder-only."""

    old = manifest()
    new = manifest(adapter="adapter-v2", processor="processor-602112px")

    assert "processor" in old.changed_components(new)
    scenario = old.update_scenario(new, "shipped-release")
    assert scenario.changed_components == frozenset({"adapter", "processor"})


def test_identical_versions_do_not_create_an_update() -> None:
    version = manifest()
    with pytest.raises(ValueError, match="identical"):
        version.update_scenario(version, "noop")


def test_round_trip_and_validation() -> None:
    version = manifest()
    assert VersionManifest.from_dict(version.to_dict()) == version
    with pytest.raises(ValueError, match="unknown"):
        VersionManifest.from_dict({**version.to_dict(), "extra": "x"})
    with pytest.raises(ValueError, match="index_policy"):
        VersionManifest(*("valid",) * 6, index_policy="").validate()
