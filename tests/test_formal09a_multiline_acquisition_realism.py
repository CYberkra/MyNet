import numpy as np

from scripts.generate_formal09a_multiline_acquisition_realism import (
    Variant,
    apply_realism,
    build_realism_basis,
    target_background_ratio,
)


def synthetic_case() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time_ns = np.arange(501, dtype=np.float64) * 1.4
    traces = 24
    path = 410.0 + 5.0 * np.sin(np.linspace(0.0, np.pi, traces))
    raw = np.zeros((time_ns.size, traces), dtype=np.float64)
    for trace, center in enumerate(path):
        phase = (time_ns - center) / 7.0
        raw[:, trace] = np.exp(-0.5 * np.square(phase)) * np.cos(2.0 * np.pi * phase)
    return raw, time_ns, path


def test_realism_basis_is_deterministic() -> None:
    raw, time_ns, _ = synthetic_case()
    left = build_realism_basis(raw.shape, 1.4e-9, time_ns, seed=17)
    right = build_realism_basis(raw.shape, 1.4e-9, time_ns, seed=17)
    assert np.array_equal(left[0], right[0])
    assert np.array_equal(left[1], right[1])


def test_realism_hits_declared_target_ratio() -> None:
    raw, time_ns, path = synthetic_case()
    basis, gain = build_realism_basis(raw.shape, 1.4e-9, time_ns, seed=19)
    variant = Variant("test", 3.5, 0.1)
    realized, report = apply_realism(
        raw, time_ns, path, basis, gain, variant, calibration_end_ns=time_ns[-1]
    )
    ratio = target_background_ratio(realized, time_ns, path)
    assert abs(ratio - 3.5) < 1e-6
    assert abs(report["achieved_target_to_background"] - 3.5) < 1e-6


def test_stronger_variant_adds_more_background_energy() -> None:
    raw, time_ns, path = synthetic_case()
    basis, gain = build_realism_basis(raw.shape, 1.4e-9, time_ns, seed=23)
    mild, mild_report = apply_realism(
        raw,
        time_ns,
        path,
        basis,
        gain,
        Variant("mild", 8.0, 0.06),
        calibration_end_ns=time_ns[-1],
    )
    strong, strong_report = apply_realism(
        raw,
        time_ns,
        path,
        basis,
        gain,
        Variant("strong", 2.5, 0.18),
        calibration_end_ns=time_ns[-1],
    )
    assert strong_report["noise_scale"] > mild_report["noise_scale"]
    assert np.std(strong - raw) > np.std(mild - raw)
