from pathlib import Path

import numpy as np

from scripts.generate_formal09b2_lateral_covariance import SpatialFit
from scripts.generate_formal09b2r1_joint_spectrum import (
    JointSpectrumFit,
    build_joint_basis,
    equal_line_joint_pool,
    fit_line_joint_spectrum,
)


def dummy_fits() -> tuple[JointSpectrumFit, SpatialFit]:
    temporal = np.linspace(-250.0, 250.0, 201)
    spatial = np.linspace(-0.7, 0.7, 141)
    ft, fx = np.meshgrid(temporal, spatial, indexing="ij")
    amplitude = np.exp(-0.5 * ((np.abs(ft) - 90.0) / 35.0) ** 2)
    amplitude *= np.exp(-0.5 * (fx / 0.25) ** 2)
    joint = JointSpectrumFit(
        "joint", ("Line3",), temporal, spatial, amplitude, {"Line3": amplitude}, {"Line3": 4}
    )
    positive_spatial = np.linspace(0.0, 0.7, 141)
    marginal = np.exp(-positive_spatial / 0.2)
    spatial_fit = SpatialFit(
        "spatial",
        ("Line3",),
        positive_spatial,
        marginal,
        {"Line3": marginal},
        {"Line3": 4},
        {"Line3": 0.2},
        {"Line3": 4.0},
        0.2,
        4.0,
    )
    return joint, spatial_fit


def test_equal_line_joint_pool_is_direction_symmetric() -> None:
    left = np.ones((5, 7))
    left[:, 1] = 8.0
    pooled = equal_line_joint_pool([left])
    assert np.allclose(pooled, pooled[:, ::-1])


def test_joint_basis_is_deterministic_and_bounded() -> None:
    joint, spatial = dummy_fits()
    time_ns = np.arange(501, dtype=np.float64) * 1.4
    left = build_joint_basis(
        (501, 32), 1.4e-9, time_ns, joint, spatial, 0.72, 17, nonstationary=True
    )
    right = build_joint_basis(
        (501, 32), 1.4e-9, time_ns, joint, spatial, 0.72, 17, nonstationary=True
    )
    assert all(np.array_equal(a, b) for a, b in zip(left, right))
    assert np.min(left[3]) >= 0.45
    assert np.max(left[3]) <= 2.2


def test_real_joint_fit_is_finite_and_uses_multiple_patches() -> None:
    amplitude, count = fit_line_joint_spectrum(
        Path("data/measured/yingshan_v15/lines/Line3.npz"), [(21, 1791)]
    )
    assert amplitude.shape == (201, 141)
    assert count > 10
    assert np.all(np.isfinite(amplitude))
    assert np.allclose(amplitude, amplitude[::-1, ::-1])
