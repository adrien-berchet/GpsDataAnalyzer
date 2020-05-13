import pytest

import numpy as np

import gps_data_analyzer as gda


_res_X = [
    [-0.15, -0.15, -0.15, -0.15, -0.15],
    [-0.025, -0.025, -0.025, -0.025, -0.025],
    [0.1, 0.1, 0.1, 0.1, 0.1],
    [0.225, 0.225, 0.225, 0.225, 0.225],
    [0.35, 0.35, 0.35, 0.35, 0.35],
]
_res_Y = [
    [0.85, 0.975, 1.1, 1.225, 1.35],
    [0.85, 0.975, 1.1, 1.225, 1.35],
    [0.85, 0.975, 1.1, 1.225, 1.35],
    [0.85, 0.975, 1.1, 1.225, 1.35],
    [0.85, 0.975, 1.1, 1.225, 1.35],
]
_res_values = [
    [0.0, 0.190739, 0.353963, 0.333906, 0.14622],
    [0.190739, 0.544946, 0.787424, 0.685123, 0.333906],
    [0.353963, 0.787424, 1.0, 0.787424, 0.353963],
    [0.333906, 0.685123, 0.787424, 0.544946, 0.190739],
    [0.14622, 0.333906, 0.353963, 0.190739, 0.0],
]


def test_extent():
    extent = gda.raster_analysis.Extent(0, 1, 2, 3)

    assert extent.xmin == extent[0] == 0
    assert extent.xmax == extent[1] == 1
    assert extent.ymin == extent[2] == 2
    assert extent.ymax == extent[3] == 3
    assert [i for i in extent] == [0, 1, 2, 3]


def test_extent_border():
    extent = gda.raster_analysis.Extent(0, 1, 2, 3, 0.1)

    assert extent.inner_xmin == 0
    assert extent.inner_xmax == 1
    assert extent.inner_ymin == 2
    assert extent.inner_ymax == 3
    assert [i for i in extent] == [-0.1, 1.1, 1.9, 3.1]

    extent.reset_border(0.2)

    assert extent.inner_xmin == 0
    assert extent.inner_xmax == 1
    assert extent.inner_ymin == 2
    assert extent.inner_ymax == 3
    assert [i for i in extent] == [-0.2, 1.2, 1.8, 3.2]


def test_simple_heatmap(simple_gps_data):
    res = gda.raster_analysis.heatmap(simple_gps_data, mesh_size=0.1)

    assert np.equal(res.X, [[0, 0], [0.2, 0.2]]).all()
    assert np.equal(res.Y, [[1, 1.2], [1, 1.2]]).all()
    assert np.allclose(res.values, [[0, 1], [1, 0]])


def test_heatmap_no_normalization(simple_gps_data):
    res = gda.raster_analysis.heatmap(simple_gps_data, mesh_size=0.1, normalize=False)

    assert np.equal(res.X, [[0, 0], [0.2, 0.2]]).all()
    assert np.equal(res.Y, [[1, 1.2], [1, 1.2]]).all()
    assert np.allclose(res.values, [[3.042058, 3.103057], [3.103057, 3.042058]])


def test_heatmap_border(simple_gps_data):
    res = gda.raster_analysis.heatmap(simple_gps_data, mesh_size=0.1, border=0.15)

    assert np.allclose(res.X, _res_X,)
    assert np.allclose(res.Y, _res_Y)
    assert np.allclose(res.values, _res_values)


def test_heatmap_nx_ny(simple_gps_data):
    res = gda.raster_analysis.heatmap(simple_gps_data, nx=5, ny=5, border=0.15)

    assert np.allclose(res.X, _res_X,)
    assert np.allclose(res.Y, _res_Y)
    assert np.allclose(res.values, _res_values)


def test_heatmap_args(simple_gps_data):
    # Test conflicting arguments
    for k in ["mesh_size", "x_size", "y_size", "nx", "ny"]:
        with pytest.raises(ValueError):
            gda.raster_analysis.heatmap(simple_gps_data, **{k: 1})

    with pytest.raises(ValueError):
        gda.raster_analysis.heatmap(simple_gps_data, mesh_size=1, nx=1, ny=1)

    with pytest.raises(ValueError):
        gda.raster_analysis.heatmap(simple_gps_data, mesh_size=1, x_size=1)

    with pytest.raises(ValueError):
        gda.raster_analysis.heatmap(simple_gps_data, x_size=1, nx=1, ny=1)

    with pytest.raises(ValueError):
        gda.raster_analysis.heatmap(simple_gps_data, y_size=1, nx=1, ny=1)

    with pytest.raises(ValueError):
        gda.raster_analysis.heatmap(simple_gps_data, x_size=1, y_size=1, nx=1, ny=1)

    with pytest.raises(ValueError):
        gda.raster_analysis.heatmap(
            simple_gps_data, mesh_size=1, x_size=1, y_size=1, nx=1, ny=1
        )

    # Test negative values
    with pytest.raises(ValueError):
        gda.raster_analysis.heatmap(simple_gps_data, mesh_size=-1)

    with pytest.raises(ValueError):
        gda.raster_analysis.heatmap(simple_gps_data, x_size=-1, y_size=1)

    with pytest.raises(ValueError):
        gda.raster_analysis.heatmap(simple_gps_data, x_size=1, y_size=-1)

    with pytest.raises(ValueError):
        gda.raster_analysis.heatmap(simple_gps_data, nx=-1, ny=1)

    with pytest.raises(ValueError):
        gda.raster_analysis.heatmap(simple_gps_data, nx=1, ny=-1)

    with pytest.raises(ValueError):
        gda.raster_analysis.heatmap(simple_gps_data, kernel_size=-1)

    with pytest.raises(ValueError):
        gda.raster_analysis.heatmap(simple_gps_data, kernel_cut=-1)
