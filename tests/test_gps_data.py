import pytest

import numpy as np
import pandas as pd
import pyproj

import gps_data_analyzer as gda


def test_create_track(simple_gps_data, simple_gps_raw_data):
    x, y, z, t = simple_gps_raw_data
    df = pd.DataFrame({"x": x, "y": y, "z": z, "t": t})
    new = gda.GpsPoints(df, x_col="x", y_col="y", z_col="z", time_col="t")

    assert new.x.tolist() == x
    assert new.y.tolist() == y
    assert new.z.tolist() == z
    assert new.t.tolist() == [pd.to_datetime(i) for i in t]
    assert new.base_columns == ["geometry", "z", "datetime"]

    new_proj = gda.GpsPoints(new, local_crs=4326)

    assert new_proj.x.tolist() == x
    assert new_proj.y.tolist() == y
    assert new_proj.z.tolist() == z
    assert new_proj.t.tolist() == [pd.to_datetime(i) for i in t]
    assert new_proj.base_columns == ["geometry", "z", "datetime"]


def test_equal_track(simple_gps_data):
    new = gda.GpsPoints(simple_gps_data)

    assert new.equals(simple_gps_data)
    assert np.equal((new == simple_gps_data), np.array(
        [
            [True, True, False, False, False],
            [True, True, True, True, True],
            [True, True, True, True, True]
        ])).all(axis=None)


def test_copy_track(simple_gps_data):
    new = simple_gps_data.copy()

    assert new.equals(simple_gps_data)
    assert id(new) != id(simple_gps_data)

    new["dt"] *= 2
    assert new.equals(simple_gps_data)

    deep_new = simple_gps_data.copy(deep=True)

    assert deep_new.equals(simple_gps_data)
    assert id(deep_new) != id(simple_gps_data)

    deep_new["dt"] *= 2
    assert not deep_new.equals(simple_gps_data)


def test_track_projection(simple_gps_data):
    new = simple_gps_data.copy()

    assert new.equals(simple_gps_data)
    assert not new.to_crs(2154).equals(simple_gps_data)

    new_proj = new.to_crs(4326)
    assert new_proj.geom_almost_equals(simple_gps_data).all()

    new.to_crs(4326, inplace=True)
    assert new.geom_almost_equals(simple_gps_data).all()

    deep_new = simple_gps_data.copy(deep=True)

    assert deep_new.equals(simple_gps_data)
    assert not deep_new.to_crs(2154).equals(simple_gps_data)

    deep_new.to_crs(2154, inplace=True)
    assert not deep_new.equals(simple_gps_data)

    deep_new_proj = deep_new.to_crs(4326)
    assert deep_new_proj.geom_almost_equals(simple_gps_data).all()

    deep_new.to_crs(4326, inplace=True)
    assert deep_new.geom_almost_equals(simple_gps_data).all()


def test_create_track_sort(simple_gps_df, simple_gps_raw_data):
    x, y, z, t = simple_gps_raw_data

    df = simple_gps_df.loc[[2, 0, 1]]
    res = gda.GpsPoints(df, x_col="x", y_col="y", z_col="z", time_col="t")

    assert res.x.tolist() == x
    assert res.y.tolist() == y
    assert np.equal(res.xy, np.vstack((x, y)).T).all()
    assert res.z.tolist() == z
    assert res.t.tolist() == [pd.to_datetime(i) for i in t]
    assert res.crs.to_epsg() == 4326
    assert np.allclose(res.dt.values, [np.nan, 23.0, 135.0], equal_nan=True)
    assert np.allclose(res.dist.values, [np.nan, 15724.02, 15723.75], equal_nan=True)
    assert np.allclose(
        res.velocity.values, [np.nan, 683.6529, 116.4722], equal_nan=True
    )


def test_create_track_crs(simple_gps_df, simple_gps_raw_data):
    x, y, z, t = simple_gps_raw_data

    # Compute projected coordinates
    proj = pyproj.Proj(2154)
    xy_proj = [proj(i, j) for i, j in zip(x, y)]
    x_proj = [i[0] for i in xy_proj]
    y_proj = [i[1] for i in xy_proj]

    df = simple_gps_df.copy()
    df["x"] = x_proj
    df["y"] = y_proj
    res = gda.GpsPoints(
        df, x_col="x", y_col="y", z_col="z", time_col="t", crs=2154, local_crs=4326
    )

    # Check results
    assert np.allclose(res.x, x)
    assert np.allclose(res.y, y)
    assert res.z.tolist() == z
    assert res.t.tolist() == [pd.to_datetime(i) for i in t]
    assert res.crs.to_epsg() == 4326
    assert np.allclose(res.dt.values, [np.nan, 23.0, 135.0], equal_nan=True)
    assert np.allclose(res.dist.values, [np.nan, 15724.02, 15723.75], equal_nan=True)
    assert np.allclose(
        res.velocity.values, [np.nan, 683.6529, 116.4722], equal_nan=True
    )


def test_create_track_proj(simple_gps_df, simple_gps_raw_data):
    x, y, z, t = simple_gps_raw_data

    df = simple_gps_df.loc[[2, 0, 1]]
    res = gda.GpsPoints(
        df, x_col="x", y_col="y", z_col="z", time_col="t", local_crs=2154
    )

    # Compute projected results
    proj = pyproj.Proj(2154)
    xy_proj = [proj(i, j) for i, j in zip(x, y)]
    x_proj = [i[0] for i in xy_proj]
    y_proj = [i[1] for i in xy_proj]

    # Check results
    assert res.x.tolist() == x_proj
    assert res.y.tolist() == y_proj
    assert res.z.tolist() == z
    assert res.t.tolist() == [pd.to_datetime(i) for i in t]
    assert res.crs.to_epsg() == 2154
    assert np.allclose(res.dt.values, [np.nan, 23.0, 135.0], equal_nan=True)
    assert np.allclose(res.dist.values, [np.nan, 20707.888, 20682.199], equal_nan=True)
    assert np.allclose(
        res.velocity.values, [np.nan, 900.343, 153.201], equal_nan=True
    )


def test_add_attribute(simple_gps_data):
    new_attr = simple_gps_data.x * 2
    new_attr.rename("two_x", inplace=True)
    simple_gps_data.add_attribute(new_attr)
    assert (simple_gps_data.two_x.tolist() == (simple_gps_data.x * 2)).all()


def test_add_attribute_name(simple_gps_data):
    new_attr = simple_gps_data.x * 2
    simple_gps_data.add_attribute(new_attr, name="two_x")
    assert (simple_gps_data.two_x.tolist() == (simple_gps_data.x * 2)).all()


def test_poi(simple_poi_data, simple_poi_raw_data):
    x, y, r = simple_poi_raw_data
    assert np.equal(simple_poi_data.x, x).all()
    assert np.equal(simple_poi_data.y, y).all()
    assert np.equal(simple_poi_data.radius, r).all()
    assert simple_poi_data.crs.to_epsg() == 4326
    assert simple_poi_data.base_columns == ["geometry"]
    with pytest.raises(AttributeError):
        simple_poi_data.z
    with pytest.raises(AttributeError):
        simple_poi_data.t


def test_mask(simple_poi_data, simple_gps_data):
    assert len(simple_gps_data) == 3

    # Drop from masks
    N = simple_gps_data.drop_from_mask(simple_poi_data)

    # Check results
    assert N == 2
    assert len(simple_gps_data) == 1
    assert np.equal(simple_gps_data.xy, [[0, 1]]).all()
    assert simple_gps_data.index == pd.core.indexes.range.RangeIndex(0, 1, 1)


def test_mask_no_csr(simple_poi_data, simple_gps_data):
    assert len(simple_gps_data) == 3
    simple_gps_data.crs = None

    # Drop from masks
    N = simple_gps_data.drop_from_mask(simple_poi_data)

    # Check results
    assert N == 2
    assert len(simple_gps_data) == 1
    assert np.equal(simple_gps_data.xy, [[0, 1]]).all()
    assert simple_gps_data.index == pd.core.indexes.range.RangeIndex(0, 1, 1)


def test_mask_polygon(simple_poi_data, simple_gps_data):
    assert len(simple_gps_data) == 3

    # Create small polygons aroung PoI points
    polygons = simple_poi_data.buffer(0.01).to_frame("geometry")
    polygons["radius"] = simple_poi_data.radius
    polygons.crs = simple_poi_data.crs

    # Drop from masks
    N = simple_gps_data.drop_from_mask(polygons)

    # Check results
    assert N == 2
    assert len(simple_gps_data) == 1
    assert np.equal(simple_gps_data.xy, [[0, 1]]).all()
    assert simple_gps_data.index == pd.core.indexes.range.RangeIndex(0, 1, 1)


def test_mask_series(simple_poi_data, simple_gps_data):
    assert len(simple_gps_data) == 3

    # Create small polygons aroung PoI points
    polygons = simple_poi_data.buffer(simple_poi_data.radius).to_frame("geometry")
    polygons.crs = simple_poi_data.crs

    # Drop from masks
    N = simple_gps_data.drop_from_mask(polygons.geometry)

    # Check results
    assert N == 2
    assert len(simple_gps_data) == 1
    assert np.equal(simple_gps_data.xy, [[0, 1]]).all()
    assert simple_gps_data.index == pd.core.indexes.range.RangeIndex(0, 1, 1)


def test_mask_polygon_no_radius(simple_poi_data, simple_gps_data):
    assert len(simple_gps_data) == 3

    # Create small polygons aroung PoI points
    polygons = simple_poi_data.buffer(0.1).to_frame("geometry")
    polygons.crs = simple_poi_data.crs

    # Drop from masks
    N = simple_gps_data.drop_from_mask(polygons)

    # Check results
    assert N == 2
    assert len(simple_gps_data) == 1
    assert np.equal(simple_gps_data.xy, [[0, 1]]).all()
    assert simple_gps_data.index == pd.core.indexes.range.RangeIndex(0, 1, 1)


def test_concatenate(simple_gps_data):
    a = simple_gps_data.copy()
    b = simple_gps_data.copy()
    res = gda.concatenate([a, b])

    assert res.x.tolist() == simple_gps_data.x.tolist() * 2
    assert res.y.tolist() == simple_gps_data.y.tolist() * 2
    assert res.z.tolist() == simple_gps_data.z.tolist() * 2
    assert (
        res.datetime.apply(
            pd.Timestamp.isoformat
        ).tolist() == simple_gps_data.datetime.apply(
            pd.Timestamp.isoformat
        ).tolist() * 2
    )
    assert np.allclose(
        res.dt.tolist(), simple_gps_data.dt.tolist() * 2, equal_nan=True)
    assert np.allclose(
        res.dist.tolist(), simple_gps_data.dist.tolist() * 2, equal_nan=True)
    assert np.allclose(
        res.velocity.tolist(), simple_gps_data.velocity.tolist() * 2, equal_nan=True)


def test_concatenate_csr(simple_gps_data):
    a = simple_gps_data.to_crs(3857)
    b = simple_gps_data.to_crs(2154)
    res = gda.concatenate([a, b], crs=4326)

    assert np.allclose(res.x.tolist(), simple_gps_data.x.tolist() * 2)
    assert np.allclose(res.y.tolist(), simple_gps_data.y.tolist() * 2)
    assert np.allclose(res.z.tolist(), simple_gps_data.z.tolist() * 2)
    assert (
        res.datetime.apply(
            pd.Timestamp.isoformat
        ).tolist() == simple_gps_data.datetime.apply(
            pd.Timestamp.isoformat
        ).tolist() * 2
    )
    assert np.allclose(
        res.dt.tolist(), simple_gps_data.dt.tolist() * 2, equal_nan=True)
    assert np.allclose(
        res.dist.tolist(), simple_gps_data.dist.tolist() * 2, equal_nan=True)
    assert np.allclose(
        res.velocity.tolist(), simple_gps_data.velocity.tolist() * 2, equal_nan=True)
