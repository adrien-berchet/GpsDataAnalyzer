import numpy as np
import pandas as pd
import pyproj
import pytest

import gps_data_analyzer as gda


def test_create_track(simple_gps_data, simple_gps_raw_data):
    x, y, z, t = simple_gps_raw_data

    assert simple_gps_data.x.tolist() == x
    assert simple_gps_data.y.tolist() == y
    assert simple_gps_data.z.tolist() == z
    assert simple_gps_data.t.tolist() == [pd.to_datetime(i) for i in t]


def test_equal_track(simple_gps_data):
    new = gda.GpsPoints(simple_gps_data.data)

    assert new == simple_gps_data


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


def test_poi_fail(simple_poi_df):
    with pytest.raises(KeyError):
        gda.PoiPoints(simple_poi_df.drop(columns=["radius"]), x_col="x", y_col="y")


def test_mask(simple_poi_data, simple_gps_data):
    assert len(simple_gps_data) == 3

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
