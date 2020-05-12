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


def test_poi(simple_poi_data, simple_poi_raw_data):
    assert np.equal(simple_poi_data.x, [0.5]).all()
    assert np.equal(simple_poi_data.y, [0.5]).all()
    assert np.equal(simple_poi_data.radius, [0.75]).all()
    assert simple_poi_data.crs.to_epsg() == 4326


def test_poi_fail(simple_poi_df):
    with pytest.raises(KeyError):
        gda.PoiPoints(simple_poi_df.drop(columns=["radius"]), x_col="x", y_col="y")


def test_mask(simple_poi_data, simple_gps_data):
    assert len(simple_gps_data) == 3

    # Drop from masks
    simple_gps_data.drop_from_mask(simple_poi_data)

    # Check results
    assert len(simple_gps_data) == 1
    assert np.equal(simple_gps_data.xy, [[0.2, 1.2]]).all()
    assert simple_gps_data.index == pd.core.indexes.range.RangeIndex(0, 1, 1)


def test_mask_polygon(simple_poi_data, simple_gps_data):
    assert len(simple_gps_data) == 3

    # Create small polygons aroung PoI points
    polygons = simple_poi_data.buffer(0.01).to_frame("geometry")
    polygons["radius"] = simple_poi_data.radius
    polygons.crs = simple_poi_data.crs

    # Drop from masks
    simple_gps_data.drop_from_mask(polygons)

    # Check results
    assert len(simple_gps_data) == 1
    assert np.equal(simple_gps_data.xy, [[0.2, 1.2]]).all()
    assert simple_gps_data.index == pd.core.indexes.range.RangeIndex(0, 1, 1)
