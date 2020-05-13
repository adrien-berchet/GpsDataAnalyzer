import numpy as np
import pandas as pd

import gps_data_analyzer as gda


def test_save_load_gps_points(simple_gps_raw_data, simple_gps_data, gpkg_filepath):
    # Save to temporary file
    gda.io.save(simple_gps_data, gpkg_filepath)

    # Load to GpsPoints
    res = gda.load_gps_points(gpkg_filepath)

    # Check results
    x, y, z, t = simple_gps_raw_data
    assert res.x.tolist() == x
    assert res.y.tolist() == y
    assert res.z.tolist() == z
    assert res.datetime.tolist() == [pd.to_datetime(i) for i in t]
    assert np.allclose(res.dt.values, [np.nan, 23.0, 135.0], equal_nan=True)
    assert np.allclose(res.dist.values, [np.nan, 15724.02, 15723.75], equal_nan=True)
    assert np.allclose(
        res.velocity.values, [np.nan, 683.6529, 116.4722], equal_nan=True
    )


def test_save_load_poi_points(simple_poi_raw_data, simple_poi_data, gpkg_filepath):
    # Save to temporary file
    gda.io.save(simple_poi_data, gpkg_filepath)

    # Load to PoiPoints
    poi = gda.load_poi_points(gpkg_filepath)

    # Check results
    x, y, r = simple_poi_raw_data
    assert poi.x.tolist() == x
    assert poi.y.tolist() == y
    assert poi.radius.tolist() == r
    assert "z" not in poi.columns
    assert "datetime" not in poi.columns
    assert "dt" not in poi.columns
    assert "dist" not in poi.columns
    assert "velocity" not in poi.columns
