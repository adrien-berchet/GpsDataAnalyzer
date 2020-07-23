import numpy as np

import gps_data_analyzer as gda

from . import check_gps_data


def test_save_load_gps_points(simple_gps_raw_data, simple_gps_data, gpkg_filepath):
    # Add column to keep
    simple_gps_data["test_col"] = 1

    # Save to temporary file
    gda.io.save(simple_gps_data, gpkg_filepath)

    # Load to GpsPoints
    res = gda.load_gps_points(gpkg_filepath)

    # Check results
    check_gps_data(res, *simple_gps_raw_data)
    assert (res.test_col == 1).all()


def test_save_load_gps_points_crs(simple_gps_raw_data, simple_gps_data, gpkg_filepath):
    # Add column to keep
    simple_gps_data["test_col"] = 1

    # Project data
    projected = gda.GpsPoints(simple_gps_data, local_crs=2154)

    # Save to temporary file
    gda.io.save(projected, gpkg_filepath)

    # Load to GpsPoints
    res = gda.load_gps_points(gpkg_filepath)

    # Check results
    x, y, z, t, dt, dist, vel = simple_gps_raw_data
    check_gps_data(res, projected.x, projected.y, projected.z, t, dt, dist, vel)
    assert res.equals(projected[res.columns])
    assert (res.test_col == 1).all()
    assert projected.crs.to_epsg() == 2154
    assert res.crs.to_epsg() == 2154


def test_save_load_poi_points(simple_poi_raw_data, simple_poi_data, gpkg_filepath):
    # Add column to keep
    simple_poi_data["test_col"] = 1

    # Save to temporary file
    gda.io.save(simple_poi_data, gpkg_filepath)

    # Load to PoiPoints
    poi = gda.load_poi_points(gpkg_filepath)

    # Check results
    x, y, r = simple_poi_raw_data
    assert poi.x.tolist() == x
    assert poi.y.tolist() == y
    assert poi.radius.tolist() == r
    assert (poi.test_col == 1).all()
    assert "z" not in poi.columns
    assert "datetime" not in poi.columns
    assert "dt" not in poi.columns
    assert "dist" not in poi.columns
    assert "velocity" not in poi.columns


def test_save_load_raster(simple_gps_data, zrd_filepath):
    raster = gda.raster_analysis.heatmap(simple_gps_data, mesh_size=0.1)

    raster.save(zrd_filepath)
    res = gda.load_raster(zrd_filepath)

    assert np.array_equal(raster.X, res.X)
    assert np.array_equal(raster.Y, res.Y)
    assert np.array_equal(raster.values, res.values)
    assert raster.extent == res.extent
    assert raster.crs == res.crs

    # Without extension
    raster.save(zrd_filepath[:-4])
    res = gda.load_raster(zrd_filepath)

    assert np.array_equal(raster.X, res.X)
    assert np.array_equal(raster.Y, res.Y)
    assert np.array_equal(raster.values, res.values)
    assert raster.extent == res.extent
    assert raster.crs == res.crs
