import pytest

import pandas as pd

import gps_data_analyzer as gda


@pytest.fixture
def simple_gps_raw_data():
    x = [0, 0.1, 0.2]
    y = [1.0, 1.1, 1.2]
    z = [0, 100, 200]
    t = ["2019/06/16-09:55:34", "2019/06/16-09:55:57", "2019/06/16-09:58:12"]
    dt = [float('nan'), 23, 135]
    dist = [float('nan'), 15724.016124, 15723.752703]
    velocity = [float('nan'), 683.652875, 116.472242]
    return x, y, z, t, dt, dist, velocity


@pytest.fixture
def simple_gps_df(simple_gps_raw_data):
    x, y, z, t, _, _, _ = simple_gps_raw_data
    df = pd.DataFrame({"x": x, "y": y, "z": z, "t": t})
    return df


@pytest.fixture
def simple_gps_data(simple_gps_df):
    return gda.GpsPoints(simple_gps_df, x_col="x", y_col="y", z_col="z", time_col="t")
