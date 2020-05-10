import pandas as pd

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
    assert res.z.tolist() == z
    assert res.t.tolist() == [pd.to_datetime(i) for i in t]
