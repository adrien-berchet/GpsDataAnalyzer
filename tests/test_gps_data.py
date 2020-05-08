import pandas as pd


def test_create_track(simple_gps_data, simple_gps_raw_data):
    x, y, z, t = simple_gps_raw_data

    assert simple_gps_data.x.tolist() == x
    assert simple_gps_data.y.tolist() == y
    assert simple_gps_data.z.tolist() == z
    assert simple_gps_data.t.tolist() == [
        pd.to_datetime(i) for i in t
    ]
