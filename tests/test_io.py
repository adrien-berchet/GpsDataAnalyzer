import os

import numpy as np
import pandas as pd

import gps_data_analyzer as gda


def test_save_load(simple_gps_raw_data, simple_gps_data, tmpdir):
    tmpdir_str = tmpdir.strpath
    filename = "test.gpz"
    path = os.path.join(tmpdir_str, filename)

    gda.io.save(simple_gps_data, path)
    res = gda.io.load(path)

    x, y, z, t = simple_gps_raw_data
    assert res["x"].tolist() == x
    assert res["y"].tolist() == y
    assert res["z"].tolist() == z
    assert res["datetime"].tolist() == [pd.to_datetime(i) for i in t]
    assert np.allclose(res["dt"].values, [np.nan, 23.0, 135.0], equal_nan=True)
    assert np.allclose(res["dist"].values, [np.nan, 0.141421, 0.141421], equal_nan=True)
    assert np.allclose(
        res["velocity"].values, [np.nan, 0.00614875, 0.00104757], equal_nan=True
    )
