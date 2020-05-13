import numpy as np
import pandas as pd

import gps_data_analyzer as gda


def test_simple_rest_time():
    x = [i for i in range(20)]
    y = [0] * 20
    z = [0] * 20
    t = ["2019/06/16-09:55:{:0}".format(i) for i in range(20)]
    df = pd.DataFrame({"x": x, "y": y, "z": z, "t": t})
    data = gda.GpsPoints(df, x_col="x", y_col="y", z_col="z", time_col="t")

    res = gda.time_analysis.compute_rest_time(data, 5)

    assert np.equal(res[:5], range(5, 10)).all()
    assert (res[5:-5] == 10).all()
    assert np.equal(res[-5:], range(9, 4, -1)).all()


def test_rest_time_with_loops():
    x = [i for i in range(11)] + [i for i in range(9, -1, -1)]
    y = [0] * 21
    z = [0] * 21
    t = ["2019/06/16-09:55:{:0}".format(i) for i in range(21)]
    df = pd.DataFrame({"x": x, "y": y, "z": z, "t": t})
    data = gda.GpsPoints(df, x_col="x", y_col="y", z_col="z", time_col="t")

    res = gda.time_analysis.compute_rest_time(data, 5)

    assert np.equal(res[:5], range(5, 10)).all()
    assert res[5] == 20
    assert np.equal(res[6:11], range(18, 9, -2)).all()
    assert np.equal(res[10:15], range(10, 19, 2)).all()
    assert res[15] == 20
    assert np.equal(res[-5:], range(9, 4, -1)).all()
