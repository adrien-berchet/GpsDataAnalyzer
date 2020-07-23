import numpy as np
import pandas as pd


def check_gps_data(data, x=None, y=None, z=None, t=None, dt=None, dist=None, vel=None):
    if x is not None:
        assert np.allclose(data.x, x, equal_nan=True)
    if y is not None:
        assert np.allclose(data.y, y, equal_nan=True)
    if z is not None:
        assert np.allclose(data.z, z, equal_nan=True)
    if t is not None:
        assert data.t.tolist() == [pd.to_datetime(i) for i in t]
    if dt is not None:
        assert np.allclose(data.dt.values, dt, equal_nan=True)
    if dist is not None:
        assert np.allclose(data.dist.values, dist, equal_nan=True)
    if vel is not None:
        assert np.allclose(data.velocity.values, vel, equal_nan=True)
