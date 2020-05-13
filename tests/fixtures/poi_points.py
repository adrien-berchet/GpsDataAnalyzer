import pytest

import pandas as pd

import gps_data_analyzer as gda


@pytest.fixture
def simple_poi_raw_data():
    x = [0.15]
    y = [1.15]
    r = [0.1]
    return x, y, r


@pytest.fixture
def simple_poi_df(simple_poi_raw_data):
    x, y, r = simple_poi_raw_data
    df = pd.DataFrame({"x": x, "y": y, "radius": r})
    return df


@pytest.fixture
def simple_poi_data(simple_poi_df):
    return gda.PoiPoints(simple_poi_df, x_col="x", y_col="y")
