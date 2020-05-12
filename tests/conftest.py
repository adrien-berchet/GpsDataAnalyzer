import os
import pytest

from .fixtures.gps_points import *  # noqa
from .fixtures.poi_points import *  # noqa


@pytest.fixture
def gpkg_filepath(tmpdir):
    tmpdir_str = tmpdir.strpath
    filename = "test.gpkg"
    return os.path.join(tmpdir_str, filename)
