import pytest
import types

import cartopy.io.img_tiles as cimgt
import numpy as np

import gps_data_analyzer as gda


def GOOGLE_IMAGE_URL_REPLACEMENT(self, tile):
    url = ('https://chart.googleapis.com/chart?chst=d_text_outline&'
           'chs=256x256&chf=bg,s,00000055&chld=FFFFFF|16|h|000000|b||||'
           'Google:%20%20(' + str(tile[0]) + ',' + str(tile[1]) + ')'
           '|Zoom%20' + str(tile[2]) + '||||||______________________'
           '______')
    return url


@pytest.fixture
def background():
    """
    This comes from the tests of the Cartopy library. It ensures that the background
    will never be changed by google and is only used in tests.
    """
    gt = cimgt.GoogleTiles()
    gt._image_url = types.MethodType(GOOGLE_IMAGE_URL_REPLACEMENT, gt)
    return gt


@pytest.mark.mpl_image_compare
def test_plot(simple_gps_data):
    # Simple plot
    fig, ax = gda.plot_utils.plot(simple_gps_data, var="z", show=False)
    return fig


@pytest.mark.mpl_image_compare
def test_plot_border(simple_gps_data):
    # Plot with border
    fig, ax = gda.plot_utils.plot(simple_gps_data, var="z", border=0.02, show=False)
    return fig


@pytest.mark.mpl_image_compare
def test_plot_cmap(simple_gps_data):
    # Plot with cmap
    fig, ax = gda.plot_utils.plot(
        simple_gps_data, var="z", border=0.02, cmap="Spectral", show=False)
    return fig


@pytest.mark.mpl_image_compare
def test_plot_annotations(simple_gps_data, simple_poi_data):
    # Plot with cmap
    fig, ax = gda.plot_utils.plot(
        simple_gps_data,
        var="z",
        border=0.02,
        show=False,
        annotations=simple_poi_data)
    return fig


@pytest.mark.mpl_image_compare
def test_plot_background(simple_gps_data, background):
    # Plot with background
    fig, ax = gda.plot_utils.plot(
        simple_gps_data,
        var="z",
        background=background,
        zoom=11,
        border=0.02,
        show=False)
    return fig


@pytest.mark.mpl_image_compare
def test_plot_segments(simple_gps_data):
    # Plot segments
    fig, ax = gda.plot_utils.plot_segments(
        simple_gps_data,
        var="velocity",
        show=False
    )
    return fig


@pytest.mark.mpl_image_compare
def test_plot_raster(simple_gps_data):
    # Define extent
    extent = gda.raster_analysis.Extent(0, 1, 0, 1, 0.1)

    # Define raster
    X, Y = extent.mesh(nx=5, ny=5)
    values = np.reshape(np.arange(0, 25), X.shape)
    raster = gda.raster_analysis.Raster(X, Y, values, extent)

    # Plot raster
    fig, _ = raster.plot(show=False)

    return fig


@pytest.mark.mpl_image_compare
def test_plot_heatmap(simple_gps_data):
    # Define raster
    raster = gda.raster_analysis.heatmap(
        simple_gps_data, weight_col="z", nx=15, ny=15, border=0.05)

    # Plot raster
    fig, ax = raster.plot(show=False)

    # Add track markers
    x = simple_gps_data.x
    y = simple_gps_data.y
    ax.plot(x, y, "k.", markersize=10, transform=ax.projection, zorder=15)

    return fig


@pytest.mark.mpl_image_compare
def test_plot_heatmap_annotations(simple_gps_data, simple_poi_data):
    # Define raster
    raster = gda.raster_analysis.heatmap(
        simple_gps_data, weight_col="z", nx=15, ny=15, border=0.05)

    # Plot raster and annotations
    fig, ax = raster.plot(show=False, annotations=simple_poi_data)

    # Add track markers
    x = simple_gps_data.x
    y = simple_gps_data.y
    ax.plot(x, y, "k.", markersize=10, transform=ax.projection, zorder=15)

    return fig
