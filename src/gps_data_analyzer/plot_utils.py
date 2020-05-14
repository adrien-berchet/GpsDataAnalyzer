import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np


def create_transparent_cmap(name="rainbow"):
    """Create a colormap with transparency for the lowest value.

    Returns:
        ``matplotlib.colors.ListedColormap``: The colormap.
    """
    default_cmap = plt.get_cmap(name)
    cmap = default_cmap(np.arange(default_cmap.N))
    cmap[:, -1] = np.linspace(0, 1, default_cmap.N)
    return ListedColormap(cmap)


def add_annotated_points(ax, points, **kwargs):
    """Add annotated points to the given axis.

    Args:
        ax (``matplotlib.pyplot.Axes``): The axis object to update.
        points (:py:class:`~gps_data_analyzer.gps_data.PoiPoints`): The points to add to
            the axis.
        kwargs: The given kwargs will be passed to :func:`matplotlib.pyplot.annotate`.

    Note:
        The `points` object must be a ``pandas.DataFrame`` with the following columns:

            * x
            * y
            * name (optional): labels are taken if not provided
            * horizontalalignment (or ha) (optional): the default value is 'center'
            * verticalalignment (or va) (optional): the default value is 'center'
            * fontsize (optional): the default value is 12
            * x_offset (optional): the default value is 0
            * y_offset (optional): the default value is 0
    """

    pts = points.copy()

    # If name column is missing, set IDs as names
    if "name" not in pts.columns:
        pts["name"] = pts.index.astype(str)

    # Set default arguments for arrowprops and bbox
    arrowprops = kwargs.pop("arrowprops", dict(facecolor="black", shrink=0.05))
    bbox = kwargs.pop("bbox", dict(facecolor="sandybrown", alpha=0.5, boxstyle="round"))

    # Create a matplotlib transform object for the coordinate system
    proj = ax.projection._as_mpl_transform(ax)

    # Add annotations
    for num, row in pts.iterrows():
        ax.annotate(
            row["name"],
            xy=(row["geometry"].x, row["geometry"].y),
            xytext=(
                row["geometry"].x + row.get("x_offset", 0),
                row["geometry"].y + row.get("y_offset", 0)
            ),
            arrowprops=arrowprops,
            xycoords=proj,
            fontsize=row.get("fontsize", 12),
            ha=row.get("ha", row.get("horizontalalignment", "center")),
            va=row.get("va", row.get("verticalalignment", "center")),
            bbox=bbox,
            **kwargs
        )


def setup_axis(
    ax=None,
    extent=None,
    projection=None,
    background=False,
    zoom=10
):
    """Setup a ``matplotlib.pyplot.Axes`` instance.

    Args:
        ax (``matplotlib.pyplot.Axes``, optional): The axis object to update.
        extent (:py:class:`~gps_data_analyzer.raster_analysis.Extent` or :py:obj:`list`\
            of :py:obj:`float`): The extent to set.
        projection (:py:class:`cartopy.crs.Projection`, optional): The projection of the
            axis (:py:class:`~cartopy.crs.PlateCarree` by default).
        background (bool or :py:class:`cartopy.io.img_tiles.GoogleWTS`, optional): If
            true, a default background is added using Google Satellite. If a
            :obj:`~cartopy.io.img_tiles.GoogleWTS` object is given, it is used.
        zoom (int, mandatory if :py:obj:`background` is not :py:obj:`None`): The zoom
            value used to generate the background.
    """

    # Define projection
    if projection is None:
        projection = ccrs.PlateCarree()

    # Create the figure instance
    if ax is None:
        fig = plt.figure()

        # Create a GeoAxes in the tile's projection.
        ax = fig.add_subplot(1, 1, 1, projection=projection)
    else:
        fig = None

    # Limit the extent of the map to the min/max coords
    if extent is not None:
        ax.set_extent(extent, crs=projection)

    # Create a terrain background instance
    if background is True:
        background = cimgt.GoogleTiles(style="satellite")  # pragma: no cover

    # Add the background data
    if background is not False:
        ax.add_image(background, zoom)

    return fig, ax


def plot(
    data,
    var=None,
    vmin=None,
    vmax=None,
    ax=None,
    show=True,
    projection=None,
    background=False,
    zoom=None,
    annotations=None,
    annotation_kwargs=None,
    border=0,
    extent=None,
    **kwargs
):
    """Plot data with background and annotations.

    Args:
        data (:py:class:`~gps_data_analyzer.gps_data._GpsBase`): The object to plot.
        var (str, optional): The column used as plotted value.
        vmin (float, optional): The lower clipping value.
        vmax (float, optional): The upper clipping value.
        ax (``matplotlib.pyplot.Axes``, optional): The axis object to update.
        show (bool, optional): If true, call :func:`plt.show` else return the figure
            and axis objects.
        projection (:py:class:`cartopy.crs.Projection`, optional): The projection of the
            axis (:py:class:`~cartopy.crs.PlateCarree` by default).
        background (bool or :py:class:`cartopy.io.img_tiles.GoogleWTS`, optional): If
            true, a default background is added using Google Satellite. If a
            :obj:`~cartopy.io.img_tiles.GoogleWTS` object is given, it is used.
        zoom (int, mandatory if :py:obj:`background` is not :py:obj:`None`): The zoom
            value used to generate the background.
        annotations (:py:class:`~gps_data_analyzer.gps_data.PoiPoints`, optional): The
            points used to annotate the figure.
        annotation_kwargs (dict): The kwargs passed to
            :func:`~gps_data_analyzer.plot_utils.add_annotated_points`.
        border (float, optional): The extra border around the data (not used if
            :py:attr:`extent` is given).
        extent (:py:class:`~gps_data_analyzer.raster_analysis.Extent` or :py:obj:`list`\
            of :py:obj:`float`): The extent to set.
        kwargs: The given kwargs will be passed to :func:`pandas.DataFrame.plot`.

    Returns:
        The figure and axis if :py:attr:`show` is False, :py:obj:`None` otherwise.
    """

    # Limit the extent of the map to the min/max coords
    if extent is None:
        xmin, ymin, xmax, ymax = data.sindex.bounds
        extent = (xmin - border, xmax + border, ymin - border, ymax + border)

    # Setup axis
    fig, ax = setup_axis(
        ax=ax,
        extent=extent,
        projection=projection,
        background=background,
        zoom=zoom
    )

    # Clip values
    if vmin is not None or vmax is not None:
        data = data.copy()
        data[var] = data[var].clip(vmin, vmax, inplace=True)

    # Add data
    data.plot(column=var, ax=ax, **kwargs)

    # Add annotations
    if annotations is not None:
        if annotation_kwargs is None:
            annotation_kwargs = {}
        add_annotated_points(ax, annotations, **annotation_kwargs)

    # Show the figure
    if show is True:
        plt.show()  # pragma: no cover
    else:
        return fig, ax


def plot_segments(gps_data, *args, **kwargs):
    """Plot segments with background and annotations.

    Args:
        gps_data (:py:class:`~gps_data_analyzer.gps_data._GpsBase`): The object to plot.

    Note:
        The given object is converted into segments then plotted using
        :func:`~gps_data_analyzer.plot_utils.plot`.
    """
    segments = gps_data.segments()

    return plot(segments, *args, **kwargs)
