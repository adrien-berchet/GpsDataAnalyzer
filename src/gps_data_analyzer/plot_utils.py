import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np


def create_transparent_cmap(name="rainbow"):
    """
    Create a matplotlib.colors.ListedColormap with transparency for the lowest
    value.
    """
    default_cmap = plt.get_cmap(name)
    cmap = default_cmap(np.arange(default_cmap.N))
    cmap[:, -1] = np.linspace(0, 1, default_cmap.N)
    return ListedColormap(cmap)


def add_annotated_points(ax, points, **kwargs):
    """
    Add annotated points to the given axis.
    The 'points' object must be a pandas.DataFrame with the following columns:
        * x
        * y
        * name - Optional: labels are taken if not provided
        * horizontalalignment (or ha) - Optional: the default value is 'center'
        * verticalalignment (or va) - Optional: the default value is 'center'
        * fontsize - Optional: the default value is 12
        * x_offset - Optional: the default value is 0
        * y_offset - Optional: the default value is 0
    The given kwargs will be passed to the ``matplotlib.pyplot.annotate()`` function.
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
    """Setup a matplotlib.pyplot.Axes instance"""

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
        background = cimgt.GoogleTiles(style="satellite")

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
    annotations=None,
    annotations_kwargs=None,
    background=False,
    border=0,
    extent=None,
    zoom=None,
    **kwargs
):
    """Plot data with background and annotations"""

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
        if annotations_kwargs is None:
            annotations_kwargs = {}
        add_annotated_points(ax, annotations, **annotations_kwargs)

    # Show the figure
    if show is True:
        plt.show()
    else:
        return fig, ax


def plot_segments(gps_data, *args, **kwargs):
    """Plot segments with background and annotations"""
    segments = gps_data.segments()

    return plot(segments)
