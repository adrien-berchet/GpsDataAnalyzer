from collections.abc import Iterable

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_cmap():
    default_cmap = plt.get_cmap("rainbow")
    cmap = default_cmap(np.arange(default_cmap.N))
    cmap[:, -1] = np.linspace(0, 1, default_cmap.N)
    return ListedColormap(cmap)


def add_annotated_points(ax, points, **kwargs):
    """Add annotated points to the given axis.
    The 'points' object must be a pandas.DataFrame with the following columns:
        * name
        * x
        * y
        * horizontalalignment (or ha) - Optional
        * verticalalignment (or va) - Optional
        * fontsize - Optional
        * x_offset - Optional
        * y_offset - Optional
    The given kwargs will be passed to the ``matplotlib.pyplot.annotate()`` function.
    """

    # Check kwargs and set default values
    def set_default(df, attr, alias=None, val=0):
        cols = df.columns.values
        name = attr if attr in cols else None
        if name is None:
            if isinstance(alias, str) and alias in cols:
                name = alias
            elif isinstance(alias, Iterable):
                for i in alias:
                    if i in cols:
                        name = i
                        break
            else:
                raise TypeError("The 'alias' argument must be string or iterable")
        if name is None:
            df[name] = val

        return name

    assert isinstance(points, pd.DataFrame)
    pts = points.copy()
    ha = set_default(pts, "ha", "horizontalalignment", val=0)
    va = set_default(pts, "va", "verticalalignment", val=0)
    fontsize = set_default(pts, "fontsize", val=12)
    x_offset = set_default(pts, "x_offset", val=0)
    y_offset = set_default(pts, "y_offset", val=0)

    arrowprops = kwargs.pop("arrowprops", dict(facecolor="black", shrink=0.05))
    bbox = kwargs.pop("bbox", dict(facecolor="sandybrown", alpha=0.5, boxstyle="round"))

    # Create a matplotlib transform object for the Geodetic coordinate system
    geodetic_transform = ccrs.Geodetic()._as_mpl_transform(ax)

    # Add annotations
    if pts is not None:
        for row in pts.iterrows():
            ax.annotate(
                row["name"],
                xy=(row["x"], row["y"]),
                xytext=(row["x"] + row[x_offset], row["y"] + row[y_offset]),
                arrowprops=arrowprops,
                xycoords=geodetic_transform,
                fontsize=row[fontsize],
                ha=row[ha],
                va=row[va],
                bbox=bbox,
                **kwargs
            )


def plot_segments(
    segments,
    var="velocity",
    vmin=None,
    vmax=None,
    ax=None,
    show=True,
    cmap=None,
    annotations=None,
    background=False,
    zoom=14,
    **kwargs
):
    """Plot segments with background and annotations"""

    # Define CMAP
    if cmap is None:
        cmap = create_cmap()

    # Create the figure instance
    if ax is None:
        fig = plt.figure()

        # Create a GeoAxes in the tile's projection.
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.OSGB())

    # Limit the extent of the map to a small longitude/latitude range
    ax.set_extent(segments.extent, crs=ccrs.OSGB())

    if background is True:
        # Create a terrain background instance
        background = cimgt.GoogleTiles(style="satellite")

    if background is not False:
        # Add the background data
        ax.add_image(background, zoom)

    # Add segments
    segments = segments.make_segments()
    if vmin is not None or vmax is not None:
        segments[var].clip(vmin, vmax, inplace=True)

    # ax.plot(x, y, 'k.', markersize=2, transform=ccrs.OSGB())
    segments.plot(var, ax=ax)

    # Add annotations
    if annotations is not None:
        add_annotated_points(ax, annotations, **kwargs)

    if show is True:
        plt.show()
