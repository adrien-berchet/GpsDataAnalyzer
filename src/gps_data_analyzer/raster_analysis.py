import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial

from .plot_utils import add_annotated_points


class Extent(object):
    """docstring for Extent"""

    def __init__(self, xmin, xmax, ymin, ymax, border):
        self.border = border
        self.inner_xmin = xmin
        self.inner_xmax = xmax
        self.inner_ymin = ymin
        self.inner_ymax = ymax
        self.xmin = xmin - border
        self.xmax = xmax + border
        self.ymin = ymin - border
        self.ymax = ymax + border

    def reset_border(self, border):
        # Define new extent
        self.border = border
        self.xmin = self.inner_xmin - border
        self.xmax = self.inner_xmax + border
        self.ymin = self.inner_ymin - border
        self.ymax = self.inner_ymax + border

    def __iter__(self):
        for i in [self.xmin, self.xmax, self.ymin, self.ymax]:
            yield i

    def mesh(self, mesh_size=None, x_size=None, y_size=None):
        # Check arguments
        err_msg = "Either 'mesh_size' or both 'x_size' and 'y_size' must be not None"
        if mesh_size is None:
            if x_size is None or y_size is None:
                raise ValueError(err_msg)
        else:
            if x_size is not None or y_size is not None:
                raise ValueError(err_msg)
            else:
                x_size = y_size = mesh_size

        # Generate mesh (use complex numbers to include the last value)
        nx = complex(0, int(np.round((self.xmax - self.xmin) / x_size)))
        ny = complex(0, int(np.round((self.ymax - self.ymin) / y_size)))
        X, Y = np.mgrid[self.xmin: self.xmax: nx, self.ymin: self.ymax: ny]

        return X, Y


class Raster(object):
    """docstring for Raster"""

    def __init__(self, X, Y, values, extent):
        assert X.size == Y.size == values.size
        self.X = X
        self.Y = Y
        self.values = values
        self.extent = extent

    def plot(
        self,
        ax=None,
        show=True,
        cmap=None,
        annotations=None,
        background=False,
        zoom=14,
        proj=None,
        annotation_kwargs=None,
    ):
        """Plot points with background and annotations"""

        # Define default projection
        if proj is None:
            proj = ccrs.Geodetic()

        # Define CMAP
        if cmap is None:
            cmap = self._create_cmap()

        # Create the figure instance
        if ax is None:
            fig = plt.figure()

            # Create a GeoAxes in the tile's projection.
            ax = fig.add_subplot(1, 1, 1, projection=proj)

        # Limit the extent of the map to a small longitude/latitude range
        ax.set_extent(self.extent, crs=proj)

        if background is True:
            # Create a default terrain background instance
            background = cimgt.GoogleTiles(style="satellite")

        if background is not False:
            # Add the background data
            ax.add_image(background, zoom)

        # Define raster
        raster_pos = np.rot90(self.values)

        # Add raster
        ax.imshow(
            raster_pos,
            cmap=cmap,
            extent=self.extent,
            origin="upper",
            transform=proj,
            zorder=10,
        )

        # Add track markers
        x, y = self._xy()
        ax.plot(x, y, "k.", markersize=2, transform=proj)

        # Add annotations
        if annotations is not None:
            if annotation_kwargs is None:
                annotation_kwargs = {}
            add_annotated_points(ax, annotations, **annotation_kwargs)

        if show is True:
            plt.show()


def heatmap(
    gps_data,
    mesh_size=None,
    x_size=None,
    y_size=None,
    border=0,
    kernel_size=None,
    kernel_cut=4.0,
    weight_col=None,
    normalize=True,
):
    # Get coordinates
    x = gps_data.x
    y = gps_data.y
    if weight_col is not None:
        weight = gps_data.weight_col.values
    else:
        weight = np.ones(len(gps_data))

    # Compute extent
    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()
    extent = Extent(xmin, xmax, ymin, ymax, border)

    # Generate mesh
    X, Y = extent.mesh(mesh_size=mesh_size, x_size=x_size, y_size=y_size)
    positions = np.vstack([X.ravel(), Y.ravel()])

    # Compute KDTree
    tree = spatial.KDTree(positions.T)

    # Init sigma if not given
    if kernel_size is None:
        kernel_size = 2.0 * mesh_size

    kde = np.zeros(len(tree.data))
    for num, _x, _y, _w in zip(x, y, weight):  # TODO: optimize this loop
        # Get the closest points of the current point
        coords_i = [_x, _y]
        in_radius_pts = tree.query_ball_point(coords_i, kernel_cut * kernel_size)

        # Compute distances and divide by the krenel size
        q = (
            np.squeeze(spatial.distance.cdist(tree.data[in_radius_pts], [coords_i]))
            / kernel_size
        )

        # Compute KDE contribution
        res = np.exp(-np.power(q, 2)) / (2.0 * kernel_size)
        kde[in_radius_pts] += res * _w

    # Normalize KDE
    if not normalize:
        kde /= kernel_size * np.sqrt(2 * np.pi)
    else:
        kde -= kde.min()
        kde /= kde.max()

        # Reshape the result
    heatmap = np.reshape(kde, X.shape)

    return Raster(X, Y, heatmap, extent)
