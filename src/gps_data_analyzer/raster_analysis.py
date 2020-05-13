import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial

from .plot_utils import add_annotated_points
from .plot_utils import create_transparent_cmap
from .plot_utils import setup_axis


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

    def __getitem__(self, key):
        return [self.xmin, self.xmax, self.ymin, self.ymax][key]

    def mesh(self, mesh_size=None, x_size=None, y_size=None, nx=None, ny=None):
        # Check arguments
        if nx is None and ny is None:
            err_msg = (
                "Either 'mesh_size' or both 'x_size' and 'y_size' must be not None"
            )
            if mesh_size is None:
                if x_size is None or y_size is None:
                    raise ValueError(err_msg)
            else:
                if x_size is not None or y_size is not None:
                    raise ValueError(err_msg)

                x_size = y_size = mesh_size

            if x_size <= 0 or y_size <= 0:
                raise ValueError("The mesh size must be > 0")

            nx = complex(0, int(np.round((self.xmax - self.xmin) / x_size)))
            ny = complex(0, int(np.round((self.ymax - self.ymin) / y_size)))
        else:
            if nx is None or ny is None:
                raise ValueError("Both 'nx' and 'ny' must be not None")
            if mesh_size is not None or x_size is not None or y_size is not None:
                raise ValueError(
                    "Either both 'nx' and 'ny' OR 'mesh_size' OR both 'x_size' and "
                    "'y_size' must be not None"
                )
            if nx <= 0 or ny <= 0:
                raise ValueError("Both 'nx' and 'ny' must be > 0")

            nx = complex(0, nx)
            ny = complex(0, ny)

        # Generate mesh (use complex numbers to include the last value)
        X, Y = np.mgrid[self.xmin : self.xmax : nx, self.ymin : self.ymax : ny]

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
        zoom=None,
        proj=None,
        annotation_kwargs=None,
    ):
        """Plot points with background and annotations"""

        # Setup axis
        fig, ax = setup_axis(
            ax=ax,
            extent=self.extent,
            projection=proj,
            background=background,
            zoom=zoom
        )

        # Define CMAP
        if cmap is None:
            cmap = create_transparent_cmap()

        # Add raster
        ax.imshow(
            self.values,
            cmap=cmap,
            extent=self.extent,
            origin="upper",
            transform=proj,
            zorder=10,
        )

        # Add annotations
        if annotations is not None:
            if annotation_kwargs is None:
                annotation_kwargs = {}
            add_annotated_points(ax, annotations, **annotation_kwargs)

        if show is True:
            plt.show()
        else:
            return fig, ax


def heatmap(
    gps_data,
    mesh_size=None,
    x_size=None,
    y_size=None,
    nx=None,
    ny=None,
    border=0,
    kernel_size=None,
    kernel_cut=4.0,
    weight_col=None,
    normalize=True,
):
    # Check arguments
    if kernel_size is not None and kernel_size <= 0:
        raise ValueError("The 'kernel_size' argument must be > 0")
    if kernel_cut is not None and kernel_cut <= 0:
        raise ValueError("The 'kernel_cut' argument must be > 0")

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
    X, Y = extent.mesh(mesh_size=mesh_size, x_size=x_size, y_size=y_size, nx=nx, ny=ny)
    positions = np.vstack([X.ravel(), Y.ravel()])

    # Compute KDTree
    tree = spatial.KDTree(positions.T)

    # Init sigma if not given
    if kernel_size is None:
        if X.size >= 4:
            dx = X[1, 0] - X[0, 0]
        else:
            dx = 0
        if Y.size >= 4:
            dy = Y[0, 1] - Y[0, 0]
        else:
            dy = 0
        mesh_size = max(dx, dy)
        if mesh_size > 0:
            kernel_size = 2.0 * mesh_size
        else:
            kernel_size = 1

    kde = np.zeros(len(tree.data))
    for num, (_x, _y, _w) in enumerate(zip(x, y, weight)):  # TODO: optimize this loop
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
