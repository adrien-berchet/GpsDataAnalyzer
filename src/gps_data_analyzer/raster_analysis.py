import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial

from .plot_utils import add_annotated_points
from .plot_utils import create_transparent_cmap
from .plot_utils import setup_axis


class Extent(object):
    """Class to manage extent of a Raster data and generate a mesh.

    Attributes:
        inner_xmin (float): The minimum X coordinate not considering the border.
        inner_xmax (float): The maximum X coordinate not considering the border.
        inner_ymin (float): The minimum Y coordinate not considering the border.
        inner_ymax (float): The maximum Y coordinate not considering the border.
        xmin (float): The minimum X coordinate considering the border.
        xmax (float): The maximum X coordinate considering the border.
        ymin (float): The minimum Y coordinate considering the border.
        ymax (float): The maximum Y coordinate considering the border.
        border (float): The extra border to add around the data values.

    Args:
        xmin (float): The minimum X coordinate.
        xmax (float): The maximum X coordinate.
        ymin (float): The minimum Y coordinate.
        ymax (float): The maximum Y coordinate.
        border (float, optional): The extra border to add around the data values.
    """

    def __init__(self, xmin, xmax, ymin, ymax, border=0):
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
        """Define new extent and recalculate the extent according to it.

        Args:
            border (float): The extra border to add around the data values.
        """
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
        """Create a mesh in the current extent.

        Args:
            mesh_size (float, optional): The space between two pixels of the heatmap.
            x_size (float, optional): The space between two pixels of the heatmap along
                the X axis.
            y_size (float, optional): The space between two pixels of the heatmap along
                the Y axis.
            nx (int, optional): The number of pixels of the heatmap along the X axis.
            ny (int, optional): The number of pixels of the heatmap along the Y axis.

        Returns:
            (``numpy.array``, ``numpy.array``): X and Y coordinates of the mesh nodes.
        """

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
    """Class to manage Raster data.

    Attributes:
        X (int): The default EPSG code of input data (only used when
            ``input_crs`` is ``None``).
        Y (str): The default column name of input data that contains X
            coordinates (only used when ``x_col`` is ``None``).
        values (str): The default column name of input data that contains Y
            coordinates (only used when ``y_col`` is ``None``).
        extent (str): The default column name of input data that contains Z
            coordinates (only used when ``z_col`` is ``None`` and ``_has_z`` is
            ``True``).

    Args:
        X (``numpy.array``): The X coordinates.
        Y (``numpy.array``): The Y coordinates.
        values (``numpy.array``): The values at each point (X, Y).
        extent (:py:class:`~gps_data_analyzer.raster_analysis.Extent`): The extent of
            the raster data.
    """

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
        projection=None,
        cmap=None,
        background=False,
        zoom=None,
        annotations=None,
        annotation_kwargs=None,
        **kwargs
    ):
        """Plot raster with background and annotations.

        Args:
            ax (``matplotlib.pyplot.Axes``, optional): The axis object to update.
            show (bool, optional): If true, call :func:`plt.show` else return the figure
                and axis objects.
            projection (:py:class:`cartopy.crs.Projection`, optional): The projection of
                the axis (:py:class:`~cartopy.crs.PlateCarree` by default).
            cmap (:py:class:`matplotlib.colors.Colormap`, optional): The colormap to use
                (a default will be created if not given).
            background (bool or :py:class:`cartopy.io.img_tiles.GoogleWTS`, optional): \
                If true, a default background is added using Google Satellite. If a
                :obj:`~cartopy.io.img_tiles.GoogleWTS` object is given, it is used.
            zoom (int, mandatory if :py:obj:`background` is not :py:obj:`None`): The
                zoom value used to generate the background.
            annotations (:py:class:`~gps_data_analyzer.gps_data.PoiPoints`, optional): \
                The points used to annotate the figure.
            annotation_kwargs (dict, optional): The kwargs passed to
                :func:`~gps_data_analyzer.plot_utils.add_annotated_points`.
            kwargs: The given kwargs will be passed to
                :func:`matplotlib.pyplot.Axes.imshow`.

        Returns:
            :py:class:`~gps_data_analyzer.raster_analysis.Raster` The 2D array
            containing the result.
        """

        # Setup axis
        fig, ax = setup_axis(
            ax=ax,
            extent=self.extent,
            projection=projection,
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
            transform=projection,
            zorder=10,
            **kwargs
        )

        # Add annotations
        if annotations is not None:
            if annotation_kwargs is None:
                annotation_kwargs = {}
            add_annotated_points(ax, annotations, **annotation_kwargs)

        if show is True:
            plt.show()  # pragma: no cover
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
    """Compute heatmap from a set of points using a Gaussian kernel.

    Args:
        gps_data (:py:class:`~gps_data_analyzer.gps_data._GpsBase`): The point set.
        mesh_size (float, optional): The space between two pixels of the heatmap.
        x_size (float, optional): The space between two pixels of the heatmap along the
            X axis.
        y_size (float, optional): The space between two pixels of the heatmap along the
            Y axis.
        nx (int, optional): The number of pixels of the heatmap along the X axis.
        ny (int, optional): The number of pixels of the heatmap along the Y axis.
        border (float, optional): The extra border around the data.
        kernel_size (float, optional): The kernel size used for computation.
        kernel_cut (float, optional): The kernel cut used for computation.
        weight_col (str, optional): The column name used as point weights.
        normalize (bool, optional): Trigger normalization of the result.

    Returns:
        :py:class:`~gps_data_analyzer.raster_analysis.Raster` The 2D array containing
        the result.
    """
    # Check arguments
    if kernel_size is not None and kernel_size <= 0:
        raise ValueError("The 'kernel_size' argument must be > 0")
    if kernel_cut is not None and kernel_cut <= 0:
        raise ValueError("The 'kernel_cut' argument must be > 0")

    # Get coordinates
    x = gps_data.x
    y = gps_data.y
    if weight_col is not None:
        weight = getattr(gps_data, weight_col).values
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

    # Reshape and rotate the result
    heatmap = np.rot90(np.reshape(kde, X.shape))

    return Raster(X, Y, heatmap, extent)
