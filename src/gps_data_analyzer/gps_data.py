import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString

from .utils import haversine
from . import io


DEFAULT_TIME_FORMAT = "%Y/%m/%d-%H:%M:%S"


def _convert_time(series, format=DEFAULT_TIME_FORMAT):
    """Convert a ``pandas.Series`` containing timestamps as strings.

    Args:
        series (pandas.Series): The timestamps given as strings.
        format (str, optional): The format used for conversion.

    Returns:
        ``pandas.Series``: The timestamps converted in ``pandas.datetime``.
    """
    return pd.to_datetime(series, format=format)


class _GpsBase(object):
    """Class to wrap a ``geopandas.GeoDataFrame`` and format it in order to store GPS
    points.

    Attributes:
        _default_input_crs (int): The default EPSG code of input data (only used when
            ``input_crs`` is ``None``).
        _default_x_col (str): The default column name of input data that contains X
            coordinates (only used when ``x_col`` is ``None``).
        _default_y_col (str): The default column name of input data that contains Y
            coordinates (only used when ``y_col`` is ``None``).
        _default_z_col (str): The default column name of input data that contains Z
            coordinates (only used when ``z_col`` is ``None`` and ``_has_z`` is
            ``True``).
        _default_time_col (str): The default column name of input data that contains
            timestamps (only used when ``time_col`` is ``None`` and ``_has_time`` is
            ``True``).
        _has_z (bool): Indicate whether the Z coordinate must be considered or not.
        _has_time (bool): Indicate whether the timestamps must be considered or not.
        _use_haversine (bool): Indicate whether the distance computations must use the
            Haversine formula or not.
        datetime_format (str): The format used to convert strings into timestamps.

    Args:
        df (``pandas.DataFrame``): The data that will be formatted and stored.
        input_crs (int): The EPSG code of the input data.
        local_crs (int): The EPSG code of the local projection to which the data will
            be transformed.
        keep_cols (:obj:`list` of :obj:`str`): The columns that should not be discarded.
        x_col (str): The name of the column that contains X coordinates.
        y_col (str): The name of the column that contains Y coordinates.
        z_col (str): The name of the column that contains Z coordinates.
        time_col (str): The name of the column that contains timestamps.
    """

    _default_input_crs = 4326
    _default_x_col = "x"
    _default_y_col = "y"
    _default_z_col = "z"
    _default_time_col = "datetime"
    _has_z = True
    _has_time = False
    _use_haversine = True

    datetime_format = DEFAULT_TIME_FORMAT

    def __init__(
        self,
        df,
        input_crs=None,
        local_crs=None,
        keep_cols=None,
        x_col=None,
        y_col=None,
        z_col=None,
        time_col=None,
    ):

        input_crs = input_crs if input_crs is not None else self._default_input_crs
        local_crs = local_crs if local_crs is not None else input_crs
        x_col = x_col if x_col is not None else self._default_x_col
        y_col = y_col if y_col is not None else self._default_y_col
        z_col = z_col if z_col is not None else self._default_z_col
        time_col = time_col if time_col is not None else self._default_time_col

        # Format data
        gdf = self._format_data(
            df,
            input_crs,
            local_crs,
            x_col,
            y_col,
            z_col,
            time_col,
            keep_cols=keep_cols,
        )

        # Save data
        self.data = gdf
        self.crs = self.data.crs

        if self._has_time:
            self._normalize_data()

    @property
    def x(self):
        """``pandas.Series``: Get X coordinates from the geometry."""
        return self.data.geometry.x

    @property
    def y(self):
        """``pandas.Series``: Get Y coordinates from the geometry."""
        return self.data.geometry.y

    @property
    def t(self):
        """Get timestamps if :py:attr:`~_has_time` is ``True`` or the `t` column
        otherwise.

        Returns:
            ``pandas.Series``: Timestamps.
        """
        if self._has_time:
            attr = "datetime"
        else:
            attr = "t"
        return self.__getattr__(attr)

    @property
    def xy(self):
        """``numpy.array``: Array with a (x,y) couple of each point."""
        return np.vstack([self.data.geometry.x, self.data.geometry.y]).T

    def __eq__(self, other):
        """Compare two :py:obj:`~_GpsBase` objects"""
        assert isinstance(other, _GpsBase), (
            "The operator == is only defined for " "'_GpsBase' objects."
        )
        return self.data.equals(other.data)

    def __iter__(self):
        """Return a generator over the rows of the internal
        ``geopandas.GeoDataFrame``"""
        for i in self.data.iterrows():
            yield i

    def __getattr__(self, attr):
        """Apply any unknown attribute to the internal ``geopandas.GeoDataFrame``."""
        return getattr(self.data, attr)

    def __len__(self):
        """Get the length of the internal data.

        Returns:
            int: The length of the internal ``geopandas.GeoDataFrame``
        """
        return len(self.data)

    def _format_data(
        self,
        df,
        input_crs,
        local_crs,
        x_col,
        y_col,
        z_col=None,
        time_col=None,
        keep_cols=None,
    ):
        """Format a ``pandas.DataFrame`` or ``geopandas.GeoDataFrame``.

        Args:
            df (``pandas.DataFrame`` or ``geopandas.GeoDataFrame``): The object to
                format.
            input_crs (int): The EPSG code of the input data.
            local_crs (int or None): The EPSG code of the local projection to which the
                data will be transformed.
            x_col (str): The name of the column containing X or lon coordinates.
            y_col (str): The name of the column containing Y or lat coordinates.
            z_col (str, optional): The name of the column containing Z coordinates.
            time_col (str, optional): The name of the column containing timestamps.
            keep_cols (:obj:`list` of :obj:`str`, optional): The names of the columns
                that should be kept (all others will be discarded).

        Note:
            The column containing timestamps can be either in string format (and should
            thus follow the format given by :py:attr:`~datetime_format`) or in a subtype
            of ``numpy.datetime64``.

        Returns:
            ``geopandas.GeoDataFrame``: The formatted data.
        """
        df = df.copy()

        # Convert time and sort by time
        if self._has_time:
            t_col = df[time_col]
            if not np.issubdtype(df[time_col].dtype, np.datetime64):
                t_col = _convert_time(df[time_col], format=self.datetime_format)
            df["datetime"] = t_col
            df.sort_values("datetime", inplace=True)

        if not isinstance(df, gpd.GeoDataFrame):
            # Drop missing coordinates
            df.dropna(subset=[x_col, y_col], inplace=True)

            # Convert to GeoDataFrame
            df = gpd.GeoDataFrame(
                df, crs=input_crs, geometry=gpd.points_from_xy(df[x_col], df[y_col])
            )
        else:
            input_crs = df.crs.to_epsg()

        # Drop useless columns
        cols = ["geometry"]
        if self._has_z:
            cols.append(z_col)
        if self._has_time:
            cols.append("datetime")
        df = df[cols + ([] if keep_cols is None else keep_cols)]

        # Normalize column names
        if self._has_z:
            df.rename(columns={z_col: "z"}, inplace=True)

        # Reset index
        df.reset_index(drop=True, inplace=True)

        # Project data
        if local_crs is not None and local_crs != input_crs:
            df.to_crs(local_crs, inplace=True)

        return df

    def _normalize_data(self):
        """Conpute time delta between consecutive points (in s)."""
        self.data["dt"] = (
            self.data["datetime"] - self.data["datetime"].shift()
        ).values / pd.Timedelta(1, "s")

        # Conpute distance between consecutive points (in m)
        shifted = self.data.geometry.shift()
        if self.crs.to_epsg() == 4326 and self._use_haversine:
            self.data["dist"] = haversine(self.y, self.x, shifted.y, shifted.x)
        else:
            self.data["dist"] = self.data.distance(shifted)

        # Conpute velocity between consecutive points (in m/s)
        self.data["velocity"] = self.data["dist"] / self.data["dt"]

    def add_attribute(self, attr, name=None):
        """Add a column to the internal ``geopandas.GeoDataFrame``.

        Args:
            attr (``pandas.Series``): The column to add.
            name (str, optional): The name of the new attribute. If not provided, the
                name of the ``pandas.Series`` is used.

        Note:
            The labels of the given ``pandas.Series`` must be the same as the ones of
            the internal ``geopandas.GeoDataFrame``.
        """
        assert isinstance(attr, pd.Series), (
            "The 'attr' argument must be a" "pandas.Series"
        )
        if name is not None:
            self.data[name] = attr
        else:
            self.data[attr.name] = attr

    def segments(self):
        """Build segments from the consecutive points.

        Returns:
            ``geopandas.GeoDataFrame``: A ``geopandas.GeoDataFrame`` containing the
            segments.
        """
        tmp = (
            self.data[["geometry"]]
            .join(self.data[["geometry"]].shift(), rsuffix="_m1")
            .dropna()
        )
        lines = tmp.apply(
            axis=1, func=lambda x: LineString([x.geometry_m1, x.geometry])
        )
        lines.name = "geometry"
        segments = self.data[["dt", "dist", "velocity"]].join(lines, how="right")
        return gpd.GeoDataFrame(segments, crs=self.crs, geometry="geometry")

    def drop_from_mask(self, mask):
        """Drop points contained in the given mask.

        Args:
            mask (:obj:`geopandas.GeoDataFrame`): The mask used to drop internal points.

        Note:
            * The mask must be a :py:obj:`_GpsBase` or ``geopandas.GeoDataFrame``
              object.
            * If the mask has a `radius` column, it will be used and drop all points at
              a distance smaller than the `radius` values.

        Returns:
            int: The number of dropped points.
        """
        mask = mask.copy()

        if isinstance(mask, pd.Series):
            mask = gpd.GeoDataFrame(mask.to_frame("geometry"), crs=mask.crs)

        # Project the mask if needed
        if self.crs is not None:
            mask = mask.to_crs(self.crs)

        # Get the points included in masks
        in_mask_pts = pd.Series(np.zeros(len(self)), dtype=bool)
        for num, i in mask.iterrows():
            in_mask_pts = in_mask_pts | (
                self.geometry.distance(i.geometry) <= i.get("radius", 0))

        # Count the number of points that are going to be dropped
        N = in_mask_pts.sum()

        # Drop points in mask
        self.data.drop(in_mask_pts.loc[in_mask_pts].index, inplace=True)
        self.data.reset_index(drop=True, inplace=True)

        return N


class GpsPoints(_GpsBase):
    """Class to store GPS points with Z coordinates and timestamps."""
    _has_z = True
    _has_time = True


def load_gps_points(path):
    """Load :py:obj:`GpsPoints` from a file.

    Args:
        path (str): The path to the file.

    Returns:
        :py:obj:`GpsPoints`: The data loaded.
    """
    return GpsPoints(io._load(path))


class PoiPoints(_GpsBase):
    """Class to store PoI points with only X and Y coordinates."""
    _has_z = False
    _has_time = False

    def __init__(self, *args, **kwargs):
        if len(args) >= 3:
            keep_cols = args[3]
        elif "keep_cols" in kwargs:
            keep_cols = kwargs["keep_cols"]
        else:
            keep_cols = []
            kwargs["keep_cols"] = keep_cols

        keep_cols.extend(["radius"])
        super().__init__(*args, **kwargs)


def load_poi_points(path):
    """Load :py:obj:`PoiPoints` from a file.

    Args:
        path (str): The path to the file.

    Returns:
        :py:obj:`PoiPoints`: The data loaded.
    """
    return PoiPoints(io._load(path))
