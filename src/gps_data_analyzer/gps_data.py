import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj.crs import CRS
from shapely.geometry import LineString

from .utils import haversine
from . import io


DEFAULT_TIME_FORMAT = "%Y/%m/%d-%H:%M:%S"


def _convert_time(series, format: str = DEFAULT_TIME_FORMAT):
    """Convert a ``pandas.Series`` containing timestamps as strings.

    Args:
        series (pandas.Series): The timestamps given as strings.
        format (str, optional): The format used for conversion.

    Returns:
        ``pandas.Series``: The timestamps converted in ``pandas.datetime``.
    """
    return pd.to_datetime(series, format=format)


class _GpsBase(gpd.GeoDataFrame):
    """Class to wrap a ``geopandas.GeoDataFrame`` and format it in order to store GPS
    points.

    Attributes:
        datetime_format (str): The format used to convert strings into timestamps.
        _default_input_crs (int): The default EPSG code of input data (only used when
            ``input_crs`` is ``None``).
        _has_z (bool): Indicate whether the Z coordinate must be considered or not.
        _has_time (bool): Indicate whether the timestamps must be considered or not.
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
        _default_dt_col (str): The default column name to store delta times
            (only used when ``_has_time`` is ``True``).
        _default_dist_col (str): The default column name to store distances
            (only used when ``_has_time`` is ``True``).
        _default_velocity_col (str): The default column name to store velocities
            (only used when ``_has_time`` is ``True``).
        _use_haversine (bool): Indicate whether the distance computations must use the
            Haversine formula or not.

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
        time_sort (bool): Sort data by ascending time (only used when ``_has_z`` is
            True).
    """

    datetime_format = DEFAULT_TIME_FORMAT

    _default_input_crs = 4326
    _has_z = True
    _has_time = False
    _default_x_col = "x"
    _default_y_col = "y"
    _default_z_col = "z"
    _default_time_col = "datetime"
    _default_dt_col = "dt"
    _default_dist_col = "dist"
    _default_velocity_col = "velocity"
    _use_haversine = True

    def __init__(
        self,
        *args,
        local_crs: int = None,
        x_col: str = None,
        y_col: str = None,
        z_col: str = None,
        time_col: str = None,
        time_sort: bool = True,
        **kwargs
    ) -> '_GpsBase':
        kwargs["crs"] = kwargs.get("crs", self._default_input_crs)
        df = gpd.GeoDataFrame(*args, **kwargs)

        x_col = x_col if x_col is not None else self._default_x_col
        y_col = y_col if y_col is not None else self._default_y_col
        z_col = z_col if z_col is not None else self._default_z_col
        time_col = time_col if time_col is not None else self._default_time_col

        # Format data
        self._format_data(
            df,
            kwargs["crs"],
            local_crs,
            x_col,
            y_col,
            z_col,
            time_col,
            time_sort,
        )

        # Compute normalized data
        if self._has_time:
            self._normalize_data(df)

        super(_GpsBase, self).__init__(df, crs=df.crs)

    @property
    def x(self) -> pd.Series:
        """``pandas.Series``: Get X coordinates from the geometry."""
        return self.geometry.x

    @property
    def y(self) -> pd.Series:
        """``pandas.Series``: Get Y coordinates from the geometry."""
        return self.geometry.y

    @property
    def z(self) -> pd.Series:
        """Get Z coordinates if :py:attr:`~_has_z` is ``True`` or the `z` column
        otherwise.

        Returns:
            ``pandas.Series``: Z coordinates.
        """
        if self._has_z:
            return self.geometry.apply(lambda x: x.z)
        else:
            return self._return_attr("z")

    @property
    def t(self) -> pd.Series:
        """Get timestamps if :py:attr:`~_has_time` is ``True`` or the `t` column
        otherwise.

        Returns:
            ``pandas.Series``: Timestamps.
        """
        if self._has_time:
            attr = self._default_time_col
        else:
            attr = "t"
        return self._return_attr(attr)

    @property
    def xy(self) -> np.array:
        """``numpy.array``: Array with a (x,y) couple of each point."""
        return np.vstack([self.geometry.x, self.geometry.y]).T

    def _return_attr(self, attr) -> pd.Series:
        try:
            return self.__getattr__(attr)
        except RecursionError:
            raise AttributeError("This object has no '{}' attribute".format(attr))

    @property
    def base_columns(self) -> list:
        base_cols = ["geometry"]
        if self._has_z:
            base_cols.append(self._default_z_col)
        if self._has_time:
            base_cols.append(self._default_time_col)
        return base_cols

    def copy(self, deep: bool = False):
        """Return a copy of the current object.

        Args:
            deep (bool): Make a deep copy, including a copy of the data and the indices.
                With ``deep=False`` neither the indices nor the data are copied.
        """
        return self.__class__(
            super(_GpsBase, self).copy(deep=deep),
            crs=self.crs,
            time_sort=False)

    @classmethod
    def _format_data(
        cls,
        df: pd.DataFrame,
        input_crs: int,
        local_crs: int,
        x_col: str,
        y_col: str,
        z_col: str = None,
        time_col: str = None,
        time_sort: bool = True,
    ) -> None:
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
        time_sort (bool): Sort data by ascending time (only used when ``_has_z`` is
                True).

        Note:
            The column containing timestamps can be either in string format (and should
            thus follow the format given by :py:attr:`~datetime_format`) or in a subtype
            of ``numpy.datetime64``.

        Returns:
            ``geopandas.GeoDataFrame``: The formatted data.
        """

        # Convert time and sort by time
        if cls._has_time and time_col in df.columns:
            t_col = df[time_col]
            if not np.issubdtype(t_col.dtype, np.datetime64):
                t_col = _convert_time(t_col, format=cls.datetime_format)
            df[cls._default_time_col] = t_col
            if cls._default_time_col != time_col:
                df.drop(columns=[time_col], inplace=True)
            if time_sort:
                df.sort_values(cls._default_time_col, inplace=True)

        # Create geometry from coordinates
        if df._geometry_column_name not in df.columns:
            # Drop missing coordinates
            df.dropna(subset=[x_col, y_col], inplace=True)

            # Set geometry column
            df.geometry = gpd.points_from_xy(
                df[x_col], df[y_col], df[z_col] if cls._has_z else None)

            # Drop useless columns
            dropped_cols = [x_col, y_col]
            if cls._has_z:
                dropped_cols.append(z_col)
            df.drop(columns=dropped_cols, inplace=True)

            # Reset index
            df.reset_index(drop=True, inplace=True)

        # Project data
        if local_crs is not None:
            self_crs = CRS(df.crs)
            local_crs = CRS(local_crs)
            if local_crs != self_crs:
                df.to_crs(local_crs, inplace=True)

    @classmethod
    def _normalize_data(cls, df) -> None:
        """Conpute time delta between consecutive points (in s)."""
        if (
            cls._default_dt_col not in df.columns
            and cls._default_time_col in df.columns
        ):
            df[cls._default_dt_col] = (
                df[cls._default_time_col] - df[cls._default_time_col].shift()
            ).values / pd.Timedelta(1, "s")

        # Conpute distance between consecutive points (in m)
        if cls._default_dist_col not in df.columns:
            shifted = df.geometry.shift()
            # TODO: use crs.is_latlon or something like this?
            if df.crs.to_epsg() == 4326 and cls._use_haversine:
                df[cls._default_dist_col] = haversine(
                    df.geometry.y, df.geometry.x,
                    shifted.geometry.y, shifted.geometry.x)
            else:
                df[cls._default_dist_col] = df.distance(shifted)

        # Conpute velocity between consecutive points (in m/s)
        if (
            cls._default_velocity_col not in df.columns
            and cls._default_dist_col in df.columns
            and cls._default_dt_col in df.columns
        ):
            df[cls._default_velocity_col] = (
                df[cls._default_dist_col] / df[cls._default_dt_col]
            )

    def add_attribute(self, attr: pd.Series, name: str = None) -> None:
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
            self[name] = attr
        else:
            self[attr.name] = attr

    def segments(self) -> gpd.GeoDataFrame:
        """Build segments from the consecutive points.

        Returns:
            ``geopandas.GeoDataFrame``: A ``geopandas.GeoDataFrame`` containing the
            segments.
        """
        tmp = (
            self[["geometry"]]
            .join(self[["geometry"]].shift(), rsuffix="_m1")
            .dropna()
        )
        lines = tmp.apply(
            axis=1, func=lambda x: LineString([x.geometry_m1, x.geometry])
        )
        lines.name = "geometry"
        segments = self[
            [self._default_dt_col, self._default_dist_col, self._default_velocity_col]
        ].join(lines, how="right")
        return gpd.GeoDataFrame(segments, crs=self.crs, geometry="geometry")

    def drop_from_mask(self, mask: gpd.GeoDataFrame) -> int:
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
            mask = mask.to_crs(self.crs, inplace=False)

        # Get the points included in masks
        in_mask_pts = pd.Series(np.zeros(len(self)), dtype=bool)
        for num, i in mask.iterrows():
            in_mask_pts = in_mask_pts | (
                self.geometry.distance(i.geometry) <= i.get("radius", 0))

        # Count the number of points that are going to be dropped
        N = in_mask_pts.sum()

        # Drop points in mask
        self.drop(in_mask_pts.loc[in_mask_pts].index, inplace=True)
        self.reset_index(drop=True, inplace=True)

        return N


class GpsPoints(_GpsBase):
    """Class to store GPS points with Z coordinates and timestamps."""
    _has_z = True
    _has_time = True


def load_gps_points(path: str) -> GpsPoints:
    """Load :py:obj:`GpsPoints` from a file.

    Args:
        path (str): The path to the file.

    Returns:
        :py:obj:`GpsPoints`: The data loaded.
    """
    data = io._load(path)
    return GpsPoints(data)


class PoiPoints(_GpsBase):
    """Class to store PoI points with only X and Y coordinates."""
    _has_z = False
    _has_time = False


def load_poi_points(path: str) -> PoiPoints:
    """Load :py:obj:`PoiPoints` from a file.

    Args:
        path (str): The path to the file.

    Returns:
        :py:obj:`PoiPoints`: The data loaded.
    """
    data = io._load(path)
    return PoiPoints(data)


def concatenate(data_sets: list, crs: int = None) -> GpsPoints:
    """Concatenate several data sets.

    Args:
        data_sets (list): A list of data sets.
        crs (int): The EPSG code to which the data sets will be projected.

    Returns:
        The new data set.
    """
    if len(data_sets) < 1:  # pragma: no cover
        raise ValueError("No data provided in 'data_sets' argument")
    if crs is None:
        for i in data_sets:
            if crs is not None and i.crs != crs:  # pragma: no cover
                raise ValueError(
                    "If all sets do not have the same CRS, use the 'crs' parameter")
            crs = i.crs

    try:
        crs = crs.to_epsg()
    except AttributeError:
        pass

    gdf = GpsPoints(
        pd.concat([i.to_crs(crs) for i in data_sets]), crs=crs, time_sort=False)

    return gdf
