import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString

from . import io


DEFAULT_TIME_FORMAT = "%Y/%m/%d-%H:%M:%S"


def _convert_time(series, format=DEFAULT_TIME_FORMAT):
    # Convert time
    return pd.to_datetime(series, format=format)


class _GpsBase(object):
    """Class to store GpsData"""

    _default_input_crs = 4326
    _default_x_col = "x"
    _default_y_col = "y"
    _default_z_col = "z"
    _default_time_col = "datetime"
    _has_z = True
    _has_time = False

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
        return self.data.geometry.x

    @property
    def y(self):
        return self.data.geometry.y

    @property
    def t(self):
        if self._has_time:
            attr = "datetime"
        else:
            attr = "t"
        return self.__getattr__(attr)

    @property
    def xy(self):
        return np.vstack([self.data.geometry.x, self.data.geometry.y]).T

    def __eq__(self, other):
        """Compare two _GpsBase objects"""
        assert isinstance(other, _GpsBase), (
            "The operator == is only defined for " "'_GpsBase' objects."
        )
        return self.data.equals(other._data)

    def __iter__(self):
        """Return an generator over the rows of the internal geopandas.GeoDataFrame"""
        for i in self.data.iterrows():
            yield i

    def __getattr__(self, attr):
        """Return the column as if it was an attribute"""
        return getattr(self.data, attr)
        # if isinstance(attr, str):
        #     not_found = set([attr]) - set(self.data.columns)
        #     is_str = True
        # elif isinstance(attr, Iterable):
        #     not_found = set(attr) - set(self.data.columns)
        #     is_str = False
        # else:
        #     raise TypeError(
        #         "The 'attr' argument must be a string or a list of strings")

        # if not_found:
        #     msg = "The attribute{} '{}' {} not found".format(
        #         "" if is_str else "s",
        #         attr,
        #         "was" if is_str else "were")
        #     raise ValueError(msg)
        # else:
        #     return self.data[attr]

    def __len__(self):
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
        # Conpute time delta between consecutive points (in s)
        self.data["dt"] = (
            self.data["datetime"] - self.data["datetime"].shift()
        ).values / pd.Timedelta(1, "s")

        # Conpute distance between consecutive points (in m)
        self.data["dist"] = self.data.distance(self.data.geometry.shift())

        # Conpute velocity between consecutive points (in m/s)
        self.data["velocity"] = self.data["dist"] / self.data["dt"]

    def add_attribute(self, attr, name=None):
        assert isinstance(attr, pd.Series), (
            "The 'attr' argument must be a" "pandas.Series"
        )
        if name is not None:
            self.data[name] = attr
        else:
            self.data[attr.name] = attr

    def segments(self):
        """Build segments from the consecutive points"""
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
        mask = mask.copy()

        if isinstance(mask, pd.Series):
            mask = gpd.GeoDataFrame(mask.to_frame("geometry"), crs=mask.crs)

        # Project the mask if needed
        if self.crs is not None:
            mask = mask.to_crs(self.crs)

        # Get the points included in masks
        in_mask_pts = pd.Series(np.zeros(len(self)), dtype=bool)
        for num, i in mask.iterrows():
            in_mask_pts = in_mask_pts | (self.geometry.distance(i.geometry) <= i.radius)

        # Drop points in mask
        self.data.drop(in_mask_pts.loc[in_mask_pts].index, inplace=True)
        self.data.reset_index(drop=True, inplace=True)


class GpsPoints(_GpsBase):
    """
    Class to wrap a geopandas.GeoDataFrame and format it in order to store GPS
    points
    """
    _has_time = True


def load_gps_points(path):
    return GpsPoints(io._load(path))


class PoiPoints(_GpsBase):
    """
    Class to wrap a geopandas.GeoDataFrame and format it in order to store Point of
    Interest points
    """
    _has_time = False
    _has_z = False

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
    return PoiPoints(io._load(path))
