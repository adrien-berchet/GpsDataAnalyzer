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
    _default_x_col = "lon"
    _default_y_col = "lat"
    _default_z_col = "alt"
    _default_time_col = "time"
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

        if not isinstance(df, gpd.GeoDataFrame):
            self._format_data(
                df,
                input_crs,
                local_crs,
                x_col,
                y_col,
                z_col,
                time_col,
                keep_cols=keep_cols,
            )
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
        # Convert time and sort by time
        if self._has_time:
            df = df.copy()
            df["datetime"] = _convert_time(df[time_col], format=self.datetime_format)
            df.sort_values("datetime", inplace=True)

        # Drop useless columns
        cols = [x_col, y_col]
        if self._has_z:
            cols.append(z_col)
        if self._has_time:
            cols.append("datetime")
        df = df[cols + ([] if keep_cols is None else keep_cols)].copy()

        # Drop missing coordinates
        df.dropna(subset=cols, inplace=True)

        # Convert to GeoDataFrame
        gdf = gpd.GeoDataFrame(
            df, crs=input_crs, geometry=gpd.points_from_xy(df[x_col], df[y_col])
        )

        # Normalize column names
        gdf.drop(columns=[x_col, y_col], inplace=True)
        # gdf.rename(columns={x_col: "x", y_col: "y"}, inplace=True)
        if self._has_z:
            gdf.rename(columns={z_col: "z"}, inplace=True)

        # Reset index
        gdf.reset_index(drop=True, inplace=True)

        # Project data
        if local_crs is not None and local_crs != input_crs:
            gdf.to_crs(local_crs, inplace=True)

        # Save data
        self.data = gdf
        self.crs = gdf.crs

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
    _has_time = True


def load_gps_points(path):
    return GpsPoints(io.load(path))


class PoI(_GpsBase):
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


def load_poi(path):
    return PoI(io.load(path))


def add_masks(gps_data, filename, crs=4326, sep=";", decimal=".", **kwargs):
    def _get_points_in_masks(masks):
        # Get the points included in masks
        in_mask_pts = pd.Series(np.zeros(len(data)), dtype=bool)
        for num, i in masks.iterrows():
            in_mask_pts = in_mask_pts | (data.geometry.distance(i.geometry) <= i.radius)

        return in_mask_pts

    data = gps_data._data

    # Load masks
    masks = pd.read_csv(filename, sep=sep, decimal=decimal, **kwargs)
    masks = gpd.GeoDataFrame(
        masks, crs=crs, geometry=gpd.points_from_xy(masks["lon"], masks["lat"])
    )
    if gps_data.local_crs is not None:
        masks.to_crs(gps_data.local_crs, inplace=True)

    # Drop points in masks
    pts_in_mask = _get_points_in_masks(masks)
    data.drop(pts_in_mask.loc[pts_in_mask].index, inplace=True)
    data.reset_index(drop=True, inplace=True)


def add_pois(gps_data, filename, sep=";", decimal=".", **kwargs):
    # Load PoIs
    gps_data.pois = pd.read_csv(filename, sep=sep, decimal=decimal, **kwargs)
