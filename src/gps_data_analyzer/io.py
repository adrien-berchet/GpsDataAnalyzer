import pandas as pd
import geopandas as gpd


def save(obj, path, mode="w", **kwargs):
    tmp = obj.data.copy()

    # Convert datetime to string
    if obj._has_time:
        tmp["datetime"] = tmp["datetime"].apply(pd.Timestamp.isoformat)

    # Save to GeoJSON
    tmp.to_file(path, driver="GPKG", encoding="utf-8")

    # TODO: Fiona>=0.19 will be able to store metadata in GPKG files. It would be nice
    # to store the data type in metadata so the load() function can now which class it
    # should call.


def _load(path):
    # Load data
    data = gpd.read_file(path, driver="GPKG")

    # Convert time columns
    if "datetime" in data.columns:
        data["datetime"] = pd.to_datetime(data["datetime"])

    # If everything could be imported properly, the new object is returned
    return data
