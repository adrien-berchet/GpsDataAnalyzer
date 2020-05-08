import os
import tempfile
import zipfile

import pandas as pd
import geopandas as gpd


def save(obj, path):
    # Create directory if it does not exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp = obj.data.copy()

        # Convert datetime to string
        if obj._has_time:
            tmp["datetime"] = tmp["datetime"].apply(pd.Timestamp.isoformat)

        # Save to GeoJSON
        data_file = os.path.join(tmpdirname, "data.geojson")
        tmp.to_file(data_file, driver="GeoJSON", encoding="utf-8")

        # Save metadata
        # metadata = {
        #     "local_crs": self.local_crs,
        #     "x_col": self.x_col,
        #     "y_col": self.y_col,
        # }
        # metadata_file = os.path.join(tmpdirname, "metadata.json")
        # with open(metadata_file, mode="w") as f:
        #     json.dump(metadata, f)

        # Zip the files to the destination
        zip_file = zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED)
        with zip_file:
            zip_file.write(data_file, arcname=os.path.basename(data_file))
            # zip_file.write(metadata_file, arcname=os.path.basename(metadata_file))


def load(path):
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Extract the Zip file
        zip_file = zipfile.ZipFile(path, 'r', compression=zipfile.ZIP_DEFLATED)
        zip_file.extractall(path=tmpdirname)

        # Load data
        data = gpd.read_file(
            os.path.join(tmpdirname, "data.geojson"), driver="GeoJSON")

        # Convert time columns
        if "datetime" in data.columns:
            data["datetime"] = pd.to_datetime(data["datetime"])
            # data["duration"] = data["duration_s"].apply(
            #     pd.Timedelta, args=["S"])

        # Load metadata
        # with open(os.path.join(tmpdirname, "metadata.json"), mode="r") as f:
        #     metadata = json.load(f)

        # local_crs = metadata.get("local_crs", None)

    # If everything could be imported properly, the new object is returned
    return data
