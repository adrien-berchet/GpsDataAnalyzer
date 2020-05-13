import numpy as np


def haversine(lat1, lon1, lat2, lon2, to_radians=True, earth_radius=6371e3):
    """
    Vectorized haversine function

    slightly modified version: of http://stackoverflow.com/a/29546836/2901002

    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees or in radians)

    All (lat, lon) coordinates must have numeric dtypes and be of equal length.

    """
    if to_radians:
        lat1 = np.radians(lat1)
        lon1 = np.radians(lon1)
        lat2 = np.radians(lat2)
        lon2 = np.radians(lon2)

    a = np.power(np.sin((lat2 - lat1) / 2.0), 2) + (
        np.cos(lat1) * np.cos(lat2) * np.power(np.sin((lon2 - lon1) / 2.0), 2)
    )

    return earth_radius * 2.0 * np.arcsin(np.sqrt(a))
