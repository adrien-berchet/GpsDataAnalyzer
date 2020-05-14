import numpy as np


def haversine(lat1, lon1, lat2, lon2, to_radians=True, earth_radius=6371e3):
    """Vectorized haversine function

    Slightly modified version: of http://stackoverflow.com/a/29546836/2901002

    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees or in radians)

    All (lat, lon) coordinates must have numeric dtypes and be of equal length.

    Args:
        lat1 (``numpy.array``): The latitude coordinates of the first point.
        lon1 (``numpy.array``): The longitude coordinates of the first point.
        lat2 (``numpy.array``): The latitude coordinates of the second point.
        lon2 (``numpy.array``): The longitude coordinates of the second point.
        to_radians (bool, optional): Convert coordinates in radians.
        earth_radius (float, optional): The earth radius used for computations.

    Returns:
        ``numpy.array``: The distances betwen each couple of points.
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
