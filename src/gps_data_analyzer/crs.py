"""
Projections and grids

This file is extracted from the Salem package
https://github.com/fmaussion/salem
This file is thus under 3-clause BSD licence.

Some updates were perfomed for simplification purpose.
"""

# Builtins
import warnings
from distutils.version import LooseVersion

# External libs
import cartopy
import cartopy.crs as ccrs
import pyproj

try:
    from osgeo import osr
    has_gdal = True
except ImportError:  # pragma: no cover
    has_gdal = False


def check_crs(crs, raise_on_error=False):  # pragma: no cover
    """Checks if the crs represents a valid grid, projection or ESPG string.
    Examples
    --------
    >>> p = check_crs('epsg:26915 +units=m')
    >>> p.srs
    'epsg:26915 +units=m'
    >>> p = check_crs('wrong')
    >>> p is None
    True
    Returns
    -------
    A valid crs if possible, otherwise None
    """

    err1, err2 = None, None

    if isinstance(crs, pyproj.Proj):
        out = crs
    elif isinstance(crs, pyproj.crs.CRS):
        out = pyproj.Proj(crs.to_wkt(), preserve_units=True)
    elif isinstance(crs, dict) or isinstance(crs, str):
        if isinstance(crs, str):
            # quick fix for https://github.com/pyproj4/pyproj/issues/345
            crs = crs.replace(' ', '').replace('+', ' +')

        # A series of try-catch to handle the (too) many changes in pyproj
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            warnings.filterwarnings('ignore', category=FutureWarning)
            try:
                out = pyproj.Proj(crs, preserve_units=True)
            except RuntimeError as e:
                err1 = str(e)
                try:
                    out = pyproj.Proj(init=crs, preserve_units=True)
                except RuntimeError as e:
                    err2 = str(e)
                    out = None
    else:
        out = None

    if raise_on_error and out is None:
        msg = ('salem could not properly parse the provided coordinate '
               'reference system (crs). This could be due to errors in your '
               'data, in PyProj, or with salem itself. If this occurs '
               'unexpectedly, report an issue to https://github.com/fmaussion/'
               'salem/issues. Full log: \n'
               'crs: {} ; \n'.format(crs))
        if err1 is not None:
            msg += 'Output of `pyproj.Proj(crs, preserve_units=True)`: {} ; \n'
            msg = msg.format(err1)
        if err2 is not None:
            msg += 'Output of `pyproj.Proj(init=crs, preserve_units=True)`: {}'
            msg = msg.format(err2)
        raise ValueError(msg)

    return out


def proj_is_latlong(proj):  # pragma: no cover
    """Shortcut function because of deprecation."""

    try:
        return proj.is_latlong()
    except AttributeError:
        return proj.crs.is_geographic


def proj_to_cartopy(proj):  # pragma: no cover
    """Converts a pyproj.Proj to a cartopy.crs.Projection
    Parameters
    ----------
    proj: pyproj.Proj
        the projection to convert
    Returns
    -------
    a cartopy.crs.Projection object
    """
    if proj is None:
        return None

    proj = check_crs(proj)

    if proj_is_latlong(proj):
        return ccrs.PlateCarree()

    srs = proj.srs
    if has_gdal:
        # this is more robust, as srs could be anything (espg, etc.)
        s1 = osr.SpatialReference()
        s1.ImportFromProj4(proj.srs)
        if s1.ExportToProj4():
            srs = s1.ExportToProj4()

    km_proj = {'lon_0': 'central_longitude',
               'lat_0': 'central_latitude',
               'x_0': 'false_easting',
               'y_0': 'false_northing',
               'lat_ts': 'latitude_true_scale',
               'o_lon_p': 'central_rotated_longitude',
               'o_lat_p': 'pole_latitude',
               'k': 'scale_factor',
               'zone': 'zone',
               }
    km_globe = {'a': 'semimajor_axis',
                'b': 'semiminor_axis',
                }
    km_std = {'lat_1': 'lat_1',
              'lat_2': 'lat_2',
              }
    kw_proj = dict()
    kw_globe = dict()
    kw_std = dict()
    for s in srs.split('+'):
        s = s.split('=')
        if len(s) != 2:
            continue
        k = s[0].strip()
        v = s[1].strip()
        try:
            v = float(v)
        except Exception:
            pass
        if k == 'proj':
            if v == 'tmerc':
                cl = ccrs.TransverseMercator
            if v == 'lcc':
                cl = ccrs.LambertConformal
            if v == 'merc':
                cl = ccrs.Mercator
            if v == 'utm':
                cl = ccrs.UTM
            if v == 'stere':
                cl = ccrs.Stereographic
            if v == 'ob_tran':
                cl = ccrs.RotatedPole
        if k in km_proj:
            kw_proj[km_proj[k]] = v
        if k in km_globe:
            kw_globe[km_globe[k]] = v
        if k in km_std:
            kw_std[km_std[k]] = v

    globe = None
    if kw_globe:
        globe = ccrs.Globe(ellipse='sphere', **kw_globe)
    if kw_std:
        kw_proj['standard_parallels'] = (kw_std['lat_1'], kw_std['lat_2'])

    # mercatoooor
    if cl.__name__ == 'Mercator':
        kw_proj.pop('false_easting', None)
        kw_proj.pop('false_northing', None)
        if LooseVersion(cartopy.__version__) < LooseVersion('0.15'):
            kw_proj.pop('latitude_true_scale', None)
    elif cl.__name__ == 'Stereographic':
        kw_proj.pop('scale_factor', None)
        if 'latitude_true_scale' in kw_proj:
            kw_proj['true_scale_latitude'] = kw_proj['latitude_true_scale']
            kw_proj.pop('latitude_true_scale', None)
    elif cl.__name__ == 'RotatedPole':
        if 'central_longitude' in kw_proj:
            kw_proj['pole_longitude'] = kw_proj['central_longitude'] - 180
            kw_proj.pop('central_longitude', None)
    else:
        kw_proj.pop('latitude_true_scale', None)

    return cl(globe=globe, **kw_proj)
