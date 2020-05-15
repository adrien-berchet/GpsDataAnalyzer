![CI](https://github.com/adrien-berchet/GpsDataAnalyzer/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/adrien-berchet/GpsDataAnalyzer/branch/master/graph/badge.svg)](https://codecov.io/gh/adrien-berchet/GpsDataAnalyzer)
[![Documentation Status](https://readthedocs.org/projects/gpsdataanalyzer/badge/?version=latest)](https://gpsdataanalyzer.readthedocs.io/en/latest/?badge=latest)


# Welcome to GpsDataAnalyzer

This package contains some helpers I use in other projects to handle GPS data.
It is currently designed to suit my own needs but it might be extended when I need new
features. Of course, comments, issues, feature requests and PR are welcome.

The documentation of this package is accessible online here:
* stable version: https://gpsdataanalyzer.readthedocs.io/en/stable/
* latest version: https://gpsdataanalyzer.readthedocs.io/en/latest/


## Installation

This package can be installed using pip:
```bash
pip install GpsDataAnalyzer
```

## Main features

The main features consists in:
* format GPS data according to their use (e.g. tracks or points of interest).
* compute some properties of these GPS data (e.g. the time spent in an area around
each point).
* compute heatmaps.
* plot figures with background, data and annotations.

## Caveats

Please note that this is a project I work on in my spare time, as such there might be
errors in the implementations and sub-optimal performance.
