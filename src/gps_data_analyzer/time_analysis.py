import logging

import numpy as np
import pandas as pd
from scipy import spatial


def compute_rest_time(gps_data, radius):
    """Compute the duration during which the track stays in a given radius of each
    point.

    Args:
        gps_data (:py:class:`~gps_data_analyzer.gps_data.PoiPoints`): The data used for
            computation.
        radius (float): The radius in which the rest time is computed around each point.

    Returns:
        ``pandas.Series``: The rest time around each point.
    """

    # TODO: need optimization and cleaning.

    def _t_inter(current, i1, i2, max_radius, geom="geometry", t="datetime"):
        d_i1 = i1[geom].distance(current[geom])
        d_i2 = i2[geom].distance(current[geom])
        t_i1 = i1[t]
        t_i2 = i2[t]
        dt = max(t_i1, t_i2) - min(t_i1, t_i2)
        dd = abs(d_i1 - d_i2)
        if dd == 0:
            return pd.Timedelta(0)
        else:
            return min(1.0, abs(d_i1 - max_radius) / dd) * dt

    def _process_one_pt(num, points, max_radius, logger=logging):
        logger.debug("{}: {}".format(num, points))
        data = gps_data.data
        pts = np.array(points)
        pts.sort()
        pos_i = np.argwhere(pts == num)[0][0]
        diff_not_one = (pts[1:] - pts[:-1]) != 1

        current = data.loc[num]

        # TODO: make a function for inf and sup parts since the only difference is the
        # order of diff_not_one and the limits for label_inf_m1 and label_sup_p1

        # Inf part
        if len(diff_not_one[:pos_i]) > 0:
            diff_not_one[0] = True
            pos_skip_inf = pos_i - np.argmax(np.flip(diff_not_one[:pos_i]))
            label_inf = pts[pos_skip_inf]
            label_inf_m1 = max(0, pts[pos_skip_inf] - 1)
            inf = data.loc[label_inf]
            inf_m1 = data.loc[label_inf_m1]
            dt_inf = current["datetime"] - inf["datetime"]
            t_inf_inter = dt_inf + _t_inter(current, inf, inf_m1, max_radius)
            logger.debug("data:\n{}".format(data.loc[[num, label_inf, label_inf_m1]]))
            logger.debug(
                "distances = {}".format(
                    data.loc[[label_inf, label_inf_m1], "geometry"].distance(
                        current["geometry"]
                    )
                )
            )
        else:
            t_inf_inter = pd.Timedelta(0)

        # Sup part
        if len(diff_not_one[pos_i:]) > 0:
            diff_not_one[-1] = True
            pos_skip_sup = pos_i + np.argmax(diff_not_one[pos_i:])
            label_sup = pts[pos_skip_sup]
            label_sup_p1 = min(data.index.max(), pts[pos_skip_sup] + 1)
            sup = data.loc[label_sup]
            sup_p1 = data.loc[label_sup_p1]
            dt_sup = sup["datetime"] - current["datetime"]
            t_sup_inter = dt_sup + _t_inter(current, sup, sup_p1, max_radius)
            logger.debug("data:\n {}".format(data.loc[[num, label_sup, label_sup_p1]]))
            logger.debug(
                "distances = {}".format(
                    data.loc[[label_sup, label_sup_p1], "geometry"].distance(
                        current["geometry"]
                    )
                )
            )
        else:
            t_sup_inter = pd.Timedelta(0)

        logger.debug("t_inf_inter = {}".format(t_inf_inter))
        logger.debug("t_sup_inter = {}".format(t_sup_inter))

        return t_inf_inter, t_sup_inter

    # Get the closest points of each points
    points = np.c_[gps_data.x.ravel(), gps_data.y.ravel()]
    tree = spatial.KDTree(points)
    points = tree.data
    in_radius_pts = tree.query_ball_point(points, radius)

    # Get the times when the track leave the circle with radius = radius
    t_min = []
    t_max = []
    for num, i in enumerate(in_radius_pts):
        t1, t2 = _process_one_pt(num, i, radius)
        t_min.append(t1)
        t_max.append(t2)

    times = pd.DataFrame({"dt_min": t_min, "dt_max": t_max}, index=gps_data.index)

    # Compute total time
    duration = times["dt_min"] + times["dt_max"]

    # Convert time in seconds
    return duration.apply(pd.Timedelta.total_seconds)
