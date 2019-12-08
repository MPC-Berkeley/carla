import numpy as np

def fix_angle(diff_ang):
    while diff_ang > np.pi:
        diff_ang -= 2 * np.pi
    while diff_ang < -np.pi:
        diff_ang += 2 * np.pi
    assert(-np.pi <= diff_ang and diff_ang <= np.pi)
    return diff_ang