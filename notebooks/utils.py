import numpy as np
from scipy.spatial.transform import Rotation as R
import pickle
import warnings

def get_critical_angles(dlong, decl, roll):
    """
    Calculates the coordinates of the Sun in spacecraft coordinates.
    Routine implicitly assumes the Sun is at infinity, but this is likely a small effect

    Parameters:
    -----------
    dlong: float
        angle from antisolar of boresight
    decl: float
        ecliptic declination of pointing
    roll: float
        roll angle around the boresight

    Returns
    -------
    sign_x : float
        sign of the X coordinate of the Sun
    sun_angle: float
        the angle of the Sun above the sunshade in degrees
    sun_angle_SC: float
        the angle of the Sun in SC Y in degrees.
    """

    rlong = R.from_euler("x", -dlong, degrees=True)
    rlat = R.from_euler("y", -decl, degrees=True)
    rroll = R.from_euler("z", -roll, degrees=True)
    rfinal = rroll * rlat * rlong

    vsun = [0, 0, -1]
    vsunf = np.atleast_2d(rfinal.apply(vsun))

    return (
        np.sign(vsunf[:, 0]).astype(int),
        57.296 * np.arcsin(vsunf[:, 2]),
        57.296 * np.abs(np.arcsin(vsunf[:, 1])),
    )


def check_range(dlong, decl, roll, ylim=15, ndays=14):
    """
    Steps through the days of the sector, gives Z and Y angles and any alarms that might be set off.

    Parameters:
    -----------
    dlong: float
        the difference in ecliptic longitude from antisolar in the middle of the sector (0 for N-S orientations, but non-zero for ecliptic pointings),
    decl: float
        the ecliptic declination of the pointing
    roll: float
        the roll about S/C boresight (Z)
    ylim: float
        the antisolar angle limit set by NGSS. Default 15 degrees.
    ndays: float
        half the number of days in a sector (assumes that the sector is centered on anti-solar). Default 14 days.
    """
    for i in range(-ndays, ndays + 1, 1):
        ans = get_critical_angles(dlong + i, decl, roll)
        cx = cy = cz = ""
        if ans[0] < 0:
            cx = "Sun on -X"
        if ans[1] > -5:
            cz = "Sun above sunshade"
        if ans[2] > ylim:
            cy = "Sun > Y angle limit"

        print("%3d %2d %6.2f %6.2f %s %s %s" % (i, ans[0], ans[1], ans[2], cx, cy, cz))


def calculate_allowable_map(dlong, decl, roll, ylim=15, ndays=14):
    """
    Given a dlong, decl array, returns the array of booleans that shows if that set of parameters is allowable.

    Parameters:
    -----------
    dlong: float
        the difference in ecliptic longitude from antisolar in the middle of the sector (0 for N-S orientations, but non-zero for ecliptic pointings),
    decl: float
        the ecliptic declination of the pointing
    roll: float
        the roll about S/C boresight (Z)
    ylim: float
        the antisolar angle limit set by NGSS. Default 15 degrees.
    ndays: float
        half the number of days
    """
    dlong3, decl3 = np.asarray([dlong - ndays, dlong, dlong + ndays]), np.asarray(
        [decl, decl, decl]
    )
    if isinstance(roll, (float, int, np.integer)):        
        sign_x, sun_angle, sun_angle_SC = get_critical_angles(
            dlong3.ravel(), decl3.ravel(), roll
        )
    elif isinstance(roll, np.ndarray) & hasattr(roll, '__iter__'):
        roll3 = np.asarray(
            [roll, roll, roll]
        )
        sign_x, sun_angle, sun_angle_SC = get_critical_angles(
            dlong3.ravel(), decl3.ravel(), roll3.ravel()
        )
    else:
        raise ValueError(roll)
    allowable = (
        (
            (sign_x > 0)
            & (sun_angle < -5)  # positive X
            & (
                sun_angle_SC < ylim
            )  # sun below sunshade  # Y angle < NGCC critical angle
        )
        .reshape(dlong3.shape)
        .all(axis=0)
    )  # All true at beginning, middle, and end of sector.
    return allowable

def build_grid():
    dlong, decl = np.mgrid[-180:179, -90:90]
    dlong, decl = dlong[::2, ::2], decl[::2, ::2]
    rolls = np.arange(-180, 179, 5)
    r = []

    a = np.zeros((dlong.shape[0], decl.shape[1], rolls.shape[0]))
    for kdx, roll in enumerate(rolls):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            allowable = calculate_allowable_map(dlong, decl, roll)
        r.append(np.vstack([dlong[allowable], decl[allowable], np.ones(allowable.sum()) * roll]).T)
    r = np.vstack(r)
    pickle.dump(r, open('allowable_pointings_grid.p', 'wb'))