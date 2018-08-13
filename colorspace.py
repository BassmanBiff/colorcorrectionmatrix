# Shared colorspace conversions

import numpy as np


def sRGB2XYZ(srgb, illuminant):
    if illuminant == 'D50':
        M = np.array([[0.4360747, 0.3850649, 0.1430804],
                      [0.2225045, 0.7168786, 0.0606169],
                      [0.0139322, 0.0971045, 0.7141733]])
    elif illuminant == 'D65':
        M = np.array([[0.412391, 0.357584, 0.180481],
                      [0.212639, 0.715169, 0.072192],
                      [0.019331, 0.119195, 0.950532]])
    else:
        raise ValueError('Invalid illuminant: {}'.format(illuminant))

    if isinstance(srgb, list):
        xyzList = []
        for l in srgb:
            xyzSubList = []
            for rgb in l:
                # (r, g, b)
                xyz = np.dot(M, rgb.transpose())
                xyzSubList.append(xyz.transpose())
            xyzList.append(np.asarray(xyzSubList))
        xyz = np.asarray(xyzList)
    elif isinstance(srgb, np.array):
        xyz = srgb.convert("RGB", M)

    return xyz


def XYZ2sRGB(xyz, illuminant):
    if illuminant == 'D50':
        M = np.array([[3.1338561, -1.6168667, -0.4906146],
                      [-0.9787684, 1.9161415,  0.0334540],
                      [0.0719453, -0.2289914,  1.4052427]])
    elif illuminant == 'D65':
        M = np.array([[3.240970, -1.537383, -0.498611],
                      [-0.969244, 1.875968, 0.041555],
                      [0.055630, -0.203977, 1.056972]])
    else:
        raise ValueError('Invalid illuminant: {}'.format(illuminant))

    if isinstance(xyz, list):
        xyzList = []
        for l in xyz:
            xyzSubList = []
            for rgb in l:
                # (r, g, b)
                xyz = np.dot(M, rgb.transpose())
                xyzSubList.append(xyz.transpose())
            xyzList.append(np.asarray(xyzSubList))
        xyz = np.asarray(xyzList)
    elif isinstance(xyz, np.array):
        srgb = xyz.convert("RGB", M)

    return srgb
