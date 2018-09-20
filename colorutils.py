import cv2
import exifread as exif
import numpy as np
import rawpy


def imshow(title, img, scale=1):
    cv2.imshow(title, cv2.resize(img, (0, 0), fx=scale, fy=scale))
    cv2.waitKey(0)


def applyMatrix(img, M):
    # img = np.array([
    #     [np.dot(M, img[i, j].T).T for j in range(img.shape[1])]
    #     for i in range(img.shape[0])])
    return img.dot(M)


# def gamma_table(gamma_r, gamma_g, gamma_b, gain_r=1.0, gain_g=1.0, gain_b=1.0):
#     rng = range(256)
#     r_tbl = [min(255, int((x / 255.) ** gamma_r * gain_r * 255.)) for x in rng]
#     g_tbl = [min(255, int((x / 255.) ** gamma_g * gain_g * 255.)) for x in rng]
#     b_tbl = [min(255, int((x / 255.) ** gamma_b * gain_b * 255.)) for x in rng]
#     return r_tbl + g_tbl + b_tbl


def applyGamma(img, gamma=2.2):
    if img.dtype == np.uint8:
        f = 255
    elif img.dtype == np.uint16:
        f = 65535
    return np.uint16(np.power(img / f, 1/gamma) * f)


def deGamma(img, gamma=2.2):
    if img.dtype == np.uint8:
        f = 255
    elif img.dtype == np.uint16:
        f = 65535
    return np.uint16(np.power(img / f, gamma) * f)


def sRGB2XYZ(rgb, illuminant):
    # Assumes linearized (degamma'd), 0-1 range
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
    return np.dot(rgb, M)


def XYZ2sRGB(xyz, illuminant):
    if illuminant == 'D50':
        M = np.array([[3.1338561, -1.6168667, -0.4906146],
                      [-0.9787684, 1.9161415,  0.0334540],
                      [0.0719453, -0.2289914,  1.4052427]])
    elif illuminant == 'D65':
        M = np.array([[3.240970, -1.537383, -0.498611],
                      [-0.969244, 1.875968,  0.041555],
                      [0.055630, -0.203977,  1.056972]])
    else:
        raise ValueError('Invalid illuminant: {}'.format(illuminant))
    return np.dot(xyz, M)


def load_image(filename, gamma=1):
    ''' Load image as 16-bit RGB (OpenCV default is BGR)'''
    if filename[-4:] == '.png':         # png, assumes sRGB
        img = np.uint16(cv2.imread(filename)) << 8
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif filename[-4:] == '.dng':       # dng with demosaicing
        raw = rawpy.imread(filename)
        img = raw.postprocess(
            demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR,
            half_size=False,
            four_color_rgb=False,
            fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode.Off,
            noise_thr=None,
            median_filter_passes=0,
            use_camera_wb=True,                 # No postprocess white balance
            use_auto_wb=False,
            user_wb=None,
            output_color=rawpy.ColorSpace.raw,  # No color space conversion
            output_bps=16,                      # 16 bits per sample
            user_flip=None,
            user_black=None,
            user_sat=None,
            no_auto_bright=True,                # No brightness adjustment
            auto_bright_thr=None,
            adjust_maximum_thr=0.75,
            bright=1.0,
            highlight_mode=rawpy.HighlightMode.Ignore,
            exp_shift=None,
            exp_preserve_highlights=0.0,
            no_auto_scale=True,
            gamma=(1, 1),                       # No gamma adjustment
            chromatic_aberration=None,
            bad_pixels_path=None)
        with open(filename, 'rb') as img_file:              # Get white level
            tags = exif.process_file(img_file)
        white_level = tags["Image Tag 0xC61D"].values[0]
        bits = int(np.log(white_level+1) / np.log(2))       # Infer bit depth
        img = img << (16 - bits)                            # Shift to 16-bit
    return img
