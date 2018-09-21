# TODO: Calculate colorspace conversions, don't use plug-and-play
# TODO: Full gamma correction (not simplified version)

import cv2
import exifread as exif
import numpy as np
import rawpy


# Input / output
def imread(filename, gamma=1):
    '''Load image as 16-bit RGB (OpenCV default is BGR)'''
    if filename[-4:] == '.png':         # png
        img = np.uint16(cv2.imread(filename)) << 8  # Load, convert to 16-bit
        img = bgr2rgb(img)                          # BGR -> RGB
    elif filename[-4:] == '.dng':       # dng with demosaicing
        raw = rawpy.imread(filename)
        img = raw.postprocess(
            demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR,
            half_size=False,
            four_color_rgb=False,
            fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode.Off,
            noise_thr=None,
            median_filter_passes=0,
            use_camera_wb=True,                 # No white balancing
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
        white_level = tags['Image Tag 0xC61D'].values[0]
        bits = int(np.log(white_level+1) / np.log(2))       # Infer bit depth
        img = img << (16 - bits)                            # Shift to 16-bit
    return img


def imshow(title, img, scale=1):
    '''Display image with optional rescaling'''
    cv2.imshow(title, cv2.resize(img, (0, 0), fx=scale, fy=scale))
    cv2.waitKey(0)


def imwrite(filename, img):
    '''Save image to disk'''
    cv2.imwrite(filename, img)


# Color conversions
def bgr2rgb(bgr):
    return bgr[..., ::-1]


def rgb2bgr(rgb):
    return rgb[..., ::-1]


def rgb2xyz(rgb, illuminant):
    ''' Convert RGB color space to XYZ
        Image must be linear (no gamma) and normalized (range 0 - 1) '''
    if illuminant == 'D50':
        M = np.array([[0.4360747, 0.3850649, 0.1430804],
                      [0.2225045, 0.7168786, 0.0606169],
                      [0.0139322, 0.0971045, 0.7141733]])
    elif illuminant == 'D65':
        M = np.array([[0.4124564, 0.3575761, 0.1804375],
                      [0.2126729, 0.7151522, 0.0721750],
                      [0.0193339, 0.1191920, 0.9503041]])
    else:
        raise ValueError("Invalid illuminant: {}".format(illuminant))
    return np.dot(rgb, M)


def xyz2rgb(xyz, illuminant):
    ''' Convert XYZ color space to RGB
        Image must be linear (no gamma) and normalized (range 0 - 1) '''
    if illuminant == 'D50':
        M = np.array([[3.1338561, -1.6168667, -0.4906146],
                      [-0.9787684, 1.9161415,  0.0334540],
                      [0.0719453, -0.2289914,  1.4052427]])
    elif illuminant == 'D65':
        M = np.array([[3.2404542, -1.5371385, -0.4985314],
                      [-0.9692660, 1.8760108,  0.0415560],
                      [0.0556434, -0.2040259,  1.0572252]])
    else:
        raise ValueError("Invalid illuminant: {}".format(illuminant))
    return np.dot(xyz, M)


if __name__ == '__main__':
    print("This is just a library!")
