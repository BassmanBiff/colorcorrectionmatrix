#!/usr/bin/env python3

import argparse
from PIL import Image, ImageChops


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('reference', action='store')
    parser.add_argument('corrected', action='store')
    parser.add_argument('-o', '--outputBaseName', action='store',
                        default='diffImg')
    args = parser.parse_args()

    resizeFactor = 1.
    # offsetX = -75
    # offsetY = -110

    corrected_img = Image.open(args.corrected, 'r').convert("RGB")
    reference_img = Image.open(args.reference, 'r').convert("RGB")
    reference_img.thumbnail((corrected_img.width * resizeFactor,
                             corrected_img.height * resizeFactor),
                            Image.LANCZOS)
    # reference_img = reference_img.rotate(-4)
    # dst = Image.new('RGB', (corrected_img.width,
    #                         corrected_img.height),
    #                 (255, 255, 255))
    # dst.paste(reference_img, (offsetX, offsetY))
    # dst.save('transformed.png')
    diffIm = ImageChops.difference(reference_img, corrected_img)

    diff_px = diffIm.getdata()
    data = []
    for i in range(len(diff_px)):
        rgbRatio = (diff_px[i][0] + diff_px[i][1] + diff_px[i][2]) / 3. / 255.
        data.append((int(rgbRatio * 255.),
                     0,
                     int((1. - rgbRatio) * 255.)))
    diffIm.putdata(data)
    diffIm.save('{}.png'.format(args.outputBaseName))
