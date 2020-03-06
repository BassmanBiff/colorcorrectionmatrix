# Introduction
This project aims to quickly and easily generate and apply a 3x3 color correction matrix (CCM).

## Disclaimer
This is a heavily-modifed fork of [lighttransport/colorcorrectionmatrix](https://github.com/lighttransport/colorcorrectionmatrix). I (dirtbirb) have added a color extraction tool (extractColor.py), extensively modified two of the original tools (computeCCM.py and correctColor.py), and removed much of the original content. I want to give credit where credit is due, but please don't blame lighttrasnport if this does something horrible to your computer. Or me, for that matter (see MIT license).

## Contents
This repository contains three related tools:
- **extractColor**: Find and extract color values from a raw or processed image containing a 24-chip ColorChecker grid.
- **computeCCM**: Compute the color correction matrix (CCM) to convert from one set of colors to another other in XYZ space.
- **correctColor**: Apply a CCM and other corrections to a raw or processed image.

Each tool is described in detail below.

## Dependencies
Version numbers only reflect the versions used for development, other versions may work too. Developed on Linux Mint 18.3 Sylvia.
- OpenCV 3.4
- Python 3.5
    - exifread 2.1
    - numpy 1.14
    - opencv-python 3.4
    - rawpy 0.12

---

# extractColor
Extract color values from an image containing a standard x-lite colorchecker grid.

This script attempts to locate color chips in an image by finding areas of max gradient, discarding shapes that don't look like a color chip, and then reconstructing any chips in the grid that it missed. Colors are then averaged within each chip and reported in a csv formatted for use with computeCCM. This auto-detection feature could be a lot more intelligent than it currently is, if anyone wants to implement this the right way!

Currently, only 8-bit png and 10-bit dng source images have been tested, but other formats up to 16-bit should work as well. Example images are provided in the `img` directory for testing.

## Usage
``` shell
$ extractColor.py [-h] -x X -y Y [-g GAMMA] [-v] input_image output_csv
```
Required arguments:
- `input_image` Source image
- `output_csv` Path to save color information
- `-x X` Expected width of color chips, in pixels
- `-y Y` Expected height of color chips, in pixels

Optional arguments:
- `-h, --help` Show help message and exit
- `-g GAMMA, --gamma GAMMA` Gamma value of input image, default 1.0 (no gamma correction)
- `-v, --verbose` Verbose output

---

# computeCCM
Compute the color correction matrix (CCM) necessary to convert one set of colors to another in XYZ color space.

This script, given two sets of 24x3 sRGB color information A and B, simply converts both to XYZ color space and solves the equation Ax = B, where x is a 3x3 CCM. The same CCM can't perfectly match all 24 colors, so a least squares routine (numpy.lingalg.lstsq()) is used to find the CCM that minimizes overall error.

Example data are provided in the `data` directory.

## Usage
``` shell
$ computeCCM.py [-h] [-g GAMMA] [-i ILLUMINANT] [-v] reference_csv source_csv output_csv
```
Required arguments:
- `reference_csv` CSV containing reference color information, to be matched
- `source_csv` CSV containing source color information, to be converted
- `output_csv` Path to save the calculated CCM as a CSV

Optional arguments:
- `-g GAMMA, --gamma GAMMA` Gamma value of reference and source data
- `-h, --help` Show help message and exit
- `-i ILLUMINANT, --illuminant ILLUMINANT` lluminant of source and reference images (default D65)
- `-v, --verbose` verbose output

---

# correctColor
Apply color correction and other operations to a raw or processed source image.

This script corrects a given image using a ccm as calculated by computeCCM using the following routine:
- Normalize (convert to range 0 - 1)
- Linearize (remove any existing gamma correction, as indicated by the -g flag)
- Convert to XYZ color space
- Correct colors by applying the provided CCM
- Convert back to sRGB color space
- Apply gamma correction (gamma = 2.2 unless specified)
- If source image is a raw format, perform auto white balance and black level
- Optionally apply auto brightness adjustment
- Save and display result

This is my first attempt at a pipeline for developing raw images, so improvements to any step in this pipeline are welcome. 

Example images and data are provided in the `img` and `data` directories.

## Usage
``` shell
correctColor.py [-h] [-b] [-g GAMMA] [-i ILLUMINANT] [-v] ccm input [output]
```
Required arguments:
- `ccm` CSV containing the CCM to apply
- `input` Source image, processed or raw

Optional arguments:
- `output` Path to save the corrected image, if desired
- `-b, --brightness` Auto-brightness adjustment (done automatically if necessary)
- `-g GAMMA, --gamma GAMMA` Gamma value of source img (default 1, no gamma applied)
- `-h, --help` Show help message and exit
- `-i ILLUMINANT, --illuminant ILLUMINANT` Illuminant, D50 or D65 (default D65)
- `-v, --verbose` Verbose output

---

# Example
This example demonstrates how to extract color info from a source image and a reference image, compute the CCM to convert the source colors to match the reference colors, and then apply that CCM to generate a corrected image.

## Commands
``` shell
./extractColor.py -g2.2 -x35 -y25 img/example_render.png data/example_colors_render.csv

./extractColor.py -g2.2 -x34 -y20 img/example_ref.png data/example_colors_ref.csv

./computeCCM.py data/example_colors_ref.csv data/example_colors_render.csv data/example_ccm.csv

./correctColor.py -g2.2 -b data/example_ccm.csv img/example_render.png img/example_render_corrected.png
```

## Result
Source image to correct:

![source image to correct](./img/example_render.png)

Reference image to match:

![reference image to match](./img/example_ref.png)

Corrected source image:

![corrected image](./img/example_render_corrected.png)

---

# License
This repo was originally published under the MIT license. It has been heavily modified from its source, but I'm leaving the MIT license as-is.

See the Dependencies section for third-party dependencies, each of which is published under its own license.

# References
Original repo: [lighttransport/colorcorrectionmatrix](https://github.com/lighttransport/colorcorrectionmatrix)
