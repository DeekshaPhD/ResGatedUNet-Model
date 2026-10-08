"""
Patch generation for the ResGatedUNet lunar crater dataset.

This script follows the DeepMoon-style randomized,
projection-corrected patch-generation approach.

Reference:
Silburt et al., DeepMoon
https://github.com/silburt/DeepMoon
"""

import os
import numpy as np

# DeepMoon-derived functions
from input_data_gen import (
    ReadLROCHeadCombinedCraterCSV,
    InitialImageCut,
    ResampleCraters,
    GenDataset
)


# ============================================================
# DATASET PARAMETERS
# ============================================================

SOURCE_IMAGE = "path/to/global_dem.png"

LROC_CATALOGUE = "path/to/LROCCraters.csv"
HEAD_CATALOGUE = "path/to/HeadCraters.csv"

OUTPUT_PREFIX = "./dataset/train"

NUM_PATCHES = 30000

INPUT_SIZE = 256
TARGET_SIZE = 256

RAW_SIZE_RANGE = [500, 6500]
RAW_SIZE_DISTRIBUTION = "log"

MIN_CRATER_PIXELS = 1.0

MOON_RADIUS_KM = 1737.4

SOURCE_REGION = [
    -180.0,
    180.0,
    -60.0,
    60.0
]

TRAIN_REGION = [
    -180.0,
    -60.0,
    -60.0,
    60.0
]

TRUNCATE_AT_PADDING = True

RING_WIDTH = 1


# ============================================================
# GENERATE PATCHES
# ============================================================

def generate_dataset():

    print("Loading DEM...")

    # Load DEM
    from PIL import Image

    image = Image.open(
        SOURCE_IMAGE
    ).convert("L")

    print("Loading crater catalogues...")

    craters = ReadLROCHeadCombinedCraterCSV(
        filelroc=LROC_CATALOGUE,
        filehead=HEAD_CATALOGUE
    )

    print("Restricting DEM to training region...")

    image = InitialImageCut(
        image,
        SOURCE_REGION,
        TRAIN_REGION
    )

    print("Filtering crater catalogue...")

    craters = ResampleCraters(
        craters,
        TRAIN_REGION,
        None,
        arad=MOON_RADIUS_KM
    )

    print("Generating patches...")

    GenDataset(
        image,
        craters,
        OUTPUT_PREFIX,

        rawlen_range=RAW_SIZE_RANGE,
        rawlen_dist=RAW_SIZE_DISTRIBUTION,

        ilen=INPUT_SIZE,
        tglen=TARGET_SIZE,

        cdim=TRAIN_REGION,
        arad=MOON_RADIUS_KM,

        minpix=MIN_CRATER_PIXELS,

        binary=True,
        rings=True,
        ringwidth=RING_WIDTH,

        truncate=TRUNCATE_AT_PADDING,

        amt=NUM_PATCHES,

        istart=0,

        verbose=True
    )

    print("Patch generation completed.")


if __name__ == "__main__":
    generate_dataset()
