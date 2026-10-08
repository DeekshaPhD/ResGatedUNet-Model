import numpy as np
import pandas as pd
import h5py
import torch

from inference.run_inference import (
    generate_probability_maps
)

from evaluation.metrics import (
    evaluate_mctm,
    print_mctm_results
)


# ============================================================
# CONFIGURATION
# ============================================================

N_TEST = 3000

DIM = 256

# ------------------------------------------------------------
# MCTM PARAMETERS
# ------------------------------------------------------------

NCC_THRESHOLD = 0.50

MIN_RADIUS = 3

MAX_RADIUS = 100

RADIUS_STEP = 2

RING_THICKNESS = 2

LOCAL_MAX_DISTANCE = 5

# ------------------------------------------------------------
# DUPLICATE SUPPRESSION
# ------------------------------------------------------------

DUPLICATE_CENTER_DISTANCE_FACTOR = 0.5

DUPLICATE_RADIUS_DIFFERENCE_FACTOR = 0.25

# ------------------------------------------------------------
# CRATER MATCHING
# ------------------------------------------------------------

CENTER_TOLERANCE = 2.0

RADIUS_TOLERANCE = 1.0

# ------------------------------------------------------------
# BOUNDARY
# ------------------------------------------------------------

BOUNDARY_TOLERANCE = 2

# ------------------------------------------------------------
# SEGMENTATION
# ------------------------------------------------------------

SEGMENTATION_THRESHOLD = 0.50


# ============================================================
# USER PATHS
# ============================================================
#
# Change these paths for your environment.
# ============================================================

TEST_IMAGE_PATH = (
    "PATH_TO/test_images.hdf5"
)

TEST_CRATER_PATH = (
    "PATH_TO/test_craters.hdf5"
)

# Example:
#
# TEST_IMAGE_PATH = (
#     r"D:\Deeksha\Objective_I\Dataset"
#     r"\Final_30000_3000_3000"
#     r"\test_images.hdf5"
# )
#
# TEST_CRATER_PATH = (
#     r"D:\Deeksha\Objective_I\Dataset"
#     r"\Final_30000_3000_3000"
#     r"\test_craters.hdf5"
# )


# ============================================================
# MODEL
# ============================================================
#
# Import your ResGatedUNet class here.
#
# Example:
#
# from model.resgated_unet import ResUNet50
#
# model = ResUNet50(
#     in_channels=1,
#     num_classes=1,
#     pretrained=False
# )
#
# checkpoint = torch.load(
#     "weights/ResGatedUNet_weights.pth",
#     map_location=device
# )
#
# model.load_state_dict(checkpoint)
#
# ============================================================


def load_test_data():

    # --------------------------------------------------------
    # TEST IMAGE DATA
    # --------------------------------------------------------

    with h5py.File(
        TEST_IMAGE_PATH,
        "r"
    ) as test:

        X_test = (
            test["input_images"][
                :N_TEST
            ]
            .astype(
                np.float32
            )
        )

        Y_test = (
            test["target_masks"][
                :N_TEST
            ]
            .astype(
                np.float32
            )
        )

    # --------------------------------------------------------
    # TARGET NORMALIZATION
    # --------------------------------------------------------

    if Y_test.max() > 1:

        Y_test = (
            Y_test / 255.0
        )

    Y_test = (
        Y_test > 0.5
    ).astype(
        np.float32
    )

    return (
        X_test,
        Y_test
    )


def run_evaluation(
    model,
    device
):

    print(
        "\nLoading test data..."
    )

    X_test, Y_test = (
        load_test_data()
    )

    print(
        "X_test:",
        X_test.shape
    )

    print(
        "Y_test:",
        Y_test.shape
    )

    # ========================================================
    # MODEL INFERENCE
    # ========================================================

    print(
        "\nRunning ResGatedUNet inference..."
    )

    probability_maps = (
        generate_probability_maps(
            model=model,
            X_test=X_test,
            device=device
        )
    )

    print(
        "Segmentation probability maps:",
        probability_maps.shape
    )

    # ========================================================
    # OPEN CRATER CATALOGUE
    # ========================================================

    print(
        "\nOpening crater catalogue..."
    )

    crater_store = pd.HDFStore(
        TEST_CRATER_PATH,
        mode="r"
    )

    try:

        print(
            "Catalogue keys:",
            len(
                crater_store.keys()
            )
        )

        # ====================================================
        # COMPLETE EVALUATION
        # ====================================================

        results = evaluate_mctm(

            probability_maps=(
                probability_maps
            ),

            y_true=(
                Y_test
            ),

            crater_store=(
                crater_store
            ),

            n_test=(
                N_TEST
            ),

            # ------------------------------------------------
            # PIXEL
            # ------------------------------------------------

            segmentation_threshold=(
                SEGMENTATION_THRESHOLD
            ),

            # ------------------------------------------------
            # BOUNDARY
            # ------------------------------------------------

            boundary_tolerance=(
                BOUNDARY_TOLERANCE
            ),

            # ------------------------------------------------
            # MCTM
            # ------------------------------------------------

            ncc_threshold=(
                NCC_THRESHOLD
            ),

            min_radius=(
                MIN_RADIUS
            ),

            max_radius=(
                MAX_RADIUS
            ),

            radius_step=(
                RADIUS_STEP
            ),

            ring_thickness=(
                RING_THICKNESS
            ),

            local_max_distance=(
                LOCAL_MAX_DISTANCE
            ),

            # ------------------------------------------------
            # DUPLICATE SUPPRESSION
            # ------------------------------------------------

            duplicate_center_distance_factor=(
                DUPLICATE_CENTER_DISTANCE_FACTOR
            ),

            duplicate_radius_difference_factor=(
                DUPLICATE_RADIUS_DIFFERENCE_FACTOR
            ),

            # ------------------------------------------------
            # CRATER MATCHING
            # ------------------------------------------------

            center_tolerance=(
                CENTER_TOLERANCE
            ),

            radius_tolerance=(
                RADIUS_TOLERANCE
            )
        )

        # ====================================================
        # PRINT
        # ====================================================

        print_mctm_results(
            results
        )

        return results

    finally:

        crater_store.close()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device
    )

    # --------------------------------------------------------
    # Load your model here.
    # --------------------------------------------------------
    #
    # Example:
    #
    # from model.resgated_unet import ResUNet50
    #
    # model = ResUNet50(
    #     in_channels=1,
    #     num_classes=1,
    #     pretrained=False
    # )
    #
    # state_dict = torch.load(
    #     "weights/ResGatedUNet_weights.pth",
    #     map_location=device
    # )
    #
    # model.load_state_dict(
    #     state_dict
    # )
    #
    # model.to(device)
    #
    # run_evaluation(
    #     model,
    #     device
    # )

    raise SystemExit(
        "Load your ResGatedUNet model in this "
        "section and call run_evaluation(model, device)."
    )
