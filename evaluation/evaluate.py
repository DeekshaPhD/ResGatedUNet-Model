import os
import sys

import h5py
import torch


REPO_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


from model.resgatedunet import ResUNet50
from preprocessing.preprocess import preprocess
from evaluation.metrics import get_metrics


# ============================================================
# CONFIGURATION
# ============================================================

TEST_FILE = os.path.join(
    REPO_ROOT,
    "data",
    "test_images.hdf5"
)

WEIGHTS_FILE = os.path.join(
    REPO_ROOT,
    "weights",
    "ResGatedUNet_weights.pth"
)

DIM = 256
BATCH_SIZE = 8

THRESHOLD = 0.5
BOUNDARY_TOLERANCE = 2
CRATER_IOU_THRESHOLD = 0.3
MIN_CRATER_AREA = 5


# ============================================================
# LOAD DATA
# ============================================================

def load_test_data(path):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Test dataset not found:\n{path}\n\n"
            "Please prepare the test dataset and "
            "place it at the configured location."
        )

    with h5py.File(path, "r") as f:

        X = f["input_images"][:]
        Y = f["target_masks"][:]

    return X, Y


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("ResGatedUNet Evaluation")
    print("=" * 60)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    print("\nLoading test dataset...")

    X_test, Y_test = load_test_data(
        TEST_FILE
    )

    print(
        f"Raw X shape: {X_test.shape}"
    )

    print(
        f"Raw Y shape: {Y_test.shape}"
    )

    Data = {
        "test": [
            X_test,
            Y_test
        ]
    }

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    print("\nPreprocessing...")

    preprocess(
        Data,
        DIM
    )

    X_test = Data["test"][0]
    Y_test = Data["test"][1]

    print(
        f"Processed X shape: {X_test.shape}"
    )

    print(
        f"Processed Y shape: {Y_test.shape}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\nCreating model...")

    model = ResUNet50(
        in_channels=1,
        num_classes=1,
        pretrained=False
    ).to(device)

    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

    if not os.path.exists(
        WEIGHTS_FILE
    ):

        raise FileNotFoundError(
            f"Weights not found:\n{WEIGHTS_FILE}"
        )

    print(
        "Loading trained weights..."
    )

    state_dict = torch.load(
        WEIGHTS_FILE,
        map_location=device
    )

    if (
        isinstance(state_dict, dict)
        and "model_state_dict" in state_dict
    ):

        state_dict = (
            state_dict["model_state_dict"]
        )

    model.load_state_dict(
        state_dict,
        strict=True
    )

    print(
        "Weights loaded successfully."
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print(
        "\nRunning evaluation..."
    )

    results = get_metrics(
        data=[
            X_test,
            Y_test
        ],
        craters=None,
        dim=DIM,
        model=model,
        threshold=THRESHOLD,
        batch_size=BATCH_SIZE,
        boundary_tolerance=BOUNDARY_TOLERANCE,
        crater_iou_threshold=CRATER_IOU_THRESHOLD,
        min_crater_area=MIN_CRATER_AREA
    )

    print("\n" + "=" * 60)
    print("Evaluation completed.")
    print("=" * 60)

    print("\nSummary:")

    for key, value in results.items():

        if isinstance(value, float):

            print(
                f"{key:25s}: {value:.6f}"
            )

        else:

            print(
                f"{key:25s}: {value}"
            )


if __name__ == "__main__":
    main()
