import numpy as np
import torch


# ============================================================
# RESGATEDUNET INFERENCE
# ============================================================

def generate_probability_maps(
    model,
    X_test,
    device
):

    """
    Generate segmentation probability maps.

    ResGatedUNet output:

        output[:, 0, :, :] -> segmentation logits
        output[:, 1, :, :] -> edge logits

    Only channel 0 is returned because MCTM operates on
    the segmentation probability map.

    The edge output is used during training for edge loss.
    """

    # --------------------------------------------------------
    # CONVERT TO TENSOR
    # --------------------------------------------------------

    if isinstance(
        X_test,
        np.ndarray
    ):

        X_tensor = (
            torch.from_numpy(
                X_test
            ).float()
        )

    else:

        X_tensor = (
            X_test.float()
        )

    # --------------------------------------------------------
    # CHANNEL DIMENSION
    # --------------------------------------------------------

    if X_tensor.ndim == 3:

        # N,H,W
        #
        # ->
        #
        # N,1,H,W

        X_tensor = (
            X_tensor.unsqueeze(1)
        )

    elif (
        X_tensor.ndim == 4
        and
        X_tensor.shape[-1] == 1
    ):

        # N,H,W,1
        #
        # ->
        #
        # N,1,H,W

        X_tensor = (
            X_tensor.permute(
                0,
                3,
                1,
                2
            )
        )

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    X_tensor = (
        X_tensor.to(device)
    )

    model.eval()

    probability_maps = []

    # ========================================================
    # INFERENCE
    # ========================================================

    with torch.no_grad():

        for i in range(
            X_tensor.shape[0]
        ):

            image_tensor = (
                X_tensor[
                    i:i + 1
                ]
            )

            # ------------------------------------------------
            # MODEL
            # ------------------------------------------------

            output = model(
                image_tensor
            )

            # ------------------------------------------------
            # VERIFY TWO OUTPUT CHANNELS
            # ------------------------------------------------

            if (
                output.ndim != 4
                or
                output.shape[1] < 2
            ):

                raise ValueError(
                    "Expected ResGatedUNet "
                    "output with two channels: "
                    "segmentation and edge."
                )

            # ------------------------------------------------
            # SEGMENTATION MAP
            # ------------------------------------------------
            #
            # IMPORTANT:
            #
            # Channel 0 = segmentation
            # Channel 1 = edge
            #
            # MCTM uses ONLY channel 0.
            # ------------------------------------------------

            segmentation_probability = (
                torch.sigmoid(
                    output[
                        :,
                        0,
                        :,
                        :
                    ]
                )
            )

            segmentation_probability = (
                segmentation_probability
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
            )

            probability_maps.append(
                segmentation_probability
            )

    # --------------------------------------------------------
    # ARRAY
    # --------------------------------------------------------

    probability_maps = np.asarray(
        probability_maps,
        dtype=np.float32
    )

    return probability_maps
