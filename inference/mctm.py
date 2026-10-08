import numpy as np
import pandas as pd

from scipy.ndimage import maximum_filter
from skimage.feature import match_template
from skimage.draw import circle_perimeter


# ============================================================
# CIRCULAR RING TEMPLATE
# ============================================================

def create_circular_ring_template(radius, thickness=2):

    radius = int(round(radius))

    if radius < 2:
        return None

    padding = thickness + 2

    size = 2 * radius + 2 * padding + 1

    template = np.zeros(
        (size, size),
        dtype=np.float32
    )

    center = size // 2

    half_t = max(
        1,
        thickness // 2
    )

    for offset in range(
        -half_t,
        half_t + 1
    ):

        r = max(
            1,
            radius + offset
        )

        rr, cc = circle_perimeter(
            center,
            center,
            r,
            shape=template.shape
        )

        template[rr, cc] = 1.0

    if template.sum() > 0:

        template /= template.sum()

    return template


# ============================================================
# MULTI-SCALE CIRCULAR TEMPLATE MATCHING
# ============================================================

def circular_template_matching(
    segmentation,
    min_radius=3,
    max_radius=100,
    radius_step=2,
    ring_thickness=2,
    ncc_threshold=0.50,
    local_max_distance=5
):

    """
    Multi-scale circular template matching.

    Input:
        segmentation:
            2-D segmentation probability map.

    Output:
        List of predicted circular crater candidates.

    Each detection contains:
        x
        y
        radius
        diameter
        score
    """

    segmentation = np.asarray(
        segmentation,
        dtype=np.float32
    )

    segmentation = np.squeeze(
        segmentation
    )

    if segmentation.ndim != 2:

        raise ValueError(
            "Expected a 2-D segmentation "
            f"map, received shape "
            f"{segmentation.shape}"
        )

    # --------------------------------------------------------
    # IMAGE-WISE NORMALIZATION
    # --------------------------------------------------------

    seg_min = segmentation.min()

    seg_max = segmentation.max()

    if seg_max > seg_min:

        segmentation = (
            (segmentation - seg_min)
            / (seg_max - seg_min)
        )

    detections = []

    # --------------------------------------------------------
    # MULTI-SCALE MATCHING
    # --------------------------------------------------------

    for radius in range(
        min_radius,
        max_radius + 1,
        radius_step
    ):

        template = create_circular_ring_template(
            radius=radius,
            thickness=ring_thickness
        )

        if template is None:
            continue

        if (
            template.shape[0]
            > segmentation.shape[0]
            or
            template.shape[1]
            > segmentation.shape[1]
        ):
            continue

        # ----------------------------------------------------
        # NORMALIZED CROSS CORRELATION
        # ----------------------------------------------------

        response = match_template(
            segmentation,
            template,
            pad_input=True
        )

        # ----------------------------------------------------
        # NCC THRESHOLD
        # ----------------------------------------------------

        candidate_mask = (
            response >= ncc_threshold
        )

        # ----------------------------------------------------
        # LOCAL MAXIMUM
        # ----------------------------------------------------

        local_max = (
            response
            ==
            maximum_filter(
                response,
                size=(
                    2 * local_max_distance
                    + 1
                )
            )
        )

        peaks = (
            candidate_mask
            &
            local_max
        )

        ys, xs = np.where(
            peaks
        )

        # ----------------------------------------------------
        # STORE CANDIDATES
        # ----------------------------------------------------

        for x, y in zip(
            xs,
            ys
        ):

            score = float(
                response[y, x]
            )

            detections.append({

                "x": float(x),

                "y": float(y),

                "radius": float(radius),

                "diameter": float(
                    2 * radius
                ),

                "score": score
            })

    # --------------------------------------------------------
    # NO DETECTIONS
    # --------------------------------------------------------

    if len(detections) == 0:

        return []

    # --------------------------------------------------------
    # SORT BY NCC SCORE
    # --------------------------------------------------------

    detections_df = pd.DataFrame(
        detections
    )

    detections_df = (
        detections_df
        .sort_values(
            "score",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return detections_df.to_dict(
        "records"
    )
