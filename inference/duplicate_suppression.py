import numpy as np


# ============================================================
# CIRCULAR DUPLICATE SUPPRESSION
# ============================================================

def remove_circular_duplicates(
    detections,
    center_distance_factor=0.5,
    radius_difference_factor=0.25
):

    """
    Remove duplicate circular detections.

    Detections are sorted by NCC score.

    A candidate is suppressed when BOTH conditions hold:

        center_distance
        <
        center_distance_factor * smaller_radius

    and

        radius_difference
        <
        radius_difference_factor * smaller_radius

    Default values:

        center_distance_factor = 0.5
        radius_difference_factor = 0.25
    """

    if len(detections) == 0:

        return []

    # --------------------------------------------------------
    # SORT BY CONFIDENCE
    # --------------------------------------------------------

    detections = sorted(
        detections,
        key=lambda x: x["score"],
        reverse=True
    )

    selected = []

    # --------------------------------------------------------
    # DUPLICATE SUPPRESSION
    # --------------------------------------------------------

    for candidate in detections:

        keep = True

        for existing in selected:

            dx = (
                candidate["x"]
                -
                existing["x"]
            )

            dy = (
                candidate["y"]
                -
                existing["y"]
            )

            center_distance = np.sqrt(
                dx ** 2
                +
                dy ** 2
            )

            smaller_radius = min(
                candidate["radius"],
                existing["radius"]
            )

            if smaller_radius <= 0:

                continue

            radius_difference = abs(
                candidate["radius"]
                -
                existing["radius"]
            )

            close_centers = (
                center_distance
                <
                center_distance_factor
                *
                smaller_radius
            )

            similar_radius = (
                radius_difference
                <
                radius_difference_factor
                *
                smaller_radius
            )

            if (
                close_centers
                and
                similar_radius
            ):

                keep = False

                break

        if keep:

            selected.append(
                candidate
            )

    return selected
