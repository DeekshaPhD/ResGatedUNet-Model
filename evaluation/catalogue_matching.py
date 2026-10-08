import numpy as np


# ============================================================
# EXTRACT GROUND-TRUTH CRATERS
# ============================================================

def extract_gt_circles_from_catalogue(
    crater_store,
    image_id
):

    key = (
        f"/img_{image_id:05d}"
    )

    if key not in crater_store:

        return []

    df = crater_store[key]

    if df is None or len(df) == 0:

        return []

    df = df.copy()

    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    required_columns = [
        "x",
        "y",
        "Diameter (pix)"
    ]

    missing = [
        c
        for c in required_columns
        if c not in df.columns
    ]

    if len(missing) > 0:

        raise ValueError(
            f"Image {image_id}: "
            f"Missing catalogue columns: "
            f"{missing}. "
            f"Available columns: "
            f"{list(df.columns)}"
        )

    gt_circles = []

    for _, row in df.iterrows():

        try:

            x = float(
                row["x"]
            )

            y = float(
                row["y"]
            )

            diameter = float(
                row["Diameter (pix)"]
            )

        except (
            ValueError,
            TypeError
        ):

            continue

        if not (
            np.isfinite(x)
            and
            np.isfinite(y)
            and
            np.isfinite(diameter)
        ):

            continue

        if diameter <= 0:

            continue

        radius = (
            diameter / 2.0
        )

        gt_circles.append({

            "x": x,

            "y": y,

            "radius": radius,

            "diameter": diameter
        })

    return gt_circles


# ============================================================
# ONE-TO-ONE CIRCULAR CRATER MATCHING
# ============================================================

def match_circular_craters(
    predicted,
    ground_truth,
    center_tolerance=2.0,
    radius_tolerance=1.0
):

    """
    Confidence-ordered one-to-one matching.

    A prediction is eligible when:

        normalized center error <= center_tolerance

    and

        normalized radius error <= radius_tolerance

    Each GT crater can be matched only once.
    """

    matches = []

    used_gt = set()

    # --------------------------------------------------------
    # SORT PREDICTIONS BY NCC SCORE
    # --------------------------------------------------------

    predicted = sorted(
        predicted,
        key=lambda x: x["score"],
        reverse=True
    )

    # --------------------------------------------------------
    # MATCH EACH PREDICTION
    # --------------------------------------------------------

    for pred_id, pred in enumerate(
        predicted
    ):

        best_gt = None

        best_distance = np.inf

        best_center_error = np.nan

        best_radius_error = np.nan

        # ----------------------------------------------------
        # GROUND-TRUTH LOOP
        # ----------------------------------------------------

        for gt_id, gt in enumerate(
            ground_truth
        ):

            # ----------------------------------------------
            # ONE-TO-ONE ENFORCEMENT
            # ----------------------------------------------

            if gt_id in used_gt:

                continue

            dx = (
                pred["x"]
                -
                gt["x"]
            )

            dy = (
                pred["y"]
                -
                gt["y"]
            )

            center_distance = np.sqrt(
                dx ** 2
                +
                dy ** 2
            )

            min_radius = min(
                pred["radius"],
                gt["radius"]
            )

            if min_radius <= 0:

                continue

            # ----------------------------------------------
            # NORMALIZED CENTER ERROR
            # ----------------------------------------------

            normalized_center_error = (
                center_distance
                /
                min_radius
            )

            # ----------------------------------------------
            # NORMALIZED RADIUS ERROR
            # ----------------------------------------------

            normalized_radius_error = (
                abs(
                    pred["radius"]
                    -
                    gt["radius"]
                )
                /
                min_radius
            )

            # ----------------------------------------------
            # MATCH CONDITION
            # ----------------------------------------------

            if (
                normalized_center_error
                <= center_tolerance
                and
                normalized_radius_error
                <= radius_tolerance
            ):

                # Select nearest eligible GT
                if (
                    center_distance
                    <
                    best_distance
                ):

                    best_distance = (
                        center_distance
                    )

                    best_gt = gt_id

                    best_center_error = (
                        normalized_center_error
                    )

                    best_radius_error = (
                        normalized_radius_error
                    )

        # ----------------------------------------------------
        # STORE MATCH
        # ----------------------------------------------------

        if best_gt is not None:

            matches.append({

                "pred_id":
                    pred_id,

                "gt_id":
                    best_gt,

                "center_distance":
                    best_distance,

                "normalized_center_error":
                    best_center_error,

                "normalized_radius_error":
                    best_radius_error,

                "radius_error":
                    abs(
                        predicted[pred_id]["radius"]
                        -
                        ground_truth[best_gt]["radius"]
                    ),

                "diameter_error":
                    abs(
                        predicted[pred_id]["diameter"]
                        -
                        ground_truth[best_gt]["diameter"]
                    )
            })

            used_gt.add(
                best_gt
            )

    return matches
