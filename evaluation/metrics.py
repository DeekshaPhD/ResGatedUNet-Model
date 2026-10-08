import numpy as np
import torch
import torch.nn as nn

from scipy.ndimage import (
    label,
    find_objects,
    binary_erosion,
    distance_transform_edt
)

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    jaccard_score
)


# ============================================================
# LOSS
# ============================================================

class BCEDiceEdgeLoss(nn.Module):

    """
    Multi-task loss for the two-output ResGatedUNet.

    Model output:
        channel 0 -> segmentation logits
        channel 1 -> edge logits

    Ground truth:
        channel 0 -> segmentation target
        channel 1 -> edge target

    Total loss:

        L = lambda1 * BCE_seg
          + lambda2 * Dice_seg
          + lambda3 * BCE_edge
    """

    def __init__(
        self,
        lambda1=1.0,
        lambda2=1.0,
        lambda3=1.5,
        smooth=1e-6
    ):

        super().__init__()

        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.smooth = smooth

        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        y_pred,
        y_true
    ):

        # ----------------------------------------------------
        # TWO MODEL OUTPUTS
        # ----------------------------------------------------

        if y_pred.ndim != 4:
            raise ValueError(
                "Expected y_pred with shape "
                "[B,2,H,W]."
            )

        if y_pred.shape[1] != 2:
            raise ValueError(
                "ResGatedUNet must output exactly "
                "two maps: segmentation and edge."
            )

        # ----------------------------------------------------
        # SEGMENTATION
        # ----------------------------------------------------

        seg_pred = y_pred[:, 0:1]

        seg_true = y_true[:, 0:1]

        # ----------------------------------------------------
        # EDGE
        # ----------------------------------------------------

        edge_pred = y_pred[:, 1:2]

        edge_true = y_true[:, 1:2]

        # ----------------------------------------------------
        # SEGMENTATION BCE
        # ----------------------------------------------------

        bce_loss = self.bce(
            seg_pred,
            seg_true
        )

        # ----------------------------------------------------
        # SEGMENTATION DICE
        # ----------------------------------------------------

        seg_prob = torch.sigmoid(
            seg_pred
        )

        intersection = (
            seg_prob * seg_true
        ).sum(
            dim=(1, 2, 3)
        )

        denominator = (
            seg_prob.sum(
                dim=(1, 2, 3)
            )
            +
            seg_true.sum(
                dim=(1, 2, 3)
            )
        )

        dice_score = (
            2.0 * intersection
            + self.smooth
        ) / (
            denominator
            + self.smooth
        )

        dice_loss = (
            1.0
            -
            dice_score.mean()
        )

        # ----------------------------------------------------
        # EDGE BCE
        # ----------------------------------------------------

        edge_loss = self.bce(
            edge_pred,
            edge_true
        )

        # ----------------------------------------------------
        # TOTAL LOSS
        # ----------------------------------------------------

        total_loss = (
            self.lambda1 * bce_loss
            +
            self.lambda2 * dice_loss
            +
            self.lambda3 * edge_loss
        )

        return (
            total_loss,
            bce_loss,
            dice_loss,
            edge_loss
        )


# ============================================================
# HELPER: TORCH -> NUMPY
# ============================================================

def _to_numpy(x):

    if torch.is_tensor(x):

        x = (
            x.detach()
            .cpu()
            .numpy()
        )

    return np.asarray(x)


# ============================================================
# BOUNDARY EXTRACTION
# ============================================================

def extract_boundary(mask):

    mask = _to_numpy(mask).astype(bool)

    if mask.size == 0:

        return np.zeros_like(
            mask,
            dtype=bool
        )

    structure = np.ones(
        (3, 3),
        dtype=bool
    )

    eroded = binary_erosion(
        mask,
        structure=structure,
        border_value=0
    )

    return (
        mask
        &
        (~eroded)
    )


# ============================================================
# PIXEL-LEVEL METRICS
# ============================================================

def calculate_pixel_metrics(
    pred_masks,
    gt_masks,
    threshold=0.5
):

    """
    Calculate global pixel-level metrics.

    Parameters
    ----------
    pred_masks:
        Segmentation probability maps.

        Supported:
            [N,H,W]
            [N,1,H,W]

    gt_masks:
        Segmentation targets.

        Supported:
            [N,H,W]
            [N,1,H,W]
            [N,2,H,W]

        If [N,2,H,W], only channel 0 is used.

    Returns
    -------
    dict
        Pseg, Rseg, F1seg, IoU, Dice
    """

    pred_masks = _to_numpy(
        pred_masks
    )

    gt_masks = _to_numpy(
        gt_masks
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    if pred_masks.ndim == 4:

        pred_masks = (
            pred_masks[:, 0]
        )

    elif pred_masks.ndim == 2:

        pred_masks = (
            pred_masks[np.newaxis, ...]
        )

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    if gt_masks.ndim == 4:

        # IMPORTANT:
        #
        # [N,2,H,W]
        #
        # channel 0 = segmentation
        # channel 1 = edge
        #
        # Pixel metrics use ONLY channel 0.

        gt_masks = (
            gt_masks[:, 0]
        )

    elif gt_masks.ndim == 2:

        gt_masks = (
            gt_masks[np.newaxis, ...]
        )

    # --------------------------------------------------------
    # BINARY MASKS
    # --------------------------------------------------------

    pred_binary = (
        pred_masks > threshold
    )

    gt_binary = (
        gt_masks > 0.5
    )

    # --------------------------------------------------------
    # FLATTEN
    # --------------------------------------------------------

    pred_flat = (
        pred_binary.flatten()
    )

    gt_flat = (
        gt_binary.flatten()
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    tp = np.logical_and(
        pred_flat,
        gt_flat
    ).sum()

    fp = np.logical_and(
        pred_flat,
        ~gt_flat
    ).sum()

    fn = np.logical_and(
        ~pred_flat,
        gt_flat
    ).sum()

    tn = np.logical_and(
        ~pred_flat,
        ~gt_flat
    ).sum()

    # --------------------------------------------------------
    # PRECISION
    # --------------------------------------------------------

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    # --------------------------------------------------------
    # RECALL
    # --------------------------------------------------------

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    # --------------------------------------------------------
    # F1
    # --------------------------------------------------------

    f1 = (
        2.0
        * precision
        * recall
        /
        (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # --------------------------------------------------------
    # IOU
    # --------------------------------------------------------

    iou = (
        tp
        /
        (tp + fp + fn)
        if (tp + fp + fn) > 0
        else 0.0
    )

    # --------------------------------------------------------
    # DICE
    # --------------------------------------------------------

    dice = (
        2.0 * tp
        /
        (2.0 * tp + fp + fn)
        if (2.0 * tp + fp + fn) > 0
        else 0.0
    )

    # --------------------------------------------------------
    # ACCURACY
    # --------------------------------------------------------

    accuracy = (
        (tp + tn)
        /
        (tp + tn + fp + fn)
        if (tp + tn + fp + fn) > 0
        else 0.0
    )

    return {

        "Pseg": float(
            precision
        ),

        "Rseg": float(
            recall
        ),

        "F1seg": float(
            f1
        ),

        "IoU": float(
            iou
        ),

        "Dice": float(
            dice
        ),

        "Pixel_Accuracy": float(
            accuracy
        ),

        "TP_pixels": int(tp),

        "FP_pixels": int(fp),

        "FN_pixels": int(fn)
    }


# ============================================================
# SINGLE-IMAGE BOUNDARY METRICS
# ============================================================

def calculate_boundary_metrics_single(
    pred_mask,
    gt_mask,
    tolerance=2
):

    pred_mask = _to_numpy(
        pred_mask
    )

    gt_mask = _to_numpy(
        gt_mask
    )

    pred_mask = np.squeeze(
        pred_mask
    )

    gt_mask = np.squeeze(
        gt_mask
    )

    pred_mask = (
        pred_mask > 0.5
    )

    gt_mask = (
        gt_mask > 0.5
    )

    pred_boundary = extract_boundary(
        pred_mask
    )

    gt_boundary = extract_boundary(
        gt_mask
    )

    n_pred = pred_boundary.sum()
    n_gt = gt_boundary.sum()

    # --------------------------------------------------------
    # BOTH EMPTY
    # --------------------------------------------------------

    if (
        n_pred == 0
        and
        n_gt == 0
    ):

        return (
            1.0,
            1.0,
            1.0
        )

    # --------------------------------------------------------
    # ONE EMPTY
    # --------------------------------------------------------

    if (
        n_pred == 0
        or
        n_gt == 0
    ):

        return (
            0.0,
            0.0,
            0.0
        )

    # --------------------------------------------------------
    # DISTANCE TRANSFORMS
    # --------------------------------------------------------

    dist_pred = distance_transform_edt(
        ~pred_boundary
    )

    dist_gt = distance_transform_edt(
        ~gt_boundary
    )

    # --------------------------------------------------------
    # BOUNDARY PRECISION
    # --------------------------------------------------------

    pred_distances = (
        dist_gt[pred_boundary]
    )

    boundary_precision = np.mean(
        pred_distances <= tolerance
    )

    # --------------------------------------------------------
    # BOUNDARY RECALL
    # --------------------------------------------------------

    gt_distances = (
        dist_pred[gt_boundary]
    )

    boundary_recall = np.mean(
        gt_distances <= tolerance
    )

    # --------------------------------------------------------
    # F1
    # --------------------------------------------------------

    denominator = (
        boundary_precision
        +
        boundary_recall
    )

    if denominator > 0:

        boundary_f1 = (
            2.0
            * boundary_precision
            * boundary_recall
            /
            denominator
        )

    else:

        boundary_f1 = 0.0

    return (
        float(boundary_precision),
        float(boundary_recall),
        float(boundary_f1)
    )


# ============================================================
# BATCH BOUNDARY METRICS
# ============================================================

def calculate_boundary_metrics(
    pred_masks,
    gt_masks,
    tolerance=2
):

    """
    Boundary-level evaluation.

    Returns:

        Pb
        Rb
        F1b
        ACD
        HD

    ACD and HD are calculated from the predicted
    and ground-truth contours.
    """

    pred_masks = _to_numpy(
        pred_masks
    )

    gt_masks = _to_numpy(
        gt_masks
    )

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    if pred_masks.ndim == 4:

        pred_masks = (
            pred_masks[:, 0]
        )

    elif pred_masks.ndim == 2:

        pred_masks = (
            pred_masks[np.newaxis, ...]
        )

    # --------------------------------------------------------
    # GROUND TRUTH
    # --------------------------------------------------------

    if gt_masks.ndim == 4:

        # IMPORTANT:
        #
        # channel 0 = segmentation
        # channel 1 = edge
        #
        # Boundary evaluation uses
        # the segmentation mask.

        gt_masks = (
            gt_masks[:, 0]
        )

    elif gt_masks.ndim == 2:

        gt_masks = (
            gt_masks[np.newaxis, ...]
        )

    precision_list = []
    recall_list = []
    f1_list = []

    acd_list = []
    hd_list = []

    # --------------------------------------------------------
    # IMAGE-WISE EVALUATION
    # --------------------------------------------------------

    for i in range(
        len(pred_masks)
    ):

        p, r, f1 = (
            calculate_boundary_metrics_single(
                pred_masks[i],
                gt_masks[i],
                tolerance=tolerance
            )
        )

        precision_list.append(p)
        recall_list.append(r)
        f1_list.append(f1)

        # Contour distances
        acd, hd = (
            calculate_contour_distances(
                pred_masks[i],
                gt_masks[i]
            )
        )

        if np.isfinite(acd):

            acd_list.append(acd)

        if np.isfinite(hd):

            hd_list.append(hd)

    return {

        "Pb": float(
            np.mean(
                precision_list
            )
        ),

        "Rb": float(
            np.mean(
                recall_list
            )
        ),

        "F1b": float(
            np.mean(
                f1_list
            )
        ),

        "ACD": float(
            np.mean(acd_list)
            if len(acd_list) > 0
            else np.nan
        ),

        "HD": float(
            np.mean(hd_list)
            if len(hd_list) > 0
            else np.nan
        )
    }


# ============================================================
# IOU / DICE
# ============================================================

def calculate_iou_dice(
    pred_mask,
    gt_mask
):

    pred_mask = _to_numpy(
        pred_mask
    )

    gt_mask = _to_numpy(
        gt_mask
    )

    # --------------------------------------------------------
    # Use segmentation channel only
    # --------------------------------------------------------

    if pred_mask.ndim == 4:

        pred_mask = (
            pred_mask[:, 0]
        )

    if gt_mask.ndim == 4:

        gt_mask = (
            gt_mask[:, 0]
        )

    pred_mask = (
        pred_mask > 0.5
    )

    gt_mask = (
        gt_mask > 0.5
    )

    if pred_mask.ndim == 2:

        pred_mask = (
            pred_mask[np.newaxis, ...]
        )

    if gt_mask.ndim == 2:

        gt_mask = (
            gt_mask[np.newaxis, ...]
        )

    iou_list = []
    dice_list = []

    for i in range(
        len(pred_mask)
    ):

        p = pred_mask[i]
        g = gt_mask[i]

        intersection = np.logical_and(
            p,
            g
        ).sum()

        union = np.logical_or(
            p,
            g
        ).sum()

        if union > 0:

            iou = (
                intersection
                /
                union
            )

        else:

            iou = 1.0

        denominator = (
            p.sum()
            +
            g.sum()
        )

        if denominator > 0:

            dice = (
                2.0
                * intersection
                /
                denominator
            )

        else:

            dice = 1.0

        iou_list.append(iou)
        dice_list.append(dice)

    return (
        np.mean(iou_list),
        np.mean(dice_list)
    )


# ============================================================
# CONTOUR DISTANCES
# ============================================================

def calculate_contour_distances(
    pred_mask,
    gt_mask
):

    pred_mask = _to_numpy(
        pred_mask
    )

    gt_mask = _to_numpy(
        gt_mask
    )

    pred_mask = (
        np.asarray(pred_mask)
        > 0.5
    )

    gt_mask = (
        np.asarray(gt_mask)
        > 0.5
    )

    pred_contour = extract_boundary(
        pred_mask
    )

    gt_contour = extract_boundary(
        gt_mask
    )

    n_pred = pred_contour.sum()
    n_gt = gt_contour.sum()

    # --------------------------------------------------------
    # BOTH EMPTY
    # --------------------------------------------------------

    if (
        n_pred == 0
        and
        n_gt == 0
    ):

        return (
            0.0,
            0.0
        )

    # --------------------------------------------------------
    # ONE EMPTY
    # --------------------------------------------------------

    if (
        n_pred == 0
        or
        n_gt == 0
    ):

        return (
            np.inf,
            np.inf
        )

    # --------------------------------------------------------
    # DISTANCE MAPS
    # --------------------------------------------------------

    dist_pred = distance_transform_edt(
        ~pred_contour
    )

    dist_gt = distance_transform_edt(
        ~gt_contour
    )

    gt_to_pred = (
        dist_pred[gt_contour]
    )

    pred_to_gt = (
        dist_gt[pred_contour]
    )

    # --------------------------------------------------------
    # AVERAGE CONTOUR DISTANCE
    # --------------------------------------------------------

    all_distances = np.concatenate(
        [
            gt_to_pred,
            pred_to_gt
        ]
    )

    acd = np.mean(
        all_distances
    )

    # --------------------------------------------------------
    # HAUSDORFF DISTANCE
    # --------------------------------------------------------

    hd = max(
        np.max(gt_to_pred),
        np.max(pred_to_gt)
    )

    return (
        float(acd),
        float(hd)
    )


# ============================================================
# CRATER OBJECT EXTRACTION
# ============================================================

def extract_crater_objects(
    mask,
    min_area=5
):

    mask = np.asarray(
        mask
    ).astype(bool)

    structure = np.ones(
        (3, 3),
        dtype=np.uint8
    )

    labeled, num_objects = label(
        mask,
        structure=structure
    )

    objects = []

    slices = find_objects(
        labeled
    )

    for object_id, slc in enumerate(
        slices,
        start=1
    ):

        if slc is None:

            continue

        component_mask = (
            labeled[slc]
            ==
            object_id
        )

        area = int(
            component_mask.sum()
        )

        if area < min_area:

            continue

        y_slice, x_slice = slc

        objects.append({

            "label":
                object_id,

            "mask":
                component_mask,

            "slice":
                slc,

            "area":
                area,

            "x":
                x_slice.start,

            "y":
                y_slice.start
        })

    return objects


# ============================================================
# CRATER MORPHOMETRY
# ============================================================

def calculate_crater_morphometry(
    crater
):

    mask = crater["mask"]

    area = crater["area"]

    # --------------------------------------------------------
    # Equivalent-circle diameter
    # --------------------------------------------------------

    diameter = (
        2.0
        *
        np.sqrt(
            area / np.pi
        )
    )

    # --------------------------------------------------------
    # Boundary
    # --------------------------------------------------------

    boundary = extract_boundary(
        mask
    )

    perimeter = boundary.sum()

    if perimeter > 0:

        circularity = (
            4.0
            *
            np.pi
            *
            area
            /
            (perimeter ** 2)
        )

    else:

        circularity = np.nan

    # --------------------------------------------------------
    # Coordinates
    # --------------------------------------------------------

    coordinates = np.argwhere(
        mask
    )

    if len(coordinates) < 3:

        eccentricity = np.nan

    else:

        covariance = np.cov(
            coordinates,
            rowvar=False
        )

        eigenvalues = (
            np.linalg.eigvalsh(
                covariance
            )
        )

        eigenvalues = np.sort(
            eigenvalues
        )

        lambda_min = eigenvalues[0]
        lambda_max = eigenvalues[-1]

        if lambda_max > 0:

            eccentricity = np.sqrt(
                max(
                    0.0,
                    1.0
                    -
                    (
                        lambda_min
                        /
                        lambda_max
                    )
                )
            )

        else:

            eccentricity = 0.0

    return {

        "diameter":
            diameter,

        "eccentricity":
            eccentricity,

        "circularity":
            circularity
    }


# ============================================================
# OBJECT IOU
# ============================================================

def object_iou(
    pred_obj,
    gt_obj
):

    pred_slice = pred_obj["slice"]
    gt_slice = gt_obj["slice"]

    y_start = min(
        pred_slice[0].start,
        gt_slice[0].start
    )

    y_end = max(
        pred_slice[0].stop,
        gt_slice[0].stop
    )

    x_start = min(
        pred_slice[1].start,
        gt_slice[1].start
    )

    x_end = max(
        pred_slice[1].stop,
        gt_slice[1].stop
    )

    height = (
        y_end - y_start
    )

    width = (
        x_end - x_start
    )

    pred_canvas = np.zeros(
        (height, width),
        dtype=bool
    )

    gt_canvas = np.zeros(
        (height, width),
        dtype=bool
    )

    pred_y0 = (
        pred_slice[0].start
        -
        y_start
    )

    pred_x0 = (
        pred_slice[1].start
        -
        x_start
    )

    gt_y0 = (
        gt_slice[0].start
        -
        y_start
    )

    gt_x0 = (
        gt_slice[1].start
        -
        x_start
    )

    ph, pw = (
        pred_obj["mask"].shape
    )

    gh, gw = (
        gt_obj["mask"].shape
    )

    pred_canvas[
        pred_y0:pred_y0 + ph,
        pred_x0:pred_x0 + pw
    ] = pred_obj["mask"]

    gt_canvas[
        gt_y0:gt_y0 + gh,
        gt_x0:gt_x0 + gw
    ] = gt_obj["mask"]

    intersection = np.logical_and(
        pred_canvas,
        gt_canvas
    ).sum()

    union = np.logical_or(
        pred_canvas,
        gt_canvas
    ).sum()

    if union == 0:

        return 0.0

    return (
        intersection
        /
        union
    )


# ============================================================
# CRATER MATCHING
# ============================================================

def match_craters(
    predicted_objects,
    gt_objects,
    iou_threshold=0.3
):

    matches = []

    used_gt = set()

    # --------------------------------------------------------
    # SORT PREDICTIONS BY AREA
    # --------------------------------------------------------

    predicted_objects = sorted(
        predicted_objects,
        key=lambda x: x["area"],
        reverse=True
    )

    # --------------------------------------------------------
    # ONE-TO-ONE MATCHING
    # --------------------------------------------------------

    for pred_id, pred_obj in enumerate(
        predicted_objects
    ):

        best_iou = 0.0

        best_gt = None

        for gt_id, gt_obj in enumerate(
            gt_objects
        ):

            if gt_id in used_gt:

                continue

            iou = object_iou(
                pred_obj,
                gt_obj
            )

            if iou > best_iou:

                best_iou = iou

                best_gt = gt_id

        if (
            best_gt is not None
            and
            best_iou >= iou_threshold
        ):

            matches.append(
                (
                    pred_id,
                    best_gt,
                    best_iou
                )
            )

            used_gt.add(
                best_gt
            )

    return (
        matches,
        predicted_objects
    )


# ============================================================
# COMPLETE EXISTING EVALUATION
# ============================================================

def get_metrics(
    data,
    craters,
    dim,
    model,
    beta=1,
    threshold=0.5,
    batch_size=8,
    boundary_tolerance=2,
    crater_iou_threshold=0.3,
    min_crater_area=5
):

    """
    Complete segmentation evaluation.

    IMPORTANT
    ---------
    ResGatedUNet output:

        output[:,0] -> segmentation
        output[:,1] -> edge

    The edge channel is used by BCEDiceEdgeLoss.

    Pixel, boundary and connected-component crater
    evaluation use ONLY the segmentation channel.
    """

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "\n********* Custom Metrics *********"
    )

    X, Y = data[0], data[1]

    # ========================================================
    # INPUT
    # ========================================================

    X = torch.as_tensor(
        X,
        dtype=torch.float32
    )

    # ========================================================
    # TARGET
    # ========================================================

    Y = torch.as_tensor(
        Y,
        dtype=torch.float32
    )

    if X.ndim == 3:

        X = X.unsqueeze(1)

    elif (
        X.ndim == 4
        and
        X.shape[-1] == 1
    ):

        X = X.permute(
            0,
            3,
            1,
            2
        )

    if Y.ndim == 3:

        Y = Y.unsqueeze(1)

    elif (
        Y.ndim == 4
        and
        Y.shape[-1] == 1
    ):

        Y = Y.permute(
            0,
            3,
            1,
            2
        )

    # --------------------------------------------------------
    # Normalize target
    # --------------------------------------------------------

    if Y.max() > 1:

        Y = Y / 255.0

    Y = (
        Y > 0.5
    ).float()

    # --------------------------------------------------------
    # REQUIRE TWO TARGET CHANNELS
    # --------------------------------------------------------

    if Y.ndim != 4:

        raise ValueError(
            "Expected Y to have shape "
            "[N,C,H,W]."
        )

    if Y.shape[1] != 2:

        raise ValueError(
            "Expected Y to contain two channels: "
            "segmentation and edge."
        )

    X = X.to(device)

    Y = Y.to(device)

    model = model.to(device)

    model.eval()

    # ========================================================
    # LOSS
    # ========================================================

    criterion = BCEDiceEdgeLoss(
        lambda1=1.0,
        lambda2=1.0,
        lambda3=1.5
    ).to(device)

    total_loss = 0.0

    total_batches = 0

    # --------------------------------------------------------
    # STORE SEGMENTATION ONLY
    # --------------------------------------------------------

    preds_list = []

    targets_list = []

    # ========================================================
    # INFERENCE
    # ========================================================

    with torch.no_grad():

        for i in range(
            0,
            X.shape[0],
            batch_size
        ):

            x_batch = X[
                i:i + batch_size
            ]

            y_batch = Y[
                i:i + batch_size
            ]

            # ------------------------------------------------
            # MODEL OUTPUT
            # ------------------------------------------------

            preds = model(
                x_batch
            )

            if preds.ndim != 4:

                raise ValueError(
                    "Model output must have "
                    "shape [B,C,H,W]."
                )

            if preds.shape[1] != 2:

                raise ValueError(
                    "ResGatedUNet must output "
                    "two channels: segmentation "
                    "and edge."
                )

            # ------------------------------------------------
            # LOSS
            # ------------------------------------------------

            # IMPORTANT:
            #
            # Do NOT write:
            #
            # total_loss, ... = criterion(...)
            # total_loss += total_loss.item()
            #
            # because this overwrites the accumulator.
            #
            batch_loss, bce_loss, dice_loss, edge_loss = (
                criterion(
                    preds,
                    y_batch
                )
            )

            total_loss += (
                batch_loss.item()
            )

            total_batches += 1

            # ------------------------------------------------
            # SIGMOID
            # ------------------------------------------------

            preds_probability = torch.sigmoid(
                preds
            )

            # ------------------------------------------------
            # ONLY SEGMENTATION MAP
            # ------------------------------------------------

            seg_pred = (
                preds_probability[:, 0]
            )

            # ------------------------------------------------
            # ONLY SEGMENTATION TARGET
            # ------------------------------------------------

            seg_true = (
                y_batch[:, 0]
            )

            preds_list.append(
                seg_pred.cpu().numpy()
            )

            targets_list.append(
                seg_true.cpu().numpy()
            )

    # ========================================================
    # AVERAGE LOSS
    # ========================================================

    avg_loss = (
        total_loss / total_batches
        if total_batches > 0
        else 0.0
    )

    # ========================================================
    # COMBINE RESULTS
    # ========================================================

    preds_all = np.concatenate(
        preds_list,
        axis=0
    )

    targets_all = np.concatenate(
        targets_list,
        axis=0
    )

    # ========================================================
    # PIXEL METRICS
    # ========================================================

    pixel_results = calculate_pixel_metrics(
        pred_masks=preds_all,
        gt_masks=targets_all,
        threshold=threshold
    )

    precision = pixel_results["Pseg"]
    recall = pixel_results["Rseg"]
    f1 = pixel_results["F1seg"]
    iou = pixel_results["IoU"]
    dice = pixel_results["Dice"]

    # ========================================================
    # BOUNDARY METRICS
    # ========================================================

    boundary_results = calculate_boundary_metrics(
        pred_masks=preds_all,
        gt_masks=targets_all,
        tolerance=boundary_tolerance
    )

    boundary_precision = (
        boundary_results["Pb"]
    )

    boundary_recall = (
        boundary_results["Rb"]
    )

    boundary_f1 = (
        boundary_results["F1b"]
    )

    average_acd = (
        boundary_results["ACD"]
    )

    average_hd = (
        boundary_results["HD"]
    )

    # ========================================================
    # CRATER-LEVEL OBJECT EVALUATION
    # ========================================================

    y_pred = (
        preds_all > threshold
    ).astype(
        np.uint8
    )

    y_true = (
        targets_all > 0.5
    ).astype(
        np.uint8
    )

    total_tp = 0
    total_fp = 0
    total_fn = 0

    diameter_errors = []
    eccentricity_errors = []
    circularity_errors = []

    # ========================================================
    # IMAGE LOOP
    # ========================================================

    for image_id in range(
        len(y_pred)
    ):

        pred_mask = y_pred[
            image_id
        ]

        gt_mask = y_true[
            image_id
        ]

        # ----------------------------------------------------
        # OBJECT EXTRACTION
        # ----------------------------------------------------

        pred_objects = (
            extract_crater_objects(
                pred_mask,
                min_area=min_crater_area
            )
        )

        gt_objects = (
            extract_crater_objects(
                gt_mask,
                min_area=min_crater_area
            )
        )

        # ----------------------------------------------------
        # ONE-TO-ONE OBJECT MATCHING
        # ----------------------------------------------------

        matches, sorted_pred_objects = (
            match_craters(
                pred_objects,
                gt_objects,
                iou_threshold=(
                    crater_iou_threshold
                )
            )
        )

        tp = len(matches)

        fp = (
            len(pred_objects)
            -
            tp
        )

        fn = (
            len(gt_objects)
            -
            tp
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

        # ----------------------------------------------------
        # MORPHOMETRY
        # ----------------------------------------------------

        for (
            pred_id,
            gt_id,
            match_iou
        ) in matches:

            pred_crater = (
                calculate_crater_morphometry(
                    sorted_pred_objects[
                        pred_id
                    ]
                )
            )

            gt_crater = (
                calculate_crater_morphometry(
                    gt_objects[
                        gt_id
                    ]
                )
            )

            # Diameter
            if (
                np.isfinite(
                    pred_crater["diameter"]
                )
                and
                np.isfinite(
                    gt_crater["diameter"]
                )
            ):

                diameter_errors.append(
                    abs(
                        pred_crater["diameter"]
                        -
                        gt_crater["diameter"]
                    )
                )

            # Eccentricity
            if (
                np.isfinite(
                    pred_crater["eccentricity"]
                )
                and
                np.isfinite(
                    gt_crater["eccentricity"]
                )
            ):

                eccentricity_errors.append(
                    abs(
                        pred_crater[
                            "eccentricity"
                        ]
                        -
                        gt_crater[
                            "eccentricity"
                        ]
                    )
                )

            # Circularity
            if (
                np.isfinite(
                    pred_crater["circularity"]
                )
                and
                np.isfinite(
                    gt_crater["circularity"]
                )
            ):

                circularity_errors.append(
                    abs(
                        pred_crater[
                            "circularity"
                        ]
                        -
                        gt_crater[
                            "circularity"
                        ]
                    )
                )

    # ========================================================
    # CRATER PRECISION / RECALL / F1
    # ========================================================

    crater_precision = (
        total_tp
        /
        (
            total_tp
            +
            total_fp
        )
        if (
            total_tp
            +
            total_fp
        ) > 0
        else 0.0
    )

    crater_recall = (
        total_tp
        /
        (
            total_tp
            +
            total_fn
        )
        if (
            total_tp
            +
            total_fn
        ) > 0
        else 0.0
    )

    crater_f1 = (
        2.0
        * crater_precision
        * crater_recall
        /
        (
            crater_precision
            +
            crater_recall
        )
        if (
            crater_precision
            +
            crater_recall
        ) > 0
        else 0.0
    )

    # ========================================================
    # MORPHOMETRIC ERRORS
    # ========================================================

    diameter_mae = (
        np.mean(
            diameter_errors
        )
        if len(diameter_errors) > 0
        else np.nan
    )

    diameter_std = (
        np.std(
            diameter_errors
        )
        if len(diameter_errors) > 1
        else np.nan
    )

    eccentricity_mae = (
        np.mean(
            eccentricity_errors
        )
        if len(eccentricity_errors) > 0
        else np.nan
    )

    eccentricity_std = (
        np.std(
            eccentricity_errors
        )
        if len(eccentricity_errors) > 1
        else np.nan
    )

    circularity_mae = (
        np.mean(
            circularity_errors
        )
        if len(circularity_errors) > 0
        else np.nan
    )

    circularity_std = (
        np.std(
            circularity_errors
        )
        if len(circularity_errors) > 1
        else np.nan
    )

    # ========================================================
    # PRINT
    # ========================================================

    print(
        "\n=============================="
    )

    print(
        "Evaluation Results"
    )

    print(
        "=============================="
    )

    print(
        f"Threshold: {threshold}"
    )

    print(
        f"Total Loss: {avg_loss:.6f}"
    )

    # --------------------------------------------------------
    # PIXEL
    # --------------------------------------------------------

    print(
        "\nPixel-level:"
    )

    print(
        f"Pseg:  {precision:.4f}"
    )

    print(
        f"Rseg:  {recall:.4f}"
    )

    print(
        f"F1seg: {f1:.4f}"
    )

    print(
        f"IoU:   {iou:.4f}"
    )

    print(
        f"Dice:  {dice:.4f}"
    )

    # --------------------------------------------------------
    # BOUNDARY
    # --------------------------------------------------------

    print(
        "\nBoundary-level:"
    )

    print(
        f"Pb:  {boundary_precision:.4f}"
    )

    print(
        f"Rb:  {boundary_recall:.4f}"
    )

    print(
        f"F1b: {boundary_f1:.4f}"
    )

    print(
        f"ACD: {average_acd:.4f} pixels"
    )

    print(
        f"HD:  {average_hd:.4f} pixels"
    )

    print(
        f"Boundary tolerance: "
        f"{boundary_tolerance} pixels"
    )

    # --------------------------------------------------------
    # CRATER
    # --------------------------------------------------------

    print(
        "\nCrater-level:"
    )

    print(
        f"Pc: {crater_precision:.4f}"
    )

    print(
        f"Rc: {crater_recall:.4f}"
    )

    print(
        f"F1c: {crater_f1:.4f}"
    )

    print(
        f"TP: {total_tp}"
    )

    print(
        f"FP: {total_fp}"
    )

    print(
        f"FN: {total_fn}"
    )

    # --------------------------------------------------------
    # MORPHOMETRY
    # --------------------------------------------------------

    print(
        "\nMorphometry:"
    )

    print(
        f"Diameter MAE: "
        f"{diameter_mae:.4f}"
    )

    print(
        f"Diameter STD: "
        f"{diameter_std:.4f}"
    )

    print(
        f"Eccentricity MAE: "
        f"{eccentricity_mae:.4f}"
    )

    print(
        f"Eccentricity STD: "
        f"{eccentricity_std:.4f}"
    )

    print(
        f"Circularity MAE: "
        f"{circularity_mae:.4f}"
    )

    print(
        f"Circularity STD: "
        f"{circularity_std:.4f}"
    )

    # ========================================================
    # RETURN
    # ========================================================

    return {

        # ----------------------------------------------------
        # LOSS
        # ----------------------------------------------------

        "loss":
            avg_loss,

        # ----------------------------------------------------
        # PIXEL
        # ----------------------------------------------------

        "Pseg":
            precision,

        "Rseg":
            recall,

        "F1seg":
            f1,

        "IoU":
            iou,

        "Dice":
            dice,

        # ----------------------------------------------------
        # BOUNDARY
        # ----------------------------------------------------

        "Pb":
            boundary_precision,

        "Rb":
            boundary_recall,

        "F1b":
            boundary_f1,

        "ACD":
            average_acd,

        "HD":
            average_hd,

        # ----------------------------------------------------
        # CRATER
        # ----------------------------------------------------

        "Pc":
            crater_precision,

        "Rc":
            crater_recall,

        "F1c":
            crater_f1,

        "crater_tp":
            total_tp,

        "crater_fp":
            total_fp,

        "crater_fn":
            total_fn,

        # ----------------------------------------------------
        # MORPHOMETRY
        # ----------------------------------------------------

        "diameter_mae":
            diameter_mae,

        "diameter_std":
            diameter_std,

        "eccentricity_mae":
            eccentricity_mae,

        "eccentricity_std":
            eccentricity_std,

        "circularity_mae":
            circularity_mae,

        "circularity_std":
            circularity_std
    }
