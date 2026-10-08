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

    def forward(self, y_pred, y_true):

        seg_pred = y_pred[:, 0:1]
        edge_pred = y_pred[:, 1:2]

        seg_true = y_true[:, 0:1]
        edge_true = y_true[:, 1:2]

        # BCE segmentation loss
        bce_loss = self.bce(
            seg_pred,
            seg_true
        )

        # Dice loss
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
            1.0 - dice_score.mean()
        )

        # Edge BCE
        edge_loss = self.bce(
            edge_pred,
            edge_true
        )

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
# BOUNDARY EXTRACTION
# ============================================================

def extract_boundary(mask):

    mask = np.asarray(mask).astype(bool)

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

    return mask & (~eroded)


# ============================================================
# SINGLE-IMAGE BOUNDARY METRICS
# ============================================================

def calculate_boundary_metrics_single(
    pred_mask,
    gt_mask,
    tolerance=2
):

    if torch.is_tensor(pred_mask):
        pred_mask = (
            pred_mask.detach()
            .cpu()
            .numpy()
        )

    if torch.is_tensor(gt_mask):
        gt_mask = (
            gt_mask.detach()
            .cpu()
            .numpy()
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

    # Both empty
    if n_pred == 0 and n_gt == 0:
        return 1.0, 1.0, 1.0

    # One empty
    if n_pred == 0 or n_gt == 0:
        return 0.0, 0.0, 0.0

    # Distance from every pixel to predicted boundary
    dist_pred = distance_transform_edt(
        ~pred_boundary
    )

    # Distance from every pixel to GT boundary
    dist_gt = distance_transform_edt(
        ~gt_boundary
    )

    # Predicted boundary precision
    pred_distances = (
        dist_gt[pred_boundary]
    )

    boundary_precision = np.mean(
        pred_distances <= tolerance
    )

    # GT boundary recall
    gt_distances = (
        dist_pred[gt_boundary]
    )

    boundary_recall = np.mean(
        gt_distances <= tolerance
    )

    denominator = (
        boundary_precision
        + boundary_recall
    )

    if denominator > 0:
        boundary_f1 = (
            2.0
            * boundary_precision
            * boundary_recall
            / denominator
        )
    else:
        boundary_f1 = 0.0

    return (
        boundary_precision,
        boundary_recall,
        boundary_f1
    )


# ============================================================
# BATCH BOUNDARY METRICS
# ============================================================

def calculate_boundary_metrics(
    pred_masks,
    gt_masks,
    tolerance=2
):

    if torch.is_tensor(pred_masks):
        pred_masks = (
            pred_masks.detach()
            .cpu()
            .numpy()
        )

    if torch.is_tensor(gt_masks):
        gt_masks = (
            gt_masks.detach()
            .cpu()
            .numpy()
        )

    pred_masks = np.asarray(
        pred_masks
    )

    gt_masks = np.asarray(
        gt_masks
    )

    if pred_masks.ndim == 4:
        pred_masks = pred_masks[:, 0]

    elif pred_masks.ndim == 2:
        pred_masks = pred_masks[np.newaxis, ...]

    if gt_masks.ndim == 4:
        gt_masks = gt_masks[:, 0]

    elif gt_masks.ndim == 2:
        gt_masks = gt_masks[np.newaxis, ...]

    precision_list = []
    recall_list = []
    f1_list = []

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

    return (
        np.mean(precision_list),
        np.mean(recall_list),
        np.mean(f1_list)
    )


# ============================================================
# IOU / DICE
# ============================================================

def calculate_iou_dice(
    pred_mask,
    gt_mask
):

    if torch.is_tensor(pred_mask):
        pred_mask = (
            pred_mask.detach()
            .cpu()
            .numpy()
        )

    if torch.is_tensor(gt_mask):
        gt_mask = (
            gt_mask.detach()
            .cpu()
            .numpy()
        )

    pred_mask = (
        np.asarray(pred_mask) > 0.5
    )

    gt_mask = (
        np.asarray(gt_mask) > 0.5
    )

    if pred_mask.ndim == 2:
        pred_mask = pred_mask[np.newaxis, ...]

    if gt_mask.ndim == 2:
        gt_mask = gt_mask[np.newaxis, ...]

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
                intersection / union
            )
        else:
            iou = 1.0

        denominator = (
            p.sum() + g.sum()
        )

        if denominator > 0:
            dice = (
                2.0 * intersection
                / denominator
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

    if torch.is_tensor(pred_mask):
        pred_mask = (
            pred_mask.detach()
            .cpu()
            .numpy()
        )

    if torch.is_tensor(gt_mask):
        gt_mask = (
            gt_mask.detach()
            .cpu()
            .numpy()
        )

    pred_mask = (
        np.asarray(pred_mask) > 0.5
    )

    gt_mask = (
        np.asarray(gt_mask) > 0.5
    )

    pred_contour = extract_boundary(
        pred_mask
    )

    gt_contour = extract_boundary(
        gt_mask
    )

    n_pred = pred_contour.sum()
    n_gt = gt_contour.sum()

    if n_pred == 0 and n_gt == 0:
        return 0.0, 0.0

    if n_pred == 0 or n_gt == 0:
        return np.inf, np.inf

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

    all_distances = np.concatenate(
        [
            gt_to_pred,
            pred_to_gt
        ]
    )

    acd = np.mean(
        all_distances
    )

    hd = max(
        np.max(gt_to_pred),
        np.max(pred_to_gt)
    )

    return acd, hd


# ============================================================
# CRATER OBJECT EXTRACTION
# ============================================================

def extract_crater_objects(
    mask,
    min_area=5
):

    mask = np.asarray(mask).astype(
        bool
    )

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
            labeled[slc] == object_id
        )

        area = int(
            component_mask.sum()
        )

        if area < min_area:
            continue

        y_slice, x_slice = slc

        objects.append(
            {
                "label": object_id,
                "mask": component_mask,
                "slice": slc,
                "area": area,
                "x": x_slice.start,
                "y": y_slice.start
            }
        )

    return objects


# ============================================================
# CRATER MORPHOMETRY
# ============================================================

def calculate_crater_morphometry(
    crater
):

    mask = crater["mask"]

    area = crater["area"]

    # Equivalent-circle diameter
    diameter = (
        2.0
        * np.sqrt(
            area / np.pi
        )
    )

    # Boundary
    boundary = extract_boundary(
        mask
    )

    perimeter = boundary.sum()

    if perimeter > 0:

        circularity = (
            4.0
            * np.pi
            * area
            / (perimeter ** 2)
        )

    else:

        circularity = np.nan

    # Coordinates
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

        eigenvalues = np.linalg.eigvalsh(
            covariance
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
                    - (
                        lambda_min
                        / lambda_max
                    )
                )
            )

        else:

            eccentricity = 0.0

    return {
        "diameter": diameter,
        "eccentricity": eccentricity,
        "circularity": circularity
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

    height = y_end - y_start
    width = x_end - x_start

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
        - y_start
    )

    pred_x0 = (
        pred_slice[1].start
        - x_start
    )

    gt_y0 = (
        gt_slice[0].start
        - y_start
    )

    gt_x0 = (
        gt_slice[1].start
        - x_start
    )

    ph, pw = pred_obj["mask"].shape
    gh, gw = gt_obj["mask"].shape

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

    return intersection / union


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

    # Preserve the original implementation:
    # predicted objects are matched in descending
    # area order.

    predicted_objects = sorted(
        predicted_objects,
        key=lambda x: x["area"],
        reverse=True
    )

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
            and best_iou >= iou_threshold
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

    return matches, predicted_objects


# ============================================================
# COMPLETE EVALUATION
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

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "\n********* Custom Metrics *********"
    )

    X, Y = data[0], data[1]

    X = torch.as_tensor(
        X,
        dtype=torch.float32
    )

    Y = torch.as_tensor(
        Y,
        dtype=torch.float32
    )

    # --------------------------------------------------------
    # Input shape
    # --------------------------------------------------------

    if X.ndim == 3:

        X = X.unsqueeze(1)

    elif (
        X.ndim == 4
        and X.shape[-1] == 1
    ):

        X = X.permute(
            0, 3, 1, 2
        )

    # --------------------------------------------------------
    # Target shape
    # --------------------------------------------------------

    if Y.ndim == 3:

        Y = Y.unsqueeze(1)

    elif (
        Y.ndim == 4
        and Y.shape[-1] == 1
    ):

        Y = Y.permute(
            0, 3, 1, 2
        )

    if Y.max() > 1:
        Y = Y / 255.0

    Y = (
        Y > 0.5
    ).float()

    if Y.ndim != 4:
        raise ValueError(
            "Expected Y to have shape [N,C,H,W]."
        )

    if Y.shape[1] != 2:
        raise ValueError(
            "Expected Y to contain "
            "segmentation and edge channels."
        )

    X = X.to(device)
    Y = Y.to(device)

    model = model.to(device)
    model.eval()

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    lambda1 = 1.0
    lambda2 = 1.0
    lambda3 = 1.5

    criterion = BCEDiceEdgeLoss(
        lambda1=lambda1,
        lambda2=lambda2,
        lambda3=lambda3
    ).to(device)

    total_loss = 0.0
    total_batches = 0

    preds_list = []
    targets_list = []

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

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

            preds = model(
                x_batch
            )

            if preds.ndim != 4:
                raise ValueError(
                    "Model output must "
                    "have shape [B,C,H,W]."
                )

            if preds.shape[1] != 2:
                raise ValueError(
                    "Model must output "
                    "segmentation and edge channels."
                )

            # Important:
            # use a separate variable for batch loss
            # so the accumulator is not overwritten.
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

            preds_sigmoid = torch.sigmoid(
                preds
            )

            seg_pred = (
                preds_sigmoid[:, 0]
            )

            seg_true = (
                y_batch[:, 0]
            )

            preds_list.append(
                seg_pred.cpu().numpy()
            )

            targets_list.append(
                seg_true.cpu().numpy()
            )

    # --------------------------------------------------------
    # Average loss
    # --------------------------------------------------------

    avg_loss = (
        total_loss / total_batches
        if total_batches > 0
        else 0.0
    )

    preds_all = np.concatenate(
        preds_list,
        axis=0
    )

    targets_all = np.concatenate(
        targets_list,
        axis=0
    )

    # --------------------------------------------------------
    # Binary predictions
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Pixel-level metrics
    # --------------------------------------------------------

    y_pred_flat = y_pred.flatten()
    y_true_flat = y_true.flatten()

    precision = precision_score(
        y_true_flat,
        y_pred_flat,
        zero_division=0
    )

    recall = recall_score(
        y_true_flat,
        y_pred_flat,
        zero_division=0
    )

    f1 = f1_score(
        y_true_flat,
        y_pred_flat,
        zero_division=0
    )

    iou = jaccard_score(
        y_true_flat,
        y_pred_flat,
        zero_division=0
    )

    intersection = np.sum(
        y_pred_flat * y_true_flat
    )

    dice = (
        2.0 * intersection
        + 1e-8
    ) / (
        np.sum(y_pred_flat)
        +
        np.sum(y_true_flat)
        +
        1e-8
    )

    # --------------------------------------------------------
    # Boundary / crater / morphometry
    # --------------------------------------------------------

    boundary_precision_list = []
    boundary_recall_list = []
    boundary_f1_list = []

    acd_list = []
    hd_list = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    diameter_errors = []
    eccentricity_errors = []
    circularity_errors = []

    for image_id in range(
        len(y_pred)
    ):

        pred_mask = y_pred[
            image_id
        ]

        gt_mask = y_true[
            image_id
        ]

        # Boundary metrics
        bp, br, bf1 = (
            calculate_boundary_metrics(
                pred_mask,
                gt_mask,
                tolerance=boundary_tolerance
            )
        )

        boundary_precision_list.append(bp)
        boundary_recall_list.append(br)
        boundary_f1_list.append(bf1)

        # Contour distances
        acd, hd = (
            calculate_contour_distances(
                pred_mask,
                gt_mask
            )
        )

        if np.isfinite(acd):
            acd_list.append(acd)

        if np.isfinite(hd):
            hd_list.append(hd)

        # Crater objects
        pred_objects = extract_crater_objects(
            pred_mask,
            min_area=min_crater_area
        )

        gt_objects = extract_crater_objects(
            gt_mask,
            min_area=min_crater_area
        )

        matches, sorted_pred_objects = (
            match_craters(
                pred_objects,
                gt_objects,
                iou_threshold=crater_iou_threshold
            )
        )

        tp = len(matches)

        fp = (
            len(pred_objects)
            - tp
        )

        fn = (
            len(gt_objects)
            - tp
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Morphometry
        #
        # match_craters() sorts predictions by area,
        # therefore use the same sorted list here.
        for pred_id, gt_id, match_iou in matches:

            pred_crater = (
                calculate_crater_morphometry(
                    sorted_pred_objects[pred_id]
                )
            )

            gt_crater = (
                calculate_crater_morphometry(
                    gt_objects[gt_id]
                )
            )

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
                        pred_crater["eccentricity"]
                        -
                        gt_crater["eccentricity"]
                    )
                )

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
                        pred_crater["circularity"]
                        -
                        gt_crater["circularity"]
                    )
                )

    # --------------------------------------------------------
    # Crater-level metrics
    # --------------------------------------------------------

    crater_precision = (
        total_tp
        /
        (
            total_tp
            + total_fp
            + 1e-8
        )
    )

    crater_recall = (
        total_tp
        /
        (
            total_tp
            + total_fn
            + 1e-8
        )
    )

    crater_f1 = (
        2.0
        * crater_precision
        * crater_recall
        /
        (
            crater_precision
            + crater_recall
            + 1e-8
        )
    )

    # --------------------------------------------------------
    # Morphometry
    # --------------------------------------------------------

    diameter_mae = (
        np.mean(diameter_errors)
        if len(diameter_errors) > 0
        else np.nan
    )

    diameter_std = (
        np.std(diameter_errors)
        if len(diameter_errors) > 1
        else np.nan
    )

    eccentricity_mae = (
        np.mean(eccentricity_errors)
        if len(eccentricity_errors) > 0
        else np.nan
    )

    eccentricity_std = (
        np.std(eccentricity_errors)
        if len(eccentricity_errors) > 1
        else np.nan
    )

    circularity_mae = (
        np.mean(circularity_errors)
        if len(circularity_errors) > 0
        else np.nan
    )

    circularity_std = (
        np.std(circularity_errors)
        if len(circularity_errors) > 1
        else np.nan
    )

    # --------------------------------------------------------
    # Boundary averages
    # --------------------------------------------------------

    boundary_precision = (
        np.mean(
            boundary_precision_list
        )
        if len(boundary_precision_list) > 0
        else np.nan
    )

    boundary_recall = (
        np.mean(
            boundary_recall_list
        )
        if len(boundary_recall_list) > 0
        else np.nan
    )

    boundary_f1 = (
        np.mean(
            boundary_f1_list
        )
        if len(boundary_f1_list) > 0
        else np.nan
    )

    average_acd = (
        np.mean(acd_list)
        if len(acd_list) > 0
        else np.nan
    )

    average_hd = (
        np.mean(hd_list)
        if len(hd_list) > 0
        else np.nan
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

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
        f"BCE + Dice + Edge Loss: "
        f"{avg_loss:.6f}"
    )

    print(
        "\nPixel-level:"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1:        {f1:.4f}"
    )

    print(
        f"IoU:       {iou:.4f}"
    )

    print(
        f"Dice:      {dice:.4f}"
    )

    print(
        "\nBoundary-level:"
    )

    print(
        f"Precision: {boundary_precision:.4f}"
    )

    print(
        f"Recall:    {boundary_recall:.4f}"
    )

    print(
        f"F1:        {boundary_f1:.4f}"
    )

    print(
        f"Tolerance: {boundary_tolerance} pixels"
    )

    print(
        f"ACD:       {average_acd:.4f} pixels"
    )

    print(
        f"HD:        {average_hd:.4f} pixels"
    )

    print(
        "\nCrater-level:"
    )

    print(
        f"TP:        {total_tp}"
    )

    print(
        f"FP:        {total_fp}"
    )

    print(
        f"FN:        {total_fn}"
    )

    print(
        f"Precision: {crater_precision:.4f}"
    )

    print(
        f"Recall:    {crater_recall:.4f}"
    )

    print(
        f"F1:        {crater_f1:.4f}"
    )

    print(
        "\nMorphometry:"
    )

    print(
        f"Diameter MAE:       {diameter_mae:.4f}"
    )

    print(
        f"Diameter STD:       {diameter_std:.4f}"
    )

    print(
        f"Eccentricity MAE:   {eccentricity_mae:.4f}"
    )

    print(
        f"Eccentricity STD:   {eccentricity_std:.4f}"
    )

    print(
        f"Circularity MAE:    {circularity_mae:.4f}"
    )

    print(
        f"Circularity STD:    {circularity_std:.4f}"
    )

    return {
        "loss": avg_loss,

        "pixel_precision": precision,
        "pixel_recall": recall,
        "pixel_f1": f1,
        "pixel_iou": iou,
        "pixel_dice": dice,

        "boundary_precision": boundary_precision,
        "boundary_recall": boundary_recall,
        "boundary_f1": boundary_f1,

        "acd": average_acd,
        "hd": average_hd,

        "crater_tp": total_tp,
        "crater_fp": total_fp,
        "crater_fn": total_fn,

        "crater_precision": crater_precision,
        "crater_recall": crater_recall,
        "crater_f1": crater_f1,

        "diameter_mae": diameter_mae,
        "diameter_std": diameter_std,

        "eccentricity_mae": eccentricity_mae,
        "eccentricity_std": eccentricity_std,

        "circularity_mae": circularity_mae,
        "circularity_std": circularity_std
    }
