import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import h5py
import pandas as pd
import cv2

def preprocess(Data, dim, low=0.1, hi=1.0):

    for key in Data:
        X = Data[key][0]
        Y = Data[key][1]

        # -----------------------------
        # INPUT (X)
        # -----------------------------
        X = X.reshape(len(X), dim, dim)
        X = X[:, np.newaxis, :, :]
        X = X.astype(np.float32) / 255.0

        for i in range(X.shape[0]):
            img = X[i, 0]
            mask = img > 0

            if np.any(mask):
                minn = img[mask].min()
                maxx = img[mask].max()

                if maxx > minn:
                    img[mask] = low + (img[mask] - minn) * (hi - low) / (maxx - minn)

            X[i, 0] = img

        # -----------------------------
        # SEGMENTATION MASK
        # -----------------------------
        Y = Y.reshape(len(Y), dim, dim)
        Y = (Y > 0).astype(np.float32)

        # -----------------------------
        # EDGE MAP
        # -----------------------------
        edge_maps = np.zeros_like(Y)

        for i in range(len(Y)):
            edge_maps[i] = get_gradient_edge_np(Y[i])

        # -----------------------------
        # STACK → (N,2,H,W)
        # -----------------------------
        Y = np.stack([Y, edge_maps], axis=1)

        Data[key][0] = X.astype(np.float32)
        Data[key][1] = Y.astype(np.float32)

    # Debug check
    print("X shape:", Data['train'][0].shape)
    print("Y shape:", Data['train'][1].shape)
    print("Unique Y:", np.unique(Data['train'][1]))

def get_gradient_edge_np(img):

    img = img.astype(np.float32)

    if img.max() <= 1.0:
        img = img * 255.0

    img = img.astype(np.uint8)

    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

    grad = np.sqrt(gx**2 + gy**2)

    if grad.max() == 0:
        return np.zeros_like(img, dtype=np.float32)

    grad = grad / grad.max()

    thresh = 0.1
    grad = (grad > thresh).astype(np.float32)

    kernel = np.ones((3,3), np.uint8)

    grad = cv2.dilate(
        grad,
        kernel,
        iterations=1
    )

    return grad.astype(np.float32)
