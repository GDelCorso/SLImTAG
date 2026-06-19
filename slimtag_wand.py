"""
Functions for magic wands wand methods.

For each method, two functions are required:

(1) <method>_preprocessing(img: np.array with dtype=uint8,
                           *args): dict | None

apply potential preprocessing computation to a single image. For example:
- for region growing, compute the RGB, grayscale, and edge matrices, and return them
- for SAM, embed image into the model (passed as one of the args) and return None

(2) <method>_inference(img: np.array with dtype=uint8,
                       pt: pair (x,y),
                       parameters: dict,
                       preprocessing: dict | None = None,
                       **kwargs): np.array with dtype=bool

produce a mask starting from img conditioned to the point pt.
parameters is a dict with necessary parameters (e.g. threshold)
preprocessing is the result of <method>_preprocessing
"""
import numpy as np
from scipy import ndimage
from scipy.special import expit # sigmoid

def region_growing_preprocessing(image):
    # REGION GROWING (need 0-255 matrices BUT with float dtype)
    # grayscale: mimic PIL's convert("L"), which uses the ITU-R 601-2 luma transform
    img_gray = (0.299*image[..., 0] + 0.587*image[..., 1] + 0.114*image[..., 2]).astype(np.float32)
    # RGB: just cast to expected dtype
    img_rgb = image.astype(np.float32)
    # edge detection (computed on grayscale)
    gx = ndimage.sobel(img_gray, axis=1)
    gy = ndimage.sobel(img_gray, axis=0)
    img_grad = np.hypot(gx, gy)
    # normalize gradient matrix to [0,1] for stability
    p = np.percentile(img_grad, 99)
    img_grad = np.clip(img_grad / (p + 1e-8), 0, 1)
    return {"rgb": img_rgb, "gray": img_gray, "edge": img_grad}

def region_growing_inference(image, point, parameters, # model_inference mandatory args
                             preprocessing=None): # model_inference mandatory kwarg
    """
    Interactive region growing segmentation with optional RGB/grayscale similarity,
    robust seed estimation, edge-aware filtering, morphological cleanup, and
    selective hole filling.
    
    Parameters keys expected
    ------------------------
    threshold : float (0–1)
        Similarity threshold for region growing (0 = strict, 1 = permissive).
    
    robust : bool
        Uses robust 3×3 trimmed mean seed estimation to reduce noise sensitivity.
    
    use_RGB : bool
        If True uses RGB Euclidean distance, otherwise grayscale intensity.
    
    use_edges : bool
        Adds gradient-based edge stopping constraint.
    
    grad_edge : float (0–1)
        Maximum normalized gradient allowed for region growing (lower = stricter).
    
    erosion : bool
        Applies morphological opening to break thin connections before labeling.
    
    fill_hole : bool
        Enables selective filling of small internal holes.
    
    max_hole_size : int
        Maximum pixel area of holes to fill.
    """
    # default values if parameters does not contain those
    thres = parameters.get("threshold", 0.15)
    robust = parameters.get("robust", True)
    use_RGB = parameters.get("use_RGB", True)
    use_edges = parameters.get("use_edges", True)
    max_grad_edge = parameters.get("grad_edge", 0.5)
    erosion = parameters.get("erosion", False)
    fill_hole = parameters.get("fill_hole", True)
    max_hole_size = parameters.get("max_hole_size", 100)

    x = int(point[0])
    y = int(point[1])
    
    if preprocessing is None:
        preprocessing = region_growing_preprocessing(image)

    # select the seed value
    if robust:
        # trimmed robust local estimator
        if use_RGB:
            patch_rgb = preprocessing["rgb"][
                max(0, y-1):min(preprocessing["rgb"].shape[0], y+2),
                max(0, x-1):min(preprocessing["rgb"].shape[1], x+2)
            ]
            vals_rgb = patch_rgb.reshape(-1, 3)
            med_rgb = np.median(vals_rgb, axis=0)
            dist_rgb = np.sum((vals_rgb - med_rgb)**2, axis=1)
            keep_idx_rgb = np.argsort(dist_rgb)[:5]
            seed_val_rgb = vals_rgb[keep_idx_rgb].mean(axis=0)
        else: # grayscale
            patch_gray = preprocessing["gray"][
                max(0, y-1):min(preprocessing["gray"].shape[0], y+2),
                max(0, x-1):min(preprocessing["gray"].shape[1], x+2)
            ].astype(np.float32)
            vals_gray = patch_gray.flatten()
            med_gray = np.median(vals_gray)
            dist_gray = np.abs(vals_gray - med_gray)
            keep_idx = np.argsort(dist_gray)[:5]
            seed_val_gray = vals_gray[keep_idx].mean()
    else:
        # pixel-perfect estimator
        seed_val_gray = preprocessing["gray"][y, x]
        seed_val_rgb = preprocessing["rgb"][y, x]
    
    # Compute similarity score
    if use_RGB:
        rgb = preprocessing["rgb"]
        dr = rgb[..., 0] - seed_val_rgb[0]
        dg = rgb[..., 1] - seed_val_rgb[1]
        db = rgb[..., 2] - seed_val_rgb[2]
        diff = np.sqrt(dr*dr + dg*dg + db*db)
    else: # grayscale
        diff = np.abs(preprocessing["gray"] - seed_val_gray) # in FLOAT 32
    
    # apply region growing
    mask = diff < round(255 * thres)
    
    # post-processing: integrate edge information
    if use_edges:
        # integrates edges by removing all area with grad greater thatn max_grad_edge
        mask = mask & (preprocessing["edge"] < max_grad_edge)
    
    # post-processing: small erosion+dilation to break small connections between zones
    if erosion:
        mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
        
    # extraction of single connected component
    labeled, _ = ndimage.label(mask)
    seed_label = labeled[y, x]
    if seed_label == 0:
        return np.zeros_like(mask).astype(bool)
    region = (labeled == seed_label)
    
    # post-processing: fill small holes
    if fill_hole:
        # holes = internal background of the region
        holes = ndimage.binary_fill_holes(region) & (~region)
        # label holes
        hole_labels, _ = ndimage.label(holes)
        # compute hole sizes
        hole_sizes = np.bincount(hole_labels.ravel())
        # keep only small holes
        small_idx = np.where(hole_sizes <= max_hole_size)[0]
        small_idx = small_idx[small_idx!=0]
        # skip label 0 (not-holes) in the remote case that it is small too
        small_holes = np.isin(
            hole_labels,
            small_idx
        )
        # fill only small holes
        region = region | small_holes

    return region

def sam_preprocessing(image, model):
    model.set_image(image)
    return None

def sam_inference(image, point, parameters, # model_inference mandatory args
                  model, # sam_inference arg
                  preprocessing=None, # model_inference mandatory kwarg (not used for SAM)
                  pt_labels=np.array([1]), multipoint=False): # sam_inference kwargs
    # default values if parameters does not contain those
    thres = parameters.get("threshold", 0.5)
    masks, scores, _ = model.predict(np.array(point),
                                     pt_labels,
                                     multimask_output=not multipoint,
                                     return_logits=True)

    masks = expit(masks) > thres
    i = np.argmax(scores)
    return masks[i]