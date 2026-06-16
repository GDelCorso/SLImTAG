# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 08:58:50 2026

@author: Giulio Del Corso
"""

#### Libraries
import os
import numpy as np
from PIL import Image, ImageDraw

from slimtag_utils import adjust_image

import json

from skopt import gp_minimize
from skopt.space import Real

#### Main class - Bayesian optimizer
class BayesianOptimization():
    
    def __init__(self, image_folder, model_inference, model_preprocessing, mask_folder=None):
        self.image_folder = image_folder
        self.mask_folder = mask_folder if mask_folder is not None else image_folder
        
        self.model_inference = model_inference
        self.model_preprocessing = model_preprocessing
        
        # Load images and masks
        self.image_list, self.mask_list, self.label_list = self.load_folder()
        
        
        
    def load_folder(self):
        list_image_names = os.listdir(self.image_folder)
        list_mask_names = os.listdir(self.mask_folder)
        
        # Selection
        valid_ext = {".jpg", ".jpeg", ".png"}
        
        images = {}
        masks = {}
        
        for fname in list_image_names:
            stem, ext = os.path.splitext(fname)
            if ext.lower() not in valid_ext:
                continue
            images[stem] = fname

        for fname in list_mask_names:
            stem, ext = os.path.splitext(fname)
            if ext.lower() not in valid_ext:
                continue
            if ext == ".png" and stem.endswith("_mask"):
                base_name = stem[:-5]
                masks[base_name] = fname
        
        # Keep coupled
        common_names = sorted(set(images.keys()) & set(masks.keys()))
        if len(common_names) == 0:
            raise RuntimeError("empty list of images/masks pairs in folder")
        
        image_list = []
        mask_list = []
        label_list = []
        
        for name in common_names:
            image_path = os.path.join(self.image_folder, images[name])
            mask_path = os.path.join(self.mask_folder, masks[name])
            image_list.append(np.array(Image.open(image_path).convert("RGB")))
            mask = Image.open(mask_path)
            mask_list.append(np.array(mask.convert("P")))
            try:
                names_json = json.loads(mask.text["labels"])
                names = {int(l): names_json[l] for l in names_json.keys()}
            except KeyError:
                names = {}
            label_list.append(names)
        return image_list, mask_list, label_list
    

    def sample(self, mask, n_points):
        """
        Given a boolean numpy array mask, return n_points pairs (x,y)
        chosen randomly among the ones with mask[y, x] == True
        """
        ys, xs = np.where(mask)
        if len(xs) == 0: # this should not happen given the checks in objective()
            raise ValueError("Provided mask has only False values")
        coords = np.stack([xs, ys], axis=1)
        n_points = min(n_points, len(coords)) # in case there are too few points with label
        idx = np.random.choice(len(coords), size=n_points, replace=False)
        return coords[idx]

    def objective(self, parameters, mask_label, n_points=10):
        """
        Compute IoU of predicted mask and ground truth mask.
        
        Image adjustments are applied before calling the model inference.
        A total of n_points predicted masks are computed starting from randomly
        chosen points inside the mask corresponding to mask_label, and the
        average IoU is returned.
        """
        # apply adjustments given by parameters
        # here we assume: parameters[0] = brightness, parameters[1] = contrast,
        # parameters[2] = gamma, all ranging from -100 to 100
        # parameters[3] = wand_threshold, parameters[4] = edge_grad, both from 0.0 to 1.0
        param_dict = dict(zip(["brightness", "contrast", "gamma", "threshold", "grad_edge"], parameters))
        image_list = [adjust_image(img,
                                   brightness=param_dict["brightness"],
                                   contrast=param_dict["contrast"],
                                   shadows=param_dict["gamma"])
                      for img in self.image_list
                      ]
        ious = [] # initialize empty list of IoUs
        # evaluate all the images
        for idx, img in enumerate(image_list):
            
            label_dict = self.label_list[idx]
            if isinstance(mask_label, str):
                try:
                    mask_id = min([k for k, v in label_dict.items() if v.lower() == mask_label.lower()])
                except ValueError:
                    mask_id = None
            else: # mask_label is int
                mask_id = mask_label if mask_label in label_dict.keys() else None
            
            if mask_id is None: # skip image if label is not present in image
                continue
            
            preprocess_info = self.model_preprocessing(img)
            mask = (self.mask_list[idx] == mask_id)
            points = self.sample(mask, n_points) # define a point list on mask
            
            for point in points:
                # compute inference on each point
                inference_mask = self.model_inference(img, point, param_dict, preprocessing=preprocess_info)
                
                if inference_mask is None: # just a safeguard
                    iou = 0.0
                else:
                    gt = mask.astype(np.uint8)
                    pred = inference_mask.astype(np.uint8)
                    
                    intersection = np.logical_and(gt, pred).sum()
                    union = np.logical_or(gt, pred).sum()
                    
                    if union == 0:
                        iou = 0.0
                    else:
                        iou = intersection / union
                ious.append(iou)
        
        return np.mean(ious)

    def optimize(self, mask_label, max_threshold=0.3, initial_points=10, maxiter=20, n_points=10):
        
        space = [
            Real(-100, 100, name="brightness"),
            Real(-80, 80, name="contrast"),
            Real(-100, 100, name="shadows"),
            Real(0.01, max_threshold, name="threshold"),
            Real(0.01, 0.9, name="grad_edge"),
        ]
        
        def objective_wrapper(parameters):
            return -self.objective(parameters, mask_label=mask_label, n_points=n_points)

        results = gp_minimize(
            func=objective_wrapper,
            dimensions=space,
            n_calls=initial_points+maxiter,
            n_initial_points=initial_points,
            #acq_func="LCB", # more stable for noisy function
            initial_point_generator="lhs",
            random_state=42
        )
        
        # Extract results
        best_params = dict(zip([dim.name for dim in space], results.x))
        best_score = -results.fun
        scores_iter = [-v for v in results.func_vals]
        best_so_far = np.maximum.accumulate(scores_iter)
        
        output = {
            "best_params": best_params,
            "best_score": best_score,
            "scores_iter": scores_iter,
            "best_so_far": best_so_far
        }
        
        # print("Result", result)
        return output

#### Auxiliary functions

def _aux_plot(image, mask, mask_label=None, points=None):
    """
    Debug function that shows what we are doing

    Parameters
    ----------
    image : np.array with dtype uint8
        numpy array with image.
    mask : np.array with dtype uint8
        numpy array with indexed masks, 0=background.
    mask_label : int, optional
        integer representing mask to be shown. The default is None.
    points : np.array with shape (N, 2) and dtype int, optional
        list of points to be shown. The default is None.

    Returns
    -------
    None. Just show the image.

    """
    image = Image.fromarray(image).convert("RGBA")

    if mask_label is None:
        mask_to_show = (mask > 0)
    else:
        mask_to_show = (mask == mask_label)

    overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
    overlay[..., 0] = 255 # mask = red
    overlay[..., 3] = mask_to_show.astype(np.uint8) * 100
    overlay_img = Image.fromarray(overlay, mode="RGBA")

    out = Image.alpha_composite(image, overlay_img)

    # Draw points if provided
    if points is not None:
        draw = ImageDraw.Draw(out)
        for x, y in points:
            r = 3
            draw.ellipse(
                (x - r, y - r, x + r, y + r),
                fill=(255, 255, 0, 255) # yellow points
            )

    out.show()