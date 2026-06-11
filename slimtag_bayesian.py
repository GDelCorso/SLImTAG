# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 08:58:50 2026

@author: Giulio Del Corso
"""

#%% Libraries
import os
import numpy as np
from PIL import Image, ImageDraw

from slimtag_utils import adjust_image

from scipy import ndimage # Region growing

from skopt import gp_minimize
from skopt.space import Real


#%% Main class - Bayesian
class BayesianOptimization:
    
    def __init__(self, path_folder, model_inference, model_preprocessing):
        self.path_folder = path_folder
        
        self.model_inference = model_inference
        self.model_preprocessing = model_preprocessing
        
        # Load images and masks
        self.image_list, self.mask_list = self.load_folder()
        
        
        
    def load_folder(self):
        list_names = os.listdir(self.path_folder)
        
        # Selection
        valid_ext = {".jpg", ".jpeg", ".png"}
        
        images = {}
        masks = {}
        
        for fname in list_names:
            stem, ext = os.path.splitext(fname)
        
            if ext.lower() not in valid_ext:
                continue
        
            if stem.endswith("_mask"):
                base_name = stem[:-5]  
                masks[base_name] = fname
            else:
                images[stem] = fname
        
        # Keep coupled 
        common_names = sorted(set(images.keys()) & set(masks.keys()))
        if len(common_names) == 0:
            raise RuntimeError("empty list of images/masks pairs in folder")
        
        image_paths = [os.path.join(self.path_folder,images[name]) for name in common_names]
        mask_path = [os.path.join(self.path_folder,masks[name]) for name in common_names]

        # Load images/masks
        image_list = [np.array(Image.open(path).convert("RGB")) for path in image_paths]
        mask_list  = [np.array(Image.open(path).convert("P")) for path in mask_path]
        
        return image_list, mask_list
    

    def sample(self, mask, mask_label, n_points):
    
        selected = (mask == mask_label)
        ys, xs = np.where(selected)
    
        if len(xs) == 0:
            raise ValueError(f"Mask label {mask_label} not found in mask")
    
        coords = np.stack([xs, ys], axis=1)
    
        n_points = min(n_points, len(coords))
    
        idx = np.random.choice(len(coords), size=n_points, replace=False)
        return coords[idx]
    

    
    
    def _aux_plot(self,image, mask, mask_label=None, points=None):

        image = Image.fromarray(image).convert("RGBA")

        if mask_label is None:
            mask_to_show = (mask > 0)
        else:
            mask_to_show = (mask == mask_label)

        overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)

        overlay[..., 0] = 255
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
                    fill=(255, 255, 0, 255)
                )

        out.show()
        

    def region_growing_objective(self, parameters, mask_label, n_points=5, beta = 0.5):
        # Beta = 1 classical F1 score, >1 improve recall, <1 improve precision
    
        # Preprocess image list to adjust w.r.t. parameters
        preprocessed_image_list = [
            adjust_image(img,
                         brightness=parameters[0],
                         contrast=parameters[1],
                         shadows=parameters[2])
            for img in self.image_list
        ]
    
        # Initialize empty list of IoI
        ious = []
    
        # Evaluate all the images
        for idx, img in enumerate(preprocessed_image_list):
            
            # For each image compute ONCE the aux values for region growing
            region_growing_gray, region_growing_rgb, region_growing_grad = \
                region_growing_preprocessing(Image.fromarray(img), model=None)
                
            # Extract the mask
            mask = self.mask_list[idx]
    
            # Define a point list on mask
            points = self.sample(mask, mask_label, n_points)
            for point in points:
                # Compute inference on each point #TODO fare il for
                inference_mask = region_growing_model_inference(
                    img,
                    point,
                    region_growing_gray=region_growing_gray,
                    region_growing_rgb=region_growing_rgb,
                    region_growing_grad=region_growing_grad,
                    tolerance=parameters[3],
                    max_grad_edge=parameters[4]
                )
                
                
                if inference_mask is None:
                    iou = 0.0
                else:
                    
                    
                    gt = (mask == mask_label).astype(np.uint8)
                    pred = inference_mask.astype(np.uint8)
                
                    intersection = np.logical_and(gt, pred).sum()
                    union = np.logical_or(gt, pred).sum()
                
                    if union == 0:
                        iou = 0.0
                    else:
                        iou = intersection / union
        
                ious.append(iou)
        
        return np.mean(ious)
    
    
    
    
    def optimize(self, mask_label, max_threshold=0.3, initial_points=10, maxiter=20, n_points = 10):

    
        space = [
            Real(-100, 100, name="brightness"),
            Real(-80, 80, name="contrast"),
            Real(-100, 100, name="shadows"),
            Real(0.01, max_threshold, name="threshold"),
            Real(0.01, 0.9, name="grad_edge"),
        ]
        
        
        def objective_wrapper(parameters):
            return -self.region_growing_objective(parameters, mask_label=mask_label, n_points=n_points)

        results = gp_minimize(
            func=objective_wrapper,
            dimensions=space,
            n_calls=initial_points+maxiter,
            n_initial_points=initial_points,
            #acq_func="LCB",             # More stable for noisy function
            initial_point_generator="lhs",
            random_state=42
        )
        
        # Extract results
        best_params = dict(zip(
            [dim.name for dim in space],
            results.x
        ))
        
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
        



#%% Temporary model inference for testing purposes
def region_growing_preprocessing(image, model):
    ## PRE-COMPUTATION for region grown
    # Grayscale image for region growing
    region_growing_gray = np.array(
        image.convert("L"),
        dtype=np.float32
    )
    
    # Edge detection for region growing
    gx = ndimage.sobel(region_growing_gray, axis=1)
    gy = ndimage.sobel(region_growing_gray, axis=0)
    region_growing_grad = np.hypot(gx, gy)
    
    # normalize for stability and make in in  [0,1]
    #self.region_growing_grad = self.region_growing_grad / (self.region_growing_grad.max() + 1e-8)
    p = np.percentile(region_growing_grad, 99)
    region_growing_grad = np.clip(region_growing_grad / (p + 1e-8), 0, 1)
    
    # RGB
    region_growing_rgb = np.array(
        image.convert("RGB"),
        dtype=np.float32
    )
    
    return region_growing_gray, region_growing_rgb, region_growing_grad


def region_growing_model_inference(image, point, 
                region_growing_gray, region_growing_rgb, region_growing_grad,
                tolerance=0.15, 
                switch_robust_estimator=True, switch_RGB=True, use_edges=True, 
                max_grad_edge=0.5, switch_erosion=False, switch_fill_hole=True, 
                max_hole_size=100):
    
    # Map coordinates
    x = int(point[0])
    y = int(point[1])
    
    # Select the seed value
    if switch_robust_estimator:
        # Trimmed robust local estimator
        # GRAYSCALE
        patch_gray = region_growing_gray[
            max(0, y-1):min(region_growing_gray.shape[0], y+2),
            max(0, x-1):min(region_growing_gray.shape[1], x+2)
        ].astype(np.float32)
        vals_gray = patch_gray.flatten()
        med_gray = np.median(vals_gray)
        dist_gray = np.abs(vals_gray - med_gray)
        keep_idx = np.argsort(dist_gray)[:5]
        seed_val_gray = vals_gray[keep_idx].mean()
    
        # RGB ROBUST SEED
        patch_rgb = region_growing_rgb[
            max(0, y-1):min(region_growing_rgb.shape[0], y+2),
            max(0, x-1):min(region_growing_rgb.shape[1], x+2)]
        
        vals_rgb = patch_rgb.reshape(-1, 3)
        med_rgb = np.median(vals_rgb, axis=0)
        dist_rgb = np.sum((vals_rgb - med_rgb)**2, axis=1)
        keep_idx_rgb = np.argsort(dist_rgb)[:5]
        seed_val_rgb = vals_rgb[keep_idx_rgb].mean(axis=0)
    else:
        # Pixel perfect estimator
        seed_val_gray = region_growing_gray[y, x]
        seed_val_rgb = region_growing_rgb[y, x]
    

    # Similarity score
    if switch_RGB:
        # RGB
        rgb = region_growing_rgb
        dr = rgb[..., 0] - seed_val_rgb[0]
        dg = rgb[..., 1] - seed_val_rgb[1]
        db = rgb[..., 2] - seed_val_rgb[2]
        diff = np.sqrt(dr*dr + dg*dg + db*db)
    else:
        # GRAY Scale
        diff = np.abs(region_growing_gray - seed_val_gray) # in FLOAT 32
    

    # Edges computation (evaluated on grayscale)
    if use_edges:
        # Integrates edges by removing all area with grad greater thatn max_grad_edge
        mask = (diff < round(tolerance*255)) & (region_growing_grad < max_grad_edge) 

    else:
        # classic region growing
        mask = diff < round(tolerance*255) 


    # Erode to avoid small connections between zones
    if switch_erosion:
        mask_clean = ndimage.binary_opening(mask, structure=np.ones((3,3)))
        
        # Extraction of single connected component
        labeled, _ = ndimage.label(mask_clean)
        seed_label = labeled[y, x]
        if seed_label == 0:
            return
    else:
        # Extraction of single connected component
        labeled, _ = ndimage.label(mask)
        seed_label = labeled[y, x]
        if seed_label == 0:
            return

    region = (labeled == seed_label)

    # Fill small holes
    if switch_fill_hole:
        # holes = internal background of the region
        holes = ndimage.binary_fill_holes(region) & (~region)
        
        # label holes
        hole_labels, num_holes = ndimage.label(holes)
        
        # compute hole sizes
        hole_sizes = np.bincount(hole_labels.ravel())
        
        # skip label 0 (background)
        small_holes = np.isin(
            hole_labels,
            np.where(hole_sizes <= max_hole_size)[0]
        )
        
        # fill only small holes
        region = region | small_holes
    
    
    return region.astype(np.uint8) if region is not None else np.zeros_like(region, dtype=np.uint8)
    




#%% Main example
if __name__ == "__main__":
    BO = BayesianOptimization(path_folder="../COCO", 
                              model_inference=region_growing_model_inference,
                              model_preprocessing=region_growing_preprocessing)
    mask_label = 3
    results = BO.optimize(mask_label=mask_label, initial_points=10, maxiter=30, n_points=50)
    
    print(results)
    
