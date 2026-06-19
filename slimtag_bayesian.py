# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 08:58:50 2026

@author: Giulio Del Corso
"""

#### Libraries
import os
import numpy as np
from PIL import Image, ImageDraw
import threading

from slimtag_utils import adjust_image, MultiButtonDialog
import slimtag_wand as wand

import tkinter as tk
from tkinter import filedialog, font
import customtkinter as ctk

import json

from skopt import gp_minimize
from skopt.space import Real

#### Main class - Bayesian optimizer
class BayesianOptimization():
    
    def __init__(self, image_folder, model_inference, model_preprocessing, mask_folder=None,
                 parent=None, progress_callback=None):
        self.image_folder = image_folder
        self.mask_folder = mask_folder if mask_folder is not None else image_folder
        
        self.model_inference = model_inference
        self.model_preprocessing = model_preprocessing
        
        self.parent = parent # when BayesianOptimization is called from a tk window, to allow communication
        if self.parent is not None:
            self.parent.opt_interface = {} # to store variables for parent (and reset if already present)
            self.parent.opt_interface["tot_calls"] = 0
            self.parent.opt_interface["current"] = 0
            
        self.progress_callback = progress_callback
        
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
                
        if self.parent is not None:
            self.parent.opt_interface["current"] += 1
            if self.progress_callback:
                self.parent.after(0, self.progress_callback)
        
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
        
        if self.parent is not None:
            self.parent.opt_interface["tot_calls"] = initial_points + maxiter
            self.parent.opt_interface["current"] = 0

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

#### Dialog window for main SLImTAG window

class OptimizerDialog(ctk.CTkToplevel):
    """
    Class for dialog window for BayesianOptimization interaction from main app
    """
    def __init__(self, parent):
        super().__init__(parent)

        self.parent = parent # assumed to be SegmentationApp()
        
        self.title("SLImTAG")
        self.geometry("600x300")
        self.resizable(False, False)
        
        self.protocol("WM_DELETE_WINDOW", self.close)
        
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        self.container = ctk.CTkFrame(self, fg_color="transparent")
        self.container.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)
        
        # attributes
        self.folder_path = {k: os.path.abspath(os.getcwd()) for k in ["images", "masks"]}
        self.folder_lbl = {}
        self.same_folder = tk.IntVar(self, value=0)
        self.ignore_mask_name = tk.IntVar(self, value=0)
        
        self.mask_label = None
        
        self.wheel_img = Image.open("images/waiting.png")
        self.wheel_img_tk = ctk.CTkImage(light_image=self.wheel_img, size=(64, 64))
        self.wheel_running = False
        self.wheel_angle = 0
        
        self.computation_done = False # True if optimize has been called (in that case SAM lost the embedding of the original image)
        
        # create pages
        self.initial_frame = self._build_initial_frame()
        self.compute_frame = self._build_compute_frame()
        self.results_frame = self._build_results_frame()
        for frame in [self.initial_frame, self.compute_frame, self.results_frame]:
            frame.grid(row=0, column=0, sticky="nsew")
        
        for fd in ["images", "masks"]:
            self._set_anchor_label(fd)
        self.show_frame(self.initial_frame)
        
        self.update_idletasks()
        self.transient(parent)
        self.grab_set()
        self.after(100, self.focus_set)
    
    def show_frame(self, frame):
        frame.tkraise()
    
    def _build_initial_frame(self):
        frame = ctk.CTkFrame(self.container, fg_color="transparent")
        
        options_frame = ctk.CTkFrame(frame, fg_color="transparent")
        options_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 0))
        options_frame.grid_rowconfigure((0, 1, 2), weight=1)
        options_frame.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(options_frame, text="Images folder").grid(row=0, column=0, padx=(10, 5), sticky="w")
        self.folder_lbl["images"] = ctk.CTkLabel(options_frame, text=self.folder_path["images"])
        self.folder_lbl["images"].grid(row=0, column=1, padx=5, sticky="ew")
        self.images_folder_btn = ctk.CTkButton(options_frame, text="Change folder", command=lambda folder=["images"]: self._select_folder(folder))
        self.images_folder_btn.grid(row=0, column=2, padx=5, pady=5, sticky="ew")
        
        ctk.CTkLabel(options_frame, text="Masks folder").grid(row=1, column=0, padx=(10, 5), sticky="w")
        self.folder_lbl["masks"] = ctk.CTkLabel(options_frame, text=self.folder_path["masks"])
        self.folder_lbl["masks"].grid(row=1, column=1, padx=5, sticky="ew")
        self.masks_folder_btn = ctk.CTkButton(options_frame, text="Change folder", command=lambda folder=["masks"]: self._select_folder(folder))
        self.masks_folder_btn.grid(row=1, column=2, padx=5, pady=5, sticky="ew")
        same_folder_check = ctk.CTkCheckBox(options_frame, text="Same as images", variable=self.same_folder, command=self._toggle_same_folder)
        same_folder_check.grid(row=1, column=3, padx=5, sticky="ew")
        
        ctk.CTkLabel(options_frame, text="Mask label").grid(row=2, column=0, padx=(10, 5), sticky="w")
        self.mask_id_list = sorted(list(self.parent.mask_labels.keys())) # ASSUME this is not empty, check done in main app
        self.mask_name_list = [f"{self.parent.mask_labels[mid]}" for mid in self.mask_id_list]
        self.mask_menu_list = [f"{mid}: {name}" for mid, name in zip(self.mask_id_list, self.mask_name_list)]
        if self.parent.active_mask_id is not None:
            selected = self.mask_menu_list[self.mask_id_list.index(self.parent.active_mask_id)]
        else:
            selected = self.mask_menu_list[0]
        self.masks_menu = ctk.CTkOptionMenu(options_frame, values=self.mask_menu_list)
        self.masks_menu.set(selected)
        self.masks_menu.grid(row=2, column=1, padx=5, sticky="ew")
        ignore_check = ctk.CTkCheckBox(options_frame, text="Ignore mask name", variable=self.ignore_mask_name)
        ignore_check.grid(row=2, column=2, padx=5, sticky="ew")
        
        buttons_frame = ctk.CTkFrame(frame, fg_color="transparent")
        buttons_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        buttons_frame.grid_rowconfigure(0, weight=1)
        buttons_frame.grid_columnconfigure(0, weight=1)
        
        inner_btn_frame = ctk.CTkFrame(buttons_frame, fg_color="transparent")
        inner_btn_frame.grid(row=0, column=0, padx=0, pady=0)
        
        btn_start = ctk.CTkButton(inner_btn_frame, text="Start", command=self.start_optimization)
        btn_start.grid(row=0, column=0, padx=5)
        btn_cancel = ctk.CTkButton(inner_btn_frame, text="Cancel", command=self.close)
        btn_cancel.grid(row=0, column=1, padx=5)
        
        frame.grid_rowconfigure((0, 1), weight=1)
        frame.grid_columnconfigure(0, weight=1)
        
        return frame
    
    def _build_compute_frame(self):
        frame = ctk.CTkFrame(self.container, fg_color="transparent")
        
        self.wheel_lbl = ctk.CTkLabel(frame, text="Optimizing parameters...", image=self.wheel_img_tk, compound="top", pady=15)
        self.wheel_lbl.grid(row=0, column=0, sticky="nsew")
        
        self.progress = ctk.CTkProgressBar(frame, progress_color="#15C2D2")
        self.progress.set(0)
        self.progress.grid(row=1, column=0, padx=15, pady=(0, 15), sticky="nsew")
        
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        
        return frame
    
    def _build_results_frame(self):
        frame = ctk.CTkFrame(self.container, fg_color="transparent")
        
        results_frame = ctk.CTkFrame(frame, fg_color="transparent")
        results_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        results_frame.grid_rowconfigure(0, weight=1)
        results_frame.grid_columnconfigure(0, weight=1)
        
        inner_res_frame = ctk.CTkFrame(results_frame, fg_color="transparent")
        inner_res_frame.grid(row=0, column=0, padx=0, pady=0)
        
        ctk.CTkLabel(inner_res_frame, text="PARAMETER", anchor="w").grid(row=0, column=0, sticky="ew", padx=10)
        ctk.CTkLabel(inner_res_frame, text="OLD").grid(row=0, column=1, sticky="ew", padx=10)
        ctk.CTkLabel(inner_res_frame, text="NEW", font=ctk.CTkFont(weight="bold")).grid(row=0, column=2, sticky="ew", padx=10)
        
        self.results_lbl = {}
        ctk.CTkLabel(inner_res_frame, text="Brightness", anchor="w").grid(row=1, column=0, sticky="ew", padx=10)
        ctk.CTkLabel(inner_res_frame, text=f"{self.parent.wand_brightness}").grid(row=1, column=1, sticky="ew", padx=10)
        self.results_lbl["brightness"] = ctk.CTkLabel(inner_res_frame, text="0", font=ctk.CTkFont(weight="bold"))
        self.results_lbl["brightness"].grid(row=1, column=2, sticky="ew", padx=10)
        ctk.CTkLabel(inner_res_frame, text="Contrast", anchor="w").grid(row=2, column=0, sticky="ew", padx=10)
        ctk.CTkLabel(inner_res_frame, text=f"{self.parent.wand_contrast}").grid(row=2, column=1, sticky="ew", padx=10)
        self.results_lbl["contrast"] = ctk.CTkLabel(inner_res_frame, text="0", font=ctk.CTkFont(weight="bold"))
        self.results_lbl["contrast"].grid(row=2, column=2, sticky="ew", padx=10)
        ctk.CTkLabel(inner_res_frame, text="Shadows", anchor="w").grid(row=3, column=0, sticky="ew", padx=10)
        ctk.CTkLabel(inner_res_frame, text=f"{self.parent.wand_gamma}").grid(row=3, column=1, sticky="ew", padx=10)
        self.results_lbl["shadows"] = ctk.CTkLabel(inner_res_frame, text="0", font=ctk.CTkFont(weight="bold"))
        self.results_lbl["shadows"].grid(row=3, column=2, sticky="ew", padx=10)
        ctk.CTkLabel(inner_res_frame, text="Wand threshold", anchor="w").grid(row=4, column=0, sticky="ew", padx=10)
        ctk.CTkLabel(inner_res_frame, text=f"{self.parent.wand_threshold:.2f}").grid(row=4, column=1, sticky="ew", padx=10)
        self.results_lbl["threshold"] = ctk.CTkLabel(inner_res_frame, text="0", font=ctk.CTkFont(weight="bold"))
        self.results_lbl["threshold"].grid(row=4, column=2, sticky="ew", padx=10)
        self.results_lbl["grad_edge"] = ctk.CTkLabel(inner_res_frame, text="0", font=ctk.CTkFont(weight="bold"))
        if self.parent.wand_model_menu.get() == "Region growing":
            ctk.CTkLabel(inner_res_frame, text="Edge tolerance", anchor="w").grid(row=5, column=0, sticky="ew", padx=10)
            ctk.CTkLabel(inner_res_frame, text=f"{self.parent.wand_edge_tolerance:.2f}").grid(row=5, column=1, sticky="ew", padx=10)
            self.results_lbl["grad_edge"].grid(row=5, column=2, sticky="ew", padx=10)
        
        buttons_frame = ctk.CTkFrame(frame, fg_color="transparent")
        buttons_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        buttons_frame.grid_rowconfigure(0, weight=1)
        buttons_frame.grid_columnconfigure(0, weight=1)
        
        inner_btn_frame = ctk.CTkFrame(buttons_frame, fg_color="transparent")
        inner_btn_frame.grid(row=0, column=0, padx=0, pady=0)
        
        btn_start = ctk.CTkButton(inner_btn_frame, text="Apply", command=self.apply)
        btn_start.grid(row=0, column=0, padx=5)
        btn_cancel = ctk.CTkButton(inner_btn_frame, text="Discard", command=self.close)
        btn_cancel.grid(row=0, column=1, padx=5)
        
        frame.grid_rowconfigure((0, 1), weight=1)
        frame.grid_columnconfigure(0, weight=1)
        
        return frame
    
    def _select_folder(self, folder): # folder = sublist of ["images", "masks"]
        p = filedialog.askdirectory(parent=self)
        if not p:
            return
        for fd in folder:
            self.folder_path[fd] = p
            self.folder_lbl[fd].configure(text=p)
            self._set_anchor_label(fd)
    
    def _toggle_same_folder(self):
        if self.same_folder.get():
            self.masks_folder_btn.configure(state="disabled")
            self.images_folder_btn.configure(command=lambda folder=["images", "masks"]: self._select_folder(folder))
            self.folder_path["masks"] = self.folder_path["images"]
            self.folder_lbl["masks"].configure(text=self.folder_path["images"])
            self._set_anchor_label("masks")
        else:
            self.masks_folder_btn.configure(state="normal")
            self.images_folder_btn.configure(command=lambda folder=["images"]: self._select_folder(folder))
    
    def _set_anchor_label(self, folder):
        # set anchor depending on label length
        self.update_idletasks()
        ft = font.Font(font=self.folder_lbl[folder].cget("font"))
        wt = ft.measure(self.folder_lbl[folder].cget("text"))
        if wt <= self.folder_lbl[folder].winfo_width():
            self.folder_lbl[folder].configure(anchor="w")
        else:
            self.folder_lbl[folder].configure(anchor="e")
            
    def _animate_wheel(self):
        if not self.wheel_running:
            return
        rotated = self.wheel_img.rotate(-self.wheel_angle)
        self.wheel_img_tk = ctk.CTkImage(light_image=rotated, size=(64, 64))
        self.wheel_lbl.configure(image=self.wheel_img_tk)
        self.wheel_angle = (self.wheel_angle + 30) % 360
        self.after(25, self._animate_wheel)
    
    def progress_bar_update(self):
        if hasattr(self, "opt_interface"):
            self.progress.set(self.opt_interface["current"]/self.opt_interface["tot_calls"])
            self.update()
        
    def start_optimization(self):
        try:
            if self.parent.wand_model_menu.get() == "Region growing":
                model_inference = wand.region_growing_inference
                model_preprocessing = wand.region_growing_preprocessing
            elif self.parent.wand_model_menu.get() in self.parent.available_sam_models:
                model_inference = lambda img, pt, parameters, preprocessing=None: wand.sam_inference(img, [pt], parameters, model=self.parent.sam, preprocessing=preprocessing)
                model_preprocessing = lambda img: wand.sam_preprocessing(img, self.parent.sam)
            else:
                MultiButtonDialog(self, message="Unknown magic wand model", buttons=[("OK", None)])
                return
            self.optimizer = BayesianOptimization(image_folder=self.folder_path["images"],
                                                  model_inference=model_inference,
                                                  model_preprocessing=model_preprocessing,
                                                  mask_folder=self.folder_path["masks"],
                                                  parent=self,
                                                  progress_callback=self.progress_bar_update)
        except RuntimeError: # raised if path_directory does not contain valid images/masks pairs
            MultiButtonDialog(self, message="Valid image/mask pairs not found in provided folders", buttons=[("OK", None)])
            return
        
        all_masks_ids = sum([[int(k) for k in lbldict] for lbldict in self.optimizer.label_list], [])
        all_masks_names = sum([[lbldict[k].lower() for k in lbldict] for lbldict in self.optimizer.label_list], [])
        selected = self.mask_menu_list.index(self.masks_menu.get())
        if self.ignore_mask_name.get():
            mask_label = self.mask_id_list[selected]
            if mask_label not in all_masks_ids:
                MultiButtonDialog(self, message=f"Label {mask_label} not found in masks", buttons=[("OK", None)])
                return
        else:
            mask_label = self.mask_name_list[selected]
            if mask_label.lower() not in all_masks_names:
                MultiButtonDialog(self, message=f"Label '{mask_label}' not found in masks", buttons=[("OK", None)])
                return
            
        self.mask_label = mask_label
        self.wheel_running = True
        self._animate_wheel()
        self.show_frame(self.compute_frame)

        threading.Thread(
            target=self._optimize,
            daemon=True,
        ).start()

    def _optimize(self):
        self.computation_done = True
        self.results = self.optimizer.optimize(mask_label=self.mask_label,
                                               initial_points=self.parent.slimtag_config["bayesian_optimization"]["initial_points"],
                                               maxiter=self.parent.slimtag_config["bayesian_optimization"]["max_iterations"],
                                               n_points=self.parent.slimtag_config["bayesian_optimization"]["n_points"]
                                               )
        self.new_params = {}
        for res in ["brightness", "contrast", "shadows"]:
            self.new_params[res] = min(max(round(self.results["best_params"][res]), -100), 100)
        for res in ["threshold", "grad_edge"]:
            self.new_params[res] = min(max(self.results["best_params"][res], 0.0), 1.0)
        # switch back after computation
        self.after(0, self.end_optimization)
    
    def end_optimization(self):
        self.wheel_running = False
        for res in ["brightness", "contrast", "shadows"]:
            self.results_lbl[res].configure(text=f"{self.new_params[res]}")
        for res in ["threshold", "grad_edge"]:
            self.results_lbl[res].configure(text=f"{self.new_params[res]:.2f}")
        self.show_frame(self.results_frame)

    def apply(self):
        self.parent.wand_brightness = self.new_params["brightness"]
        self.parent.wand_contrast = self.new_params["contrast"]
        self.parent.wand_gamma = self.new_params["shadows"]
        self.parent.wand_threshold = self.new_params["threshold"]
        self.parent.wand_edge_tolerance = self.new_params["grad_edge"]
        
        self.parent.wand_brightness_lbl.configure(text=str(self.parent.wand_brightness))
        self.parent.wand_contrast_lbl.configure(text=str(self.parent.wand_contrast))
        self.parent.wand_gamma_lbl.configure(text=str(self.parent.wand_gamma))
        self.parent.wand_threshold_lbl.configure(text=f"{self.parent.wand_threshold:.2f}")
        self.parent.wand_threshold_slider.set(self.parent.wand_threshold)
        if self.parent.wand_model_menu.get() == "Region growing":
            self.parent.wand_edge_tolerance_lbl.configure(text=f"{self.parent.wand_edge_tolerance:.2f}")
            self.parent.wand_edge_tolerance_slider.set(self.parent.wand_edge_tolerance)
        self.close()
    
    def close(self):
        if self.computation_done:
            self.parent.async_loader()
        self.destroy()
    
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