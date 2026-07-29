"""Methods for the io responsibilities of SegmentationApp."""

import io
import json
import os
import shutil
import tarfile
import threading

import numpy as np
from PIL import Image, PngImagePlugin
from tkinter import filedialog

from slimtag_color_utils import hex_to_rgb, rgb_to_hex
from slimtag_utils import MultiButtonDialog


class IOMixin:
    """Composable io behaviour for ``SegmentationApp``."""

    def load_image(self, pil_image, mask=None, change_canvas=None, reset_view=True):
        '''
        Load the current image in memory as self.image_orig and display it.
        
        Takes a PIL image as input, which can be provided by the several load_*
        methods.
        
        If mask is None, create new zero mask array; otherwise this is a uint8 array
        with the same shape as the image. Used e.g. to load masks corresponding to
        different volume slices.
        
        If change_canvas is not None, it is a string among keys of self.canvas_frames
        '''
        self.orig_w, self.orig_h = pil_image.size
        self.image_orig = pil_image
        if mask is None:
            self.mask_orig = np.zeros((self.orig_h, self.orig_w), np.uint8)
            self.mask_locked = np.full(self.mask_orig.shape, False)
        else:
            self.mask_orig = mask
            self.mask_locked = np.full(self.mask_orig.shape, False)
            self.update_lock()
        self.sam_preview = np.full(self.mask_orig.shape, False)
        


        # Async load of the SAM model to avoid freezed interface
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self.async_loader, daemon=True)
            self.thread.start()



        # raise canvas
        # TODO: folder management -- raise appropriate canvas
        if change_canvas is not None:
            self.show_canvas_frame(change_canvas)
        
        self.update_idletasks()
        
        if reset_view:
            # reset zoom
            self.zoom = 1.0
            # define the view parameters (equal to the canvas size)
            self.view_x = 0
            self.view_y = 0
            self.view_w = self.canvas.winfo_width()
            self.view_h = self.canvas.winfo_height()
        
        self.update_display(update_image=True)
        
        self.show_preview_frame("image")
        self.update_preview_frame()
    
    def open_image(self, path=None, add_mask=True):
        '''
        Load a .png or .jpg image and define an empty mask on it.
        
        Return True if a NEW image is correctly loaded, False otherwise
        '''

        if self.modified:
            confirm = MultiButtonDialog(self, message="There are unsaved changes. What do you want to do?",
                                        buttons=(("Save changes", "save"), ("Discard changes", "discard"), ("Cancel", None))
                                       )
            answer = confirm.return_value
            if answer == "save":
                self.save_mask()
                self.set_modified(False)
            elif answer == "discard":
                self.set_modified(False)
            else:
                return False

        self.deactivate_tools()
        self.set_controls_state(False)
        
        # Dialog
        if path is None:
            # Reset path
            self.list_images = None
            self.list_index = 0
            
            p = filedialog.askopenfilename(filetypes=[("Image files", ("*.png", "*.jpg", "*.jpeg"))])
            if not p:
                return False
            
        else:
            p = path
        
        self.path_original_image = p
        self.quicksave_path = os.path.splitext(p)[0] + "_mask.png"

        self.set_status("loading", "Loading image...")
        
        # reset masks
        self.clear_all_masks()
        
        img = Image.open(p).convert("RGBA").convert("RGB") # explicit conversion to normalize RGBA images
        
        self.load_image(img, change_canvas="default")
        
        self.update_title()
        
        # reset history
        self.undo_stack.clear()
        
        if add_mask:
            self.add_mask("mask_1")
        

        
        if self.list_images is None:
            self.images_num_label_var.set("Image 1 of 1")
            #self.next_image_btn.configure(state="disabled")
            # TODO image navigation

        self.toggle_all_masks_hide(set_hide=False, enabled=True)
        self.toggle_all_masks_lock(set_lock=False, enabled=True)
        
        self.set_status("ready", "Ready")
        
        return True
        
    def load_folder(self): # TODO rewrite open folder
        '''
        Aux function to load a whole folder to speed up image segmentation
        '''
        
        # Check already existing images/mask
        if self.modified:
            confirm = MultiButtonDialog(self, message="There are unsaved changes. What do you want to do?",
                                        buttons=(("Save changes", "save"), ("Discard changes", "discard"), ("Cancel", None))
                                       )
            answer = confirm.return_value
            if answer == "save":
                self.save_mask()
                self.set_modified(False)
            elif answer == "discard":
                self.set_modified(False)
            else:
                return
            
        self.deactivate_tools()
        self.set_controls_state(False)
        
        # Select a directory
        path_directory =  filedialog.askdirectory()
        if not path_directory:
            return
        
        # Define an aux directory to save masks:
        self.path_aux_save = path_directory+"_mask"
            
        if os.path.isdir(self.path_aux_save): # TODO improve
            shutil.rmtree(self.path_aux_save)
        os.mkdir(self.path_aux_save)

        # Define the list of possible images
        self.list_images = sorted([os.path.join(path_directory, f) for f in os.listdir(path_directory) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        self.list_index = 0
        
        # Load the image corresponding to list index
        self.set_status("loading", "Loading image...")

        
        self.open_image(path=self.list_images[self.list_index])
        
        self.images_num_label_var.set(f"Image {self.list_index+1} of {len(self.list_images)}")
        self.next_image_btn.configure(state="disabled") # Originally disabled
        
        self.set_status("ready", "Ready")
        
    def next_image(self): # TODO previous image (even better, a parameter 'direction'='+' or '-')
        '''
        Binding for next image
        '''
        if self.list_images != None and (self.list_index < len(self.list_images)-1):
            
            
            # Load the image corresponding to list index
            self.set_status("loading", "Loading next image...")
            
            # Save # TODO - Add a warning
            self.save_mask(switch_fast=True)
            
            self.list_index += 1
        
            self.open_image(path=self.list_images[self.list_index])
            
            self.images_num_label_var.set(f"Image {self.list_index+1} of {len(self.list_images)}")
            
            self.next_image_btn.configure(state="disabled") # Disable next img
            self.switch_computed_magic_wand = False # Disable MAGIC WAND
            self.magic_btn.configure(state="disabled")
            
            self.set_status("ready", "Ready")

    def load_mask(self): # TODO rename "open mask"?
        """
        Upload an existing mask
        - Indexed PNG (mode "P"): direct recovery of mask indices.
        - RGB PNG: legacy color-based reconstruction.
        - (for volumes): TAR containing indexed PNG files, as produced by save_mask
        """
        if not self.image_is_loaded():
            return
        
        self.deactivate_tools()
        if self.is_volume_loaded:
            def_ext = ".tar"
            ftypes = [("TAR archive of indexed PNGs", ".tar")]
        else:
            def_ext = ".png"
            ftypes = [("PNG (indexed or RGB)", "*.png")]
        p = filedialog.askopenfilename(filetypes=ftypes)
        if not p:
            return

        self.set_status("loading", "Loading mask...")
        
        self.push_undo()
        ext = os.path.splitext(p)[1].lower()
        if ext != def_ext:
            return
        
        self.quicksave_path = p
        
        # RESET ALL MASKS
        self.clear_all_masks()
        
        
        if self.is_volume_loaded:
            with tarfile.open(p, mode="r") as tf:
                # recover metadata
                metajson = json.load(tf.extractfile(tf.getmember("meta.json")))
                labels = sorted([k for k in metajson["labels"].keys() if int(k) <= self.slimtag_config["mask"]["max_masks"]], key=int)
                for l in labels:
                    self.mask_labels[int(l)] = metajson["labels"][l]["name"]
                    self.mask_colors[int(l)] = hex_to_rgb(metajson["labels"][l]["color"])
                    self.mask_widgets[int(l)] = self.create_mask_widget(int(l))
                    self.mask_widgets[int(l)].pack(fill="x", expand=True)
                if len(labels) > 0:
                    self.change_mask(target_id=int(labels[0]))
                self.volume_mask = np.zeros(tuple(metajson["shape"]), dtype=np.uint8)
                for member in tf:
                    if member.isfile() and member.name.endswith(".png"):
                        f = tf.extractfile(member)
                        if f is None:
                            continue
                        img = Image.open(io.BytesIO(f.read()))
                        idx = int(member.name.split(".")[0])
                        self.volume_mask[..., idx] = np.array(img, dtype=np.uint8)
            z = round(self.volume_zslider.get())
            self.mask_orig = self.volume_mask[..., z]

        else:
            
            img = Image.open(p)
            
            # CASE 1: Indexed PNG
            if img.mode == "P":
                arr = np.array(img, dtype=np.uint8)
                self.mask_orig = arr.copy()
                labels = np.unique(arr)
                labels = labels[labels != 0][:self.slimtag_config["mask"]["max_masks"]]
                palette = img.getpalette()
                try:
                    names = json.loads(img.text["labels"])
                except KeyError:
                    names = {str(l): f"mask_{l}" for l in labels.tolist()}
                    
                for l in labels.tolist():
                    self.mask_labels[l] = names[str(l)]#f"mask_{l}"
                    idx = l * 3
                    self.mask_colors[l] = tuple(palette[idx:idx+3])
                    self.mask_widgets[l] = self.create_mask_widget(l)
                    self.mask_widgets[l].pack(fill="x", expand=True)
                
                if len(labels) > 0:
                    self.change_mask(target_id=labels[0])
            
            # CASE 2: Generic RGB PNG
            else:
                img = img.convert("RGB")
                arr = np.array(img)
                h, w, _ = arr.shape
                
                arr_flat = arr.reshape(-1, 3)
                arr_flat_nonblack = arr_flat[~np.all(arr_flat == 0, axis=1)]
                
                if len(arr_flat_nonblack) == 0:
                    self.mask_orig = np.zeros((h, w), np.uint8)
                    return
                
                unique_colors = []
                seen = set()
                for color in arr_flat_nonblack:
                    t = tuple(color)
                    if t not in seen:
                        seen.add(t)
                        unique_colors.append(t)
                        if len(unique_colors) >= self.slimtag_config["mask"]["max_masks"]:
                            break
    
                try:
                    names = json.loads(img.text["labels"])
                except KeyError:
                    names = {str(l+1): f"mask_{l+1}" for l in range(len(unique_colors))}
                
                mask = np.zeros((h, w), np.uint8)
                for i, color in enumerate(unique_colors, 1):
                    mask[np.all(arr == color, axis=-1)] = i
                    self.mask_labels[i] = names[str(i)]#f"mask_{i}"
                    self.mask_colors[i] = color
                    self.mask_widgets[i] = self.create_mask_widget(i)
                    self.mask_widgets[i].pack(fill="x", expand=True)
                
                self.mask_orig = mask
                if unique_colors:
                    self.change_mask(target_id=1)
        
        # prepare empty mask with same size for SAM preview
        self.sam_preview = np.full(self.mask_orig.shape, False)

        self.toggle_all_masks_hide(set_hide=False, enabled=True)
        self.toggle_all_masks_lock(set_lock=False, enabled=True)
        
        self.update_display(update_image=True)
        self.set_status("ready", "Ready")


    def save_mask(self, switch_fast=False):
        '''
        Save mask as a proper indexed png file and an associated png image to 
        see the identified masks.
        '''
        # Check if an image is loaded
        if not self.image_is_loaded():
            return
        if self.mask_orig is None:
            return
    
        if not switch_fast:
            # Save as()
            if self.is_volume_loaded:
                def_ext = ".tar"
                ftypes = [("TAR archive of indexed PNGs", ".tar")]
            else:
                def_ext = ".png"
                ftypes = [("PNG (indexed)", "*.png")]
            p = filedialog.asksaveasfilename(defaultextension=def_ext, filetypes=ftypes)
            if not p:
                return
        else: # Save()
            # If working with a folder # TODO adapt to volume folder
            if self.list_images != None:
                p = os.path.join(self.path_aux_save, os.path.splitext(os.path.basename(self.list_images[self.list_index]))[0]+".png")
            # Otherwise
            else:
                p = self.quicksave_path
        
        self.set_status("loading", "Saving mask...")
        
        # palette (common to all PNGs)
        palette = [0, 0, 0] * 256  # index 0 = black background
        for mid, color in self.mask_colors.items():
            palette[mid*3:mid*3+3] = list(color)

        # save mask names in png metadata (common to all PNGs))
        metadata = PngImagePlugin.PngInfo()
        labels = {i: self.mask_labels[i] for i in self.mask_labels.keys()}
        metadata.add_text("labels", json.dumps(labels))
        
        if self.is_volume_loaded:
            n_slices = self.volume_mask.shape[2]
            nd = len(str(n_slices)) # max number of digits, for zero padding in namefiles
            with tarfile.open(p, mode="w") as tf:
                metajson = {"shape": list(self.volume_mask.shape),
                            "labels": {i: {"name": self.mask_labels[i],
                                           "color": rgb_to_hex(self.mask_colors[i])
                                           } for i in self.mask_labels.keys()}
                            }
                metafile = io.BytesIO(json.dumps(metajson, indent=2).encode("utf-8"))
                tinfo = tarfile.TarInfo(name="meta.json")
                tinfo.size = len(metafile.getbuffer())
                tf.addfile(tarinfo=tinfo, fileobj=metafile)
                for z in range(n_slices):
                    if self.volume_mask[..., z].any():
                        mem_buffer = io.BytesIO()
                        png_file = Image.fromarray(self.volume_mask[..., z], mode="P")
                        png_file.putpalette(palette)
                        png_file.save(mem_buffer, format="PNG", pnginfo=metadata)
                        mem_buffer.seek(0) # rewind buffer
                        tinfo = tarfile.TarInfo(name=f"{z:0{nd}d}.png")
                        tinfo.size = len(mem_buffer.getbuffer())
                        tf.addfile(tarinfo=tinfo, fileobj=mem_buffer)
        else:
            mask_to_save = Image.fromarray(self.mask_orig, mode="P")
            #mask_to_save = mask_to_save.resize((self.orig_w, self.orig_h), Image.NEAREST) # resize to original (probably not necessary?)
            mask_to_save.putpalette(palette)
            mask_to_save.save(p, pnginfo=metadata)
        
        self.set_modified(False)
        self.set_status("ready", "Ready")

#%% BIOMEDICAL LOAD

    def biomedical_load(self, path=None, add_mask=True):
        '''
        Load a DICOM/NIFTI/NRRD image and define an empty mask on it.
        '''
    
        if self.modified:
            confirm = MultiButtonDialog(self, message="There are unsaved changes. What do you want to do?",
                                        buttons=(("Save changes", "save"), ("Discard changes", "discard"), ("Cancel", None))
                                       )
            answer = confirm.return_value
            if answer == "save":
                self.save_mask()
                self.set_modified(False)
            elif answer == "discard":
                self.set_modified(False)
            else:
                return
    
        self.deactivate_tools()
        self.set_controls_state(False)
        
        # Dialog
        if path is None:
            # Reset path
            self.list_images = None
            self.list_index = 0
            
            p = filedialog.askopenfilename(filetypes=[("Biomedical data files", ("*.dcm", "*.nrrd", "*.nii"))])
            if not p:
                return
            
        else:
            p = path
    
        self.set_status("loading", "Loading volume...")

        # reset masks
        self.clear_all_masks()
        
        metadata, spacing, volume = self._load_medical_volume(p)
        self.biomedical_data["metadata"] = metadata
        self.biomedical_data["spacing"] = spacing
        self.biomedical_data["volume"] = volume
        
        # TODO debug, remove
        # print("Metadata:")
        # print(metadata)
    
        # print("\nSpacing:")
        # print(spacing)
    
        # print("\nVolume shape:")
        # print(volume.shape)
    
        # print("\nData type:")
        # print(volume.dtype)
        
        if volume.shape[2] == 1:
            canvas_frame = "default"
            initial_slice = 0
            self.is_volume_loaded = False
            slice_mask = None
        else:
            canvas_frame = "volume"
            initial_slice = volume.shape[2] // 2
            self.is_volume_loaded = True
            self.canvas_frames["volume"].slider.configure(to=volume.shape[2]-1)
            self.canvas_frames["volume"].slider.set(initial_slice)
            self.zlabel_var.set(f"z: {initial_slice}")
            self.volume_mask = np.zeros(volume.shape, dtype=np.uint8)
            slice_mask = self.volume_mask[..., initial_slice]
        
        # normalize, cut intensity peaks
        # do this also when volume is already uint8
        arr = volume.astype(np.float32)
        v_min, v_max = np.percentile(arr, (1, 99))
        # small check if there are not enough different values (e.g. volumes already representing masks)
        # to avoid division by zero
        # done with np.isclose just for additional safety (v_min==v_max should be sufficient)
        if np.isclose(v_min, v_max):
            v_min, v_max = 0, 1 # do not change array
        self.volume_disp = (255 * np.clip((arr - v_min) / (v_max - v_min), 0, 1)).astype(np.uint8)
        
        if self.is_volume_loaded:
            scale = max(self.volume_disp.shape[0], self.volume_disp.shape[1]) / self.slimtag_config["view"]["preview_dim"]
            new_x = np.linspace(0, self.volume_disp.shape[0]-1, int(self.volume_disp.shape[0] / scale)).astype(np.int32)
            new_y = np.linspace(0, self.volume_disp.shape[1]-1, int(self.volume_disp.shape[1] / scale)).astype(np.int32)
            self.volume_preview = self.volume_disp[np.ix_(new_x, new_y)]

        img = Image.fromarray(self.volume_disp[..., initial_slice]).convert("RGB")
        self.load_image(img, mask=slice_mask, change_canvas=canvas_frame)
        
        
        self.path_original_image = p
        self.quicksave_path = os.path.splitext(p)[0] + "_mask." + ("tar" if self.is_volume_loaded else "png")
        
        self.update_title()

        # reset history
        self.undo_stack.clear()
        
        if add_mask:
            self.add_mask("mask_1")
        
        self.update_display(update_image=True)
        
        self.show_preview_frame("image")
        self.update_preview_frame()
        
        if self.list_images is None:
            self.images_num_label_var.set("Image 1 of 1")
            #self.next_image_btn.configure(state="disabled")
            # TODO image navigation
    
        self.toggle_all_masks_hide(set_hide=False, enabled=True)
        self.toggle_all_masks_lock(set_lock=False, enabled=True)
        
        self.set_status("ready", "Ready")
