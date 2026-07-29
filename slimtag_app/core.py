"""Methods for the core responsibilities of SegmentationApp."""

import os

import customtkinter as ctk
import numpy as np
import tomlkit

import slimtag_wand as wand
from slimtag_app.constants import SAM_MODELS, STATUS_COLOR
from slimtag_utils import MultiButtonDialog, adjust_image


class CoreMixin:
    """Composable core behaviour for ``SegmentationApp``."""

    def async_loader(self): # TODO rethink async_loader
        #print("Loading SAM model")
        self.status_sam_label.configure(text="(Loading image into SAM...)")
        #  Thread-safe upload of shared variable
        with self.lock:
            self.switch_computed_magic_wand = False
            
            if len(self.mask_labels) == 0 or self.active_mask_id is None: # disable all buttons if there are no masks
                self.set_controls_state(False)
            else:
                self.set_controls_state(True)
            
            # apply adjustments to magic wand pre-computation
            image = adjust_image(np.array(self.image_orig), self.wand_brightness, self.wand_contrast, self.wand_gamma)
            
            # compute preprocessing for region growing
            self.region_growing_preprocess = wand.region_growing_preprocessing(image)
            
            # SAM computation
            if self.slimtag_config["modules"]["sam"]:
                if self.sam is not None:
                    wand.sam_preprocessing(image, self.sam)
            
            # Turn on switch
            self.switch_computed_magic_wand = True
            
        #print("Loaded SAM model")

        if len(self.mask_labels) == 0 or self.active_mask_id is None: # disable all buttons if there are no masks
            self.set_controls_state(False)
        else:
            self.set_controls_state(True)

        self.status_sam_label.configure(text="")
        
        if self.list_images != None:
            self.next_image_btn.configure(state="normal")
            
        # Refresh and update display
        self.update_display(update_image=True)
        self.reset_bbox()
        
    
    def sam_loader(self, model_type):
        """
        Load a SAM model.
        
        Here model_type is one of the keys of SAM_MODELS.
        """
        self.set_status("loading", "Loading SAM model...")
        sam = self._segment_anything.sam_model_registry[SAM_MODELS[model_type]["type"]](checkpoint=SAM_MODELS[model_type]["path"])
        sam.to(self.sam_device).eval()
        self.sam = self._segment_anything.SamPredictor(sam)
        self.set_status("ready", "Ready")
    
    def wand_model_select(self, model_type):
        """
        Command for self.wand_model_menu
        """
        if model_type == "Region growing":
            self.wand_threshold = 0.15
            self.wand_edge_tolerance_slider.configure(state="normal",
                                                      button_color=ctk.ThemeManager.theme["CTkSlider"]["button_color"]
                                                      )
            # self.wand_auto_update.configure(state="normal",
            #                                 image=self.icons_dict["AutoUpdate"]["normal"]
            #                                 )
        elif model_type in self.available_sam_models:
            self.wand_threshold = 0.5
            self.wand_edge_tolerance_slider.configure(state="disabled",
                                                      button_color=["gray60", "gray45"]
                                                      )
            # self.wand_auto_update.configure(state="disabled",
            #                                 image=self.icons_dict["AutoUpdate"]["disabled"])
            if model_type != self.last_sam_model:
                self.last_sam_model = model_type
                self.sam_loader(model_type)
                self.async_loader()
        self.wand_threshold_slider.set(self.wand_threshold)
        self.wand_threshold_lbl.configure(text=f"{self.wand_threshold:.2f}")
    
    def load_config_file(self, toml_file):
        """
        Load a TOML file, and return the corresponding tomlkit.TOMLDocument
        object with the "correct" default values.
        
        """
        with open(toml_file, "rb") as config_path:
            cfg = tomlkit.load(config_path)
        
        class Field():
            """
            Dummy class for field descriptor in schema
            """
            def __init__(self, type_, default=..., required=False):
                self.type_ = type_
                self.default = default
                self.required = required
        
        def validate(cfg, schema, path=""):
            """
            Recursively check that the TOML config is well structured
            
            Schema allows to check keys existence, type consistency, and
            either add default values or raise KeyError
            """
            for key, rule in schema.items():
                # update current schema level
                current_path = f"{path}.{key}" if path else key
                # check nested tables and call recursively
                if isinstance(rule, dict):
                    if key not in cfg:
                        cfg[key] = {}
                    validate(cfg[key], rule, current_path)
                    continue
                # if we are at a leaf, check if key exists
                # if not, check Field to determine what to do
                if key not in cfg:
                    if rule.required:
                        raise KeyError(f"{current_path}")
                    elif rule.default is not ...:
                        cfg[key] = rule.default
                        continue
                    else:
                        continue
                # if we are here, we are at a leaf and key exists
                # so we just check consistency with type
                value = cfg[key]
                if not isinstance(value, rule.type_):
                    raise TypeError(f"'{current_path}' should have type {rule.type_.__name__}, got {type(value).__name__} instead")

        # structure to be ckecked
        expected = {
            "main": {
                "appearance": Field(str, default="dark"),
                "undo_depth": Field(int, default=10)
            },
            "modules": {
                "sam": Field(bool, required=True),
                "biomedical": Field(bool, required=True)
            },
            "view": {
                "zoom": {
                    "max_pixel": Field(int, default=32),
                    "min_pixel": Field(int, default=6144)
                },
                "refresh_rate_brush": Field(float, default=0.05),
                "preview_dim": Field(int, default=250)
            },
            "mask": {
                "max_masks": Field(int, default=20),
                "default_mask_colors": Field(list, required=True)
            }
        }
        
        # validate and eventually populate with defaults
        validate(cfg, expected)
        
        # manually check inconsistencies
        if cfg["main"]["appearance"] not in ["light", "dark"]:
            cfg["main"]["appearance"] = "dark"
        
        return cfg
    
    def save_config_file(self):
        pass #TODO
    
    def quit_program(self):
        """
        Quit program.
        """
        if self.modified:
            if self.list_images != None:
                # in folder mode, bypass check and always save changes on the current mask
                self.save_mask(switch_fast=True)
                self.quit()
                self.destroy()
            else:
                confirm = MultiButtonDialog(self, message="There are unsaved changes. What do you want to do?",
                                            buttons=(("Save & Quit", "save"), ("Discard & Quit", "discard"), ("Cancel", None))
                                           )
                answer = confirm.return_value
                if answer == "save":
                    self.save_mask()
                    self.quit()
                    self.destroy()
                elif answer == "discard":
                    self.quit()
                    self.destroy()
                else:
                    return
        else:
            self.quit()
            self.destroy()
    
    #%% STATUS METHODS
    # update window title
    def update_title(self):
        title_string = f"{'*' if self.modified else ''}SLImTAG{f' [{os.path.basename(self.path_original_image)}]' if self.path_original_image is not None else ''}"
        self.title(title_string)

    def image_is_loaded(self):
        '''
        Warning message if no image has been loaded.
        In that case, user can load image from warning dialog
        '''
        if self.image_orig is None:
            warn = MultiButtonDialog(self, message="WARNING: No image loaded",
                                     buttons=[("Import image...", "import"), ("Cancel", None)])
            action = warn.return_value
            if action == "import":
                return self.open_image(add_mask=False)
            else:
                return False
        return True

    def set_status(self, state, text):
        """
        Set icon color and text for status bar.
        """
        try:
            self.status_icon.configure(text_color=STATUS_COLOR[state])
        except KeyError:
            self.status_icon.configure(text_color=STATUS_COLOR["idle"])
        self.status_label.configure(text=text)
        self.update_idletasks()
    
    def set_modified(self, state):
        """
        Check if self.modified is different than state, and in that case update
        """
        if self.modified != state:
            if state == True:
                self.modified = True
            else: # state == False
                self.modified = False
            self.update_title()
    
    #%% APPEARANCE (DARK/LIGHT)
    def toggle_appearance(self):
        ctk.set_appearance_mode(self.slimtag_config["main"]["appearance"])
        '''
        self.set_menu_theme(self.menu_bar, self.slimtag_config["main"]["appearance"])
        for menu in self.topmenu_items:
            self.set_menu_theme(self.topmenu_items[menu], self.slimtag_config["main"]["appearance"])
        if hasattr(self, 'active_context_menu'):
            self.set_menu_theme(self.active_context_menu, self.slimtag_config["main"]["appearance"])
        '''

    def set_menu_theme(self, menu, mode):
        if mode.lower() == 'dark':
            menu.configure(bd=0, background="#242424", fg="#999999", activebackground="#242424", activeforeground="white", activeborderwidth=0)
        else:
            menu.configure(bd=0, background="#d9d9d9", fg="#000000", activebackground="#d9d9d9", activeforeground="#242424", activeborderwidth=0)
