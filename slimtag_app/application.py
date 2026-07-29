'''
SLImTAG: Simple Light-weight Image TAGging tool

SLImTAG is a simple and intuitive GUI tool for interactive image segmentation
integrating several tools such as brushes, connected component selection, and
magic wand selection (both classical and AI-based).

It supports multiple masks with color previews, undo history, and easy load
and save of masks.

v0.1 - 17 Apr 2026

Giulio Del Corso, Oscar Papini, Federico Volpini
'''

#%% Libraries
import os
import warnings

# TkInter and CustomTkInter GUI
import tkinter as tk
import customtkinter as ctk
from CTkMenuBarPlus import CTkMenuBar
import screeninfo

# Image manipulation and TkInter interaction (ImageTk)
from PIL import Image, ImageTk

# Custom utils
from slimtag_utils import ProportionalDropdownMenu, SplashScreen, Tooltip

from slimtag_app.constants import (
    CONFIG_FILE_PATH,
    HIGHLIGHT_COLOR,
    SAM_MODELS,
    STATUS_COLOR,
    STATUS_SYMBOL,
)
from slimtag_app.controls import ControlsMixin
from slimtag_app.core import CoreMixin
from slimtag_app.display import DisplayMixin
from slimtag_app.events import EventsMixin
from slimtag_app.io import IOMixin
from slimtag_app.masks import MasksMixin
from slimtag_app.tools import ToolsMixin

# Asynchronous threading import
import threading

#%% Global parameters

# Suppress all warnings, Only on PROD
# warnings.filterwarnings('ignore')


#%% SLImTAG main class
class SegmentationApp(
    CoreMixin,
    DisplayMixin,
    ControlsMixin,
    MasksMixin,
    IOMixin,
    EventsMixin,
    ToolsMixin,
    ctk.CTk,
):
    def __init__(self):
        super().__init__()
        self.slimtag_config = self.load_config_file(CONFIG_FILE_PATH)
        
        # TODO
        # we're still designing light mode palette and icons,
        # for now hard force dark mode
        self.slimtag_config["main"]["appearance"] = "dark" # TO BE REMOVED WHEN DONE
        
        # TODO move set_appearance_mode to preferences window
        # optionsmenu "dark", "light" with default value: "dark"
        ctk.set_appearance_mode(self.slimtag_config["main"]["appearance"])
        
        self.title("SLImTAG")
        self.geometry("1300x900")
        self.minsize(800, 600)
        self.iconphoto(False, ImageTk.PhotoImage(file=os.path.join("images", "main_icon.png")))
        
        # hide main window and open splash screen
        self.update_idletasks()
        self.withdraw()
        splash = SplashScreen(self)
        
        #%% Optional imports
        
        self._load_medical_volume = None
        self._torch = None
        self._segment_anything = None
        
        if self.slimtag_config["modules"]["biomedical"]: # Custom biomedical utils
            try:
                from slimtag_biomedical import load_medical_volume
                self._load_medical_volume = load_medical_volume
            except ModuleNotFoundError:
                warnings.warn("libraries for 'biomedical' not found, 'biomedical = True' will be ignored")
                self.slimtag_config["modules"]["biomedical"] = False
        
        if self.slimtag_config["modules"]["sam"]: # SAM segmentation models
            # Torch and SAM (Segment anything model)
            try:
                import torch
                self._torch = torch
                import segment_anything
                self._segment_anything = segment_anything
                # Suppress specific PyTorch warnings
                warnings.filterwarnings(
                    "ignore",
                    message="You are using `torch.load` with `weights_only=False`"
                )
            except ModuleNotFoundError:
                warnings.warn("libraries for 'sam' not found, 'sam = True' will be ignored")
                self.slimtag_config["modules"]["sam"] = False

        #%% Attributes
        
        # Full image and mask
        self.image_orig = None
        self.mask_orig = None
        # Displayed image and mask
        self.image_disp = None
        self.mask_disp = None
        # blended image+mask for fast pan&zoom
        self.blended = None
        # current image preview (in sub canvas)
        self.current_preview_canvas = None
        self.preview_scale = 1.0
        # Matrix of locked masks
        self.mask_locked = None
        
        # Biomedical dictionary
        self.biomedical_data = {"metadata": None, "spacing": None, "volume": None}

        # aux display variables
        self.tk_ov = None
        self.sam_preview_pil = None
        self.tk_sam_preview = None
        self.volume_disp = None # volume display casted as uint8 array
        self.volume_mask = None # 3D numpy array for volume masks
        self.volume_preview = None # resized volume for fast slider preview
        self.volume_zslider = None
        self.zslider_preview = None # TopLevel object that contains slider preview
        self.zslider_preview_img = None # preview image, to prevent it from being garbage collected
        self.is_volume_loaded = False # boolean switch to check if a volume is loaded
        
        # to keep track of delayed events
        self.resizing_event = None
        self.zoom_event_id = None
        self.update_opacity_id = None
        
        # boolean switch to check if mask is modified and not saved
        # TODO: for multiple images import
        self.modified = False
        
        # zoom & pan status
        self.zoom = 1.0
        self._pan_start = None
        monitor_dims = sum([[m.width, m.height] for m in screeninfo.get_monitors()], [])
        self.min_monitor_dim = min(monitor_dims)
        self.max_monitor_dim = max(monitor_dims)
        # Define a max and min zoom
        self.zoom_max = self.min_monitor_dim / self.slimtag_config["view"]["zoom"]["max_pixel"]
        self.zoom_min = self.max_monitor_dim / self.slimtag_config["view"]["zoom"]["min_pixel"]
        
        # labels for zoom and mouse position
        self.pos_label_var = tk.StringVar(self, value="| x: 0 | y: 0 |")
        self.zlabel_var = tk.StringVar(self, value="z: 0")
        self.zoom_label_var = tk.StringVar(self, value="Zoom: 100%")
        
        # Original values for rescale
        self.orig_h = None
        self.orig_w = None

        # Masks stuff
        self.mask_labels = {}
        self.mask_colors = {}
        self.mask_widgets = {}
        self.active_mask_id = None
        self.mask_opacity = 150 # [0-255]
        self.mask_outline = tk.BooleanVar(self, value=False) # use outlined masks instead of filled ones
        
        # List images and index for folder segmentation
        self.list_images = None
        self.list_index = None
        self.path_aux_save = None
        self.path_original_image = None
        self.quicksave_path = None
        
        self.images_num_label_var = tk.StringVar(self, value="Image 0 of 0")

        # mouse position for events that need it
        self.mouse = {'x': None, 'y': None}
        splash.step(10)
        
        # tools
        self.tools = ["brush", "eraser", "polygon", "bbox", "cut", "clean", "bucket", "undo",
                      "smooth", "fill", "denoise", "interpolate",
                      "wand", "wand_all", "wand_multi", "wand_box",
                      "ruler", "area",
                      "custom_1", "custom_2", "custom_3", "custom_4"]
        # tools buttons
        self.tool_btn = {}
        # tools status
        self.tool_active = {tool: False for tool in self.tools}
        # tools icons
        self.tool_icon = {}
        
        for tool in self.tools:
            # TODO change wirh f"images/buttons/{tool}_light_on.png"
            self.tool_icon[tool] = {"normal": ctk.CTkImage(light_image=Image.open(f"images/buttons/{tool}_light_on.png").convert("RGBA"),
                                                           dark_image=Image.open(f"images/buttons/{tool}_dark_on.png").convert("RGBA"),
                                                           size=(31, 31)),
                                    "disabled": ctk.CTkImage(light_image=Image.open(f"images/buttons/{tool}_light_off.png").convert("RGBA"),
                                                             dark_image=Image.open(f"images/buttons/{tool}_dark_off.png").convert("RGBA"),
                                                             size=(31, 31))
                                    }
        # map tool -> corresponding options frame
        self.tool_opt_map = {}
        self.tool_opt_map.update(dict.fromkeys(["brush", "eraser"], "brush"))
        self.tool_opt_map.update(dict.fromkeys(["wand", "wand_all", "wand_multi", "wand_box"], "wand"))
        self.tool_opt_map.update(dict.fromkeys(["smooth"], "smooth"))
        # TODO tool frame for each tool
        self.tool_opt_map.update(dict.fromkeys(["polygon", "bbox", "cut", "clean", "bucket",
                                                "fill", "denoise", "interpolate",
                                                "ruler", "area"], "empty"))
        # TODO create custom empty frames, one for each custom button
        self.tool_opt_map["custom_1"] = "empty"
        self.tool_opt_map["custom_2"] = "empty"
        self.tool_opt_map["custom_3"] = "empty"
        self.tool_opt_map["custom_4"] = "empty"

        # brush control
        self.last_brush_pos = None
        self.brush_shape = "Circle"
        self.brush_size = 30
        self.brush_rot = 0
        self.brush_line_ratio = 8 # 'Line' shape is a rectangle with dimension (self.brush_size)x((self.brush_size)/(2*self.brush_line_ratio))
        self.brush_rot_delta = 10 # number of degrees added to or subracted from self.brush_rot for each step of mouse wheel (for Ctrl+Wheel rotation)

        # smooth control
        self.smooth_iter = 1 # number of iterations of outer cycle
        self.smooth_n_erosions = 1
        self.smooth_n_dilations = 1
        
        # undo list
        self.undo_stack = []
        
        # Position top left of the view (in pixels of the original image)
        # please note that these are NOT bounded to image size
        self.view_x = None
        self.view_y = None
        # Corresponding view size (width, height; pixels of the original image)
        self.view_w = None
        self.view_h = None

        # SAM management
        self.sam = None
        self.last_sam_model = None # prevent model reload if the same model is chosen
        self.sam_device = None
        self.available_sam_models = []
        if self.slimtag_config["modules"]["sam"]:
            self.sam_device = "cuda" if torch.cuda.is_available() else "cpu"
            for sam_model in SAM_MODELS:
                if os.path.exists(SAM_MODELS[sam_model]["path"]):
                    self.available_sam_models.append(sam_model)
        self.sam_points = []
        self.sam_pt_labels = []
        self.sam_preview = None # boolean matrix for multipoint SAM preview
        # to store IDs of <Return> and <Escape> events for multipoint SAM tool
        self.sam_bind_enter = None
        self.sam_bind_esc = None
        
        # SAM preprocessing (for sliders: range -100 .. 100)
        self.wand_brightness = 0
        self.wand_contrast = 0
        self.wand_gamma = 0
        
        # Magic wand threshold (e.g. SAM model threshold), range 0.0 .. 1.0
        self.wand_threshold = 0.15 # region growing has 0.15 as default value (SAM instead has 0.5)
        # Region growing edge tolerance
        self.wand_edge_tolerance = 0.5
        
        self.region_growing_preprocess = None
        
        # boolean to track if buttons are pressed
        self.b3_pressed = False # right mouse button
        self.mid_pressed = False # middle mouse button
        self.shift_pressed = False # shift (any)
        
        # asynchronous mechanism to speed up image loading
        self.switch_computed_magic_wand = False     # True if SAM is loaded
        self.thread = None                          # Threading variable
        self.lock = threading.Lock()              # To protect shared varaibles

        splash.step(10)
        
        
        # dictionary for icons
        self.icons_dict = {}
        for img in ["Eye", "Lock"]:
            for st in ["Open", "Closed"]:
                self.icons_dict[f"{img}{st}"] = {"normal": ctk.CTkImage(light_image=Image.open(f"images/icons/{img}{st}_light_on.png").convert("RGBA"),
                                                                        dark_image=Image.open(f"images/icons/{img}{st}_dark_on.png").convert("RGBA"),
                                                                        size=(16, 16)),
                                                 "disabled": ctk.CTkImage(light_image=Image.open(f"images/icons/{img}{st}_light_off.png").convert("RGBA"),
                                                                          dark_image=Image.open(f"images/icons/{img}{st}_dark_off.png").convert("RGBA"),
                                                                          size=(16, 16))
                                                 }
        for img in ["NewMask", "ManualUpdate", "AutoUpdate"]:
            self.icons_dict[f"{img}"] = {"normal": ctk.CTkImage(light_image=Image.open(f"images/icons/{img}_light_on.png").convert("RGBA"),
                                                                dark_image=Image.open(f"images/icons/{img}_dark_on.png").convert("RGBA"),
                                                                size=(16, 16)),
                                         "disabled": ctk.CTkImage(light_image=Image.open(f"images/icons/{img}_light_off.png").convert("RGBA"),
                                                                  dark_image=Image.open(f"images/icons/{img}_dark_off.png").convert("RGBA"),
                                                                  size=(16, 16))
                                         }
        
        self.icons_dict["MenuIcon"] = ctk.CTkImage(light_image=Image.open("images/icons/MenuIcon.png").convert("RGBA"), size=(31, 31))
        
        splash.step(5)

        #%% Top Menu
        
        self.menu_bar = CTkMenuBar(self, bg_color=ctk.ThemeManager.theme["CTkTextbox"]["fg_color"][1])
        
        # File
        file_button = self.menu_bar.add_cascade("File")
        file_menu = ProportionalDropdownMenu(widget=file_button)
        file_menu.add_option("Exit Program", command=self.quit_program, accelerator="Ctrl+Q")
        file_menu.build_menu()

        # Edit
        edit_button = self.menu_bar.add_cascade("Edit")
        edit_menu = ProportionalDropdownMenu(widget=edit_button)
        edit_menu.add_option("Undo", command=self.undo, accelerator="Ctrl+Z", tabs=2)
        # TODO implement preferences window
        edit_menu.add_option("Preferences", command=None, state="disabled")
        edit_menu.build_menu()

        # View
        view_button = self.menu_bar.add_cascade("View")
        view_menu = ProportionalDropdownMenu(widget=view_button)
        view_menu.add_option("Zoom in", command=self.zoom_in, accelerator="Ctrl++", tabs=2)
        view_menu.add_option("Zoom out", command=self.zoom_out, accelerator="Ctrl+-", tabs=2)
        view_menu.add_option("Reset zoom", command=self.reset_zoom, accelerator="Ctrl+0")
        view_menu.build_menu()

        # Image
        image_button = self.menu_bar.add_cascade("Image")
        image_menu = ProportionalDropdownMenu(widget=image_button)
        image_menu.add_option("Import image", command=self.open_image, accelerator="Ctrl+I")
        # TODO reactivate import folder
        image_menu.add_option("Import folder", command=self.load_folder, accelerator="Ctrl+F", state="disabled")
        image_menu.build_menu()

        # Mask
        mask_button = self.menu_bar.add_cascade("Mask")
        mask_menu = ProportionalDropdownMenu(widget=mask_button)
        mask_menu.add_option("Load mask", command=self.load_mask, accelerator="")
        mask_menu.add_option("Save mask", command=lambda s=True: self.save_mask(switch_fast=s), accelerator="Ctrl+S")
        mask_menu.add_option("Save mask as...", command=lambda s=False: self.save_mask(switch_fast=s))
        mask_menu.add_separator()
        mask_menu.add_option("Clear active mask", command=self.clear_active_mask)
        mask_menu.add_option("Clear all masks", command=self.clear_all_masks)
        mask_menu.build_menu()

        # Wand
        wand_button = self.menu_bar.add_cascade("Magic wand")
        wand_menu = ProportionalDropdownMenu(widget=wand_button)
        # TODO implement load/save configuration
        wand_menu.add_option("Load configuration", command=None, state="disabled")
        wand_menu.add_option("Save configuration", command=None, state="disabled")
        wand_menu.build_menu()
        
        # Help
        help_button = self.menu_bar.add_cascade("Help")
        help_menu = ProportionalDropdownMenu(widget=help_button)
        # TODO implement help functions
        help_menu.add_option("Documentation", command=None, state="disabled")
        help_menu.add_option("About", command=None, state="disabled")
        help_menu.build_menu()
        
        # (Optional) Biomedical
        if self.slimtag_config["modules"]["biomedical"]:
            bio_button = self.menu_bar.add_cascade("Biomedical tools",text_color=HIGHLIGHT_COLOR)
            bio_menu = ProportionalDropdownMenu(widget=bio_button)
            bio_menu.add_option("Import NRRD/NIFTI/DICOM", command=self.biomedical_load)
            bio_menu.build_menu()

        #%% Main UI elements
        self.main_container = ctk.CTkFrame(self, fg_color="transparent", corner_radius=0)
        self.main_container.pack(fill="both", expand=True) # forced to use .pack() by CTkMenuBarPlus.CTkMenuBar
        self.main_container.pack_propagate(False)
        
        panels_width = 250
        # Left panel for tools
        self.left_panel = ctk.CTkFrame(self.main_container, width=panels_width, corner_radius=0)
        self.left_panel.grid(row=0, column=0, sticky="nsew")
        self.left_panel.grid_rowconfigure(5, weight=1)
        
        # Main canvas
        # TODO different frames with different widgets depending on load type
        # e.g. previous/next image for folder, slider with z-axis for medical...
        self.main_canvas_frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.main_canvas_frame.grid(row=0, column=1, sticky="nsew")
        self.main_canvas_frame.grid_rowconfigure(0, weight=1)
        self.main_canvas_frame.grid_columnconfigure(0, weight=1)
        
        # Right panel for masks
        self.right_panel = ctk.CTkFrame(self.main_container, width=panels_width, corner_radius=0)
        self.right_panel.grid(row=0, column=2, sticky="nsew")
        self.right_panel.grid_rowconfigure(1, weight=1)
        self.right_panel.grid_rowconfigure(2, weight=2)
        
        # Statusbar
        self.statusbar = ctk.CTkFrame(self.main_container, height=32, fg_color=("gray92", "gray14"), corner_radius=0)
        self.statusbar.grid(row=1, column=0, columnspan=3, sticky="nsew", padx=0, pady=0)
        self.statusbar.grid_columnconfigure(3, weight=1)
        
        # Grid configuration for main window
        self.main_container.grid_columnconfigure(1, weight=1)
        self.main_container.grid_rowconfigure(0, weight=1)
        
        #%% Left panel: Tools
        # Frame for main menu
        # self.main_menu_frame = ctk.CTkFrame(self.left_panel, corner_radius=0)
        # self.main_menu_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=(0, 5))
        
        # # label created as fake button for consistency reasons
        # self.main_menu_label = ctk.CTkButton(self.main_menu_frame,
        #                                      width=44, height=44,
        #                                      text="",
        #                                      fg_color="transparent",
        #                                      border_width=0,
        #                                      hover=False,
        #                                      state="disabled",
        #                                      command=None)
        # self.main_menu_label.grid(row=0, column=0, sticky="nsew", padx=(4, 2), pady=4)
        # # actual main menu
        # self.main_menu_btn = ctk.CTkButton(self.main_menu_frame,
        #                                    width=44, height=44,
        #                                    text="", image=self.icons_dict["MenuIcon"],
        #                                    fg_color="transparent",
        #                                    command=None)
        # self.main_menu_btn.grid(row=0, column=1, sticky="nsew", padx=(2, 4), pady=4)
        
        # Frames for buttons
        self.tools_btn_frame = {i: ctk.CTkFrame(self.left_panel, corner_radius=0) for i in range(5)}
        frame_paddings = [(0, 5)] + 3*[5] + [(5, 0)]
        for i in range(5):
            self.tools_btn_frame[i].grid(row=i, column=0, sticky="nsew", padx=0, pady=frame_paddings[i])
        
        # Buttons
        # TODO all commands, in particular add the "right-click" that are a different tool now
        # I think we should keep the right-click behaviour in any case (for "pro users")
        self.create_tool_button("brush", 0, 0, 0, help_text="Brush [B]")
        self.create_tool_button("eraser", 0, 0, 1, help_text="Eraser")
        self.create_tool_button("polygon", 0, 1, 0)
        self.create_tool_button("bbox", 0, 1, 1, help_text="Rectangiular Mask")
        self.create_tool_button("cut", 0, 2, 0, help_text="Cut component [C]")
        self.create_tool_button("clean", 0, 2, 1, help_text="Keep component")
        self.create_tool_button("bucket", 0, 3, 0, last_row=True)
        self.create_tool_button("undo", 0, 3, 1, command=self.undo, last_row=True, help_text="Undo [Ctrl-Z]")
        self.create_tool_button("smooth", 1, 0, 0, help_text="Smooth [S]")
        self.create_tool_button("fill", 1, 0, 1, help_text="Fill holes")
        self.create_tool_button("denoise", 1, 1, 0, last_row=True)
        self.create_tool_button("interpolate", 1, 1, 1, last_row=True)
        self.create_tool_button("wand", 2, 0, 0, help_text="Magic wand [M]")
        self.create_tool_button("wand_all", 2, 0, 1)
        self.create_tool_button("wand_multi", 2, 1, 0, last_row=True, help_text="Multipoint magic wand")
        self.create_tool_button("wand_box", 2, 1, 1, None, last_row=True)
        self.create_tool_button("ruler", 3, 0, 0, None, last_row=True)
        self.create_tool_button("area", 3, 0, 1, None,last_row=True)
        self.create_tool_button("custom_1", 4, 0, 0, None)
        self.create_tool_button("custom_2", 4, 0, 1, None)
        self.create_tool_button("custom_3", 4, 1, 0, None, last_row=True)
        self.create_tool_button("custom_4", 4, 1, 1, None, last_row=True)
        
        #%% Main view: Canvas
        self.canvas_frames = {}
        
        default_canvas_frame = ctk.CTkFrame(self.main_canvas_frame, fg_color="transparent")
        default_canvas_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        default_canvas_frame.grid_rowconfigure(0, weight=1)
        default_canvas_frame.grid_columnconfigure(0, weight=1)
        default_canvas_frame.canvas = ctk.CTkCanvas(default_canvas_frame, bg="black", highlightthickness=0)
        default_canvas_frame.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas_frames["default"] = default_canvas_frame
        
        volume_canvas_frame = ctk.CTkFrame(self.main_canvas_frame, fg_color="transparent")
        volume_canvas_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        volume_canvas_frame.grid_rowconfigure(0, weight=1)
        volume_canvas_frame.grid_columnconfigure(0, weight=1)
        volume_canvas_frame.canvas = ctk.CTkCanvas(volume_canvas_frame, bg="black", highlightthickness=0)
        volume_canvas_frame.canvas.grid(row=0, column=0, sticky="nsew")
        slider_frame = ctk.CTkFrame(volume_canvas_frame, corner_radius=0)
        slider_frame.grid(row=1, column=0, sticky="nsew", padx=0, pady=0)
        slider_frame.grid_columnconfigure(0, weight=1)
        volume_canvas_frame.slider = ctk.CTkSlider(slider_frame, from_=0, to=1, command=lambda v: self.on_zslider_move(round(v)))
        volume_canvas_frame.slider.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        volume_canvas_frame.slider.set(0)
        volume_canvas_frame.slider.bind("<Button-1>", self.start_zlider_preview)
        volume_canvas_frame.slider.bind("<B1-Motion>", self.move_zslider_preview)
        volume_canvas_frame.slider.bind("<ButtonRelease-1>", self.end_zslider_preview)
        volume_canvas_frame.zlabel = ctk.CTkLabel(slider_frame, textvariable=self.zlabel_var, anchor="w", width=40)
        volume_canvas_frame.zlabel.grid(row=0, column=1, sticky="e", padx=(0, 10))
        self.canvas_frames["volume"] = volume_canvas_frame
        
        # default view
        self.canvas = None
        self.show_canvas_frame("default")
        
        #%% Right panel: Masks
        # Global masks buttons
        self.mask_controls_frame = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.mask_controls_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.mask_controls_frame.grid_columnconfigure(1, weight=1)
        
        self.new_mask_btn = ctk.CTkButton(self.mask_controls_frame, text="New mask",
                                          image=self.icons_dict["NewMask"]["normal"],
                                          width=0, height=34,
                                          anchor="w",
                                          fg_color="transparent",
                                          command=self.add_mask)
        Tooltip(self.new_mask_btn, text="Add new mask [N]")
        self.new_mask_btn.grid(row=0, column=0, sticky="w", padx=(10, 5), pady=5)
        self.hide_all_mask_btn = ctk.CTkButton(self.mask_controls_frame, text="",
                                               image=self.icons_dict["EyeOpen"]["disabled"],
                                               width=34, height=34,
                                               fg_color="transparent",
                                               state="disabled",
                                               command=lambda: self.toggle_all_masks_hide(not self.hide_all_mask_btn.hidden))
        self.hide_all_mask_btn.grid(row=0, column=2, sticky="ew", padx=(5, 2), pady=5)
        self.hide_all_mask_btn.hidden = False
        Tooltip(self.hide_all_mask_btn, text="Hide all masks")
        self.lock_all_mask_btn = ctk.CTkButton(self.mask_controls_frame, text="",
                                          image=self.icons_dict["LockOpen"]["disabled"],
                                          width=34, height=34,
                                          fg_color="transparent",
                                          state="disabled",
                                          command=lambda: self.toggle_all_masks_lock(not self.lock_all_mask_btn.locked))
        self.lock_all_mask_btn.grid(row=0, column=3, sticky="ew", padx=2, pady=5)
        self.lock_all_mask_btn.locked = False
        Tooltip(self.lock_all_mask_btn, text="Lock all masks")
        self.clear_all_mask_btn = ctk.CTkButton(self.mask_controls_frame, text="×",
                                  font=ctk.CTkFont(size=24, weight="bold"),
                                  width=34, height=34,
                                  fg_color="transparent",
                                  text_color="#AB2B22",
                                  command=self.clear_all_masks)
        self.clear_all_mask_btn.bind("<Enter>", lambda e: self.clear_all_mask_btn.configure(fg_color="#AB2B22", text_color="white"))
        self.clear_all_mask_btn.bind("<Leave>", lambda e: self.clear_all_mask_btn.configure(fg_color="transparent", text_color="#AB2B22"))
        self.clear_all_mask_btn.grid(row=0, column=4, sticky="ew", padx=(2, 23), pady=5)
        Tooltip(self.clear_all_mask_btn, text="Clear all masks")
        
        # ScrollFrame for mask list
        self.mask_list_frame = ctk.CTkScrollableFrame(self.right_panel, corner_radius=0, height=100)
        self.mask_list_frame.grid(row=1, column=0, sticky="nsew", padx=0, pady=(0, 5))
        self.mask_list_frame._scrollbar.configure(height=0) # https://stackoverflow.com/a/76957827
        
        splash.step(15) 
        
        #%% Right panel: Tools options
        # Frame for tool options
        self.tool_opt_container = ctk.CTkScrollableFrame(self.right_panel, corner_radius=0)
        self.tool_opt_container.grid(row=2, column=0, sticky="nsew", padx=0, pady=5)
        self.tool_opt_container.grid_columnconfigure(0, weight=1)
        
        self.tool_opt_frame = {}
        
        empty_frame = ctk.CTkFrame(self.tool_opt_container, fg_color="transparent")
        empty_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)# pady=5
        self.tool_opt_frame["empty"] = empty_frame
        
        brush_frame = ctk.CTkFrame(self.tool_opt_container, fg_color="transparent")
        brush_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.tool_opt_frame["brush"] = brush_frame
        
        wand_frame = ctk.CTkFrame(self.tool_opt_container, fg_color="transparent")
        wand_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.tool_opt_frame["wand"] = wand_frame
        
        ccomp_frame = ctk.CTkFrame(self.tool_opt_container, fg_color="transparent")
        ccomp_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.tool_opt_frame["ccomp"] = ccomp_frame
        
        smooth_frame = ctk.CTkFrame(self.tool_opt_container, fg_color="transparent")
        smooth_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.tool_opt_frame["smooth"] = smooth_frame
        
        for tool in self.tool_opt_frame:
            self.tool_opt_frame[tool].grid_columnconfigure(0, weight=1)
        
        # set empty frame at start
        self.current_tool_frame = None
        self.show_tool_frame("empty")
        
        splash.step(20)
        
        # Brush options
        ctk.CTkLabel(self.tool_opt_frame["brush"], text="Brush settings:", fg_color="transparent", font=ctk.CTkFont(size=17, weight='bold'), anchor="w").grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=(10, 0))
        ctk.CTkLabel(self.tool_opt_frame["brush"], text="Shape", fg_color="transparent", anchor="w").grid(row=1, column=0, sticky="ew", padx=10, pady=(10, 2))
        self.brush_shape_btn = ctk.CTkSegmentedButton(self.tool_opt_frame["brush"], values=["Circle", "Square", "Line"], command=lambda v: (setattr(self, "brush_shape", v), print (self.brush_shape)));
        self.brush_shape_btn.set(self.brush_shape)
        self.brush_shape_btn.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=0)
        
        ctk.CTkLabel(self.tool_opt_frame["brush"], text="Size", fg_color="transparent", anchor="w").grid(row=3, column=0, sticky="ew", padx=(10, 5), pady=(10, 2)) #font=ctk.CTkFont(size=11),
        self.brush_size_lbl = ctk.CTkLabel(self.tool_opt_frame["brush"], text=str(self.brush_size), fg_color="transparent", anchor="e")
        self.brush_size_lbl.grid(row=3, column=1, sticky="ew", padx=(5, 10), pady=(10, 2))
        self.brush_size_slider = ctk.CTkSlider(self.tool_opt_frame["brush"], from_=5, to=100,
                                               command=lambda v: (setattr(self,"brush_size",int(v)), self.brush_size_lbl.configure(text=str(self.brush_size))))
        self.brush_size_slider.set(self.brush_size)
        self.brush_size_slider.grid(row=4, column=0, columnspan=2, sticky="ew", padx=10, pady=0)
        
        ctk.CTkLabel(self.tool_opt_frame["brush"], text="Rotation", fg_color="transparent", anchor="w").grid(row=5, column=0, sticky="ew", padx=(10, 5), pady=(10, 2))
        self.brush_rot_lbl = ctk.CTkLabel(self.tool_opt_frame["brush"], text=f"{self.brush_rot}°", fg_color="transparent", anchor="e")
        self.brush_rot_lbl.grid(row=5, column=1, sticky="ew", padx=(5, 10), pady=(10, 2))
        self.brush_rot_slider = ctk.CTkSlider(self.tool_opt_frame["brush"], from_=0, to=180,
                                              command=lambda v: self.set_brush_rotation_slider(v))
        self.update_brush_rotation_slider()
        self.brush_rot_slider.grid(row=6, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))
        
        # Magic wand options
        wand_models = ["Region growing"] + self.available_sam_models
        ctk.CTkLabel(self.tool_opt_frame["wand"], text="Magic wand settings:", fg_color="transparent", font=ctk.CTkFont(size=17, weight='bold'), anchor="w").grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=(10, 0))
        self.wand_model_menu = ctk.CTkOptionMenu(self.tool_opt_frame["wand"], values=wand_models, command=self.wand_model_select)
        self.wand_model_menu.set("Region growing")
        self.wand_model_menu.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=(10, 0))
        Tooltip(self.wand_model_menu, text="Magic wand method")
        
        self.wand_adj_frame = ctk.CTkFrame(self.tool_opt_frame["wand"], border_width=1)
        self.wand_adj_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=10, pady=(10, 0))
        ctk.CTkLabel(self.wand_adj_frame, text="Preprocessing", fg_color="transparent", font=ctk.CTkFont(weight='bold')).grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=(3,0))
        ctk.CTkLabel(self.wand_adj_frame, text="Brightness", fg_color="transparent", anchor="w").grid(row=1, column=0, sticky="ew", padx=(10,0), pady=(3,0))
        self.wand_brightness_lbl = ctk.CTkLabel(self.wand_adj_frame, text=str(self.wand_brightness), fg_color="transparent", anchor="e")
        self.wand_brightness_lbl.grid(row=1, column=1, sticky="ew", padx=(0,10), pady=(3,0))
        ctk.CTkLabel(self.wand_adj_frame, text="Contrast", fg_color="transparent", anchor="w").grid(row=2, column=0, sticky="ew", padx=(10,0), pady=(3,0))
        self.wand_contrast_lbl = ctk.CTkLabel(self.wand_adj_frame, text=str(self.wand_contrast), fg_color="transparent", anchor="e")
        self.wand_contrast_lbl.grid(row=2, column=1, sticky="ew", padx=(0,10), pady=(3,0))
        ctk.CTkLabel(self.wand_adj_frame, text="Shadows", fg_color="transparent", anchor="w").grid(row=3, column=0, sticky="ew", padx=(10,0), pady=(3,0))
        self.wand_gamma_lbl = ctk.CTkLabel(self.wand_adj_frame, text=str(self.wand_gamma), fg_color="transparent", anchor="e")
        self.wand_gamma_lbl.grid(row=3, column=1, sticky="ew", padx=(0,10), pady=(3,0))

        self.wand_auto_update = ctk.CTkButton(self.wand_adj_frame, text="Auto", image=self.icons_dict["AutoUpdate"]["normal"], command=self.wand_update_bayesian)
        self.wand_auto_update.grid(row=4, column=1, sticky="ew", padx=(5, 10), pady=(3,10))
        Tooltip(self.wand_auto_update, text="Auto compute preprocessing parameters")
        self.wand_manual_update = ctk.CTkButton(self.wand_adj_frame, text="Manual", image=self.icons_dict["ManualUpdate"]["normal"], command=self.manual_wand_preprocessing)
        self.wand_manual_update.grid(row=4, column=0, sticky="ew", padx=(10, 5), pady=(3,10))
        Tooltip(self.wand_manual_update, text="Select preprocessing parameters")
        
        self.wand_adj_frame.grid_columnconfigure([0, 1], weight=1)
        
        ctk.CTkLabel(self.tool_opt_frame["wand"], text="Wand threshold", fg_color="transparent", anchor="w").grid(row=3, column=0, sticky="ew", padx=(10, 5), pady=(10, 2))
        self.wand_threshold_lbl = ctk.CTkLabel(self.tool_opt_frame["wand"], text=f"{self.wand_threshold:.2f}", fg_color="transparent", anchor="e")
        self.wand_threshold_lbl.grid(row=3, column=1, sticky="ew", padx=(5, 10), pady=(10, 2))
        self.wand_threshold_slider = ctk.CTkSlider(self.tool_opt_frame["wand"], from_=0.0, to=1.0,
                                                   command=lambda v: (setattr(self,"wand_threshold",float(v)), self.wand_threshold_lbl.configure(text=f"{self.wand_threshold:.2f}")))
        self.wand_threshold_slider.set(self.wand_threshold)
        self.wand_threshold_slider.grid(row=4, column=0, columnspan=2, sticky="ew", padx=10, pady=0)
        
        ctk.CTkLabel(self.tool_opt_frame["wand"], text="Edge tolerance", fg_color="transparent", anchor="w").grid(row=5, column=0, sticky="ew", padx=(10, 5), pady=(10, 2))
        self.wand_edge_tolerance_lbl = ctk.CTkLabel(self.tool_opt_frame["wand"], text=f"{self.wand_edge_tolerance:.2f}", fg_color="transparent", anchor="e")
        self.wand_edge_tolerance_lbl.grid(row=5, column=1, sticky="ew", padx=(5, 10), pady=(10, 2))
        self.wand_edge_tolerance_slider = ctk.CTkSlider(self.tool_opt_frame["wand"], from_=0.0, to=1.0,
                                                   command=lambda v: (setattr(self,"wand_edge_tolerance",float(v)), self.wand_edge_tolerance_lbl.configure(text=f"{self.wand_edge_tolerance:.2f}")))
        self.wand_edge_tolerance_slider.set(self.wand_edge_tolerance)
        self.wand_edge_tolerance_slider.grid(row=6, column=0, columnspan=2, sticky="ew", padx=10, pady=0)
        
        # Smoothing options
        ctk.CTkLabel(self.tool_opt_frame["smooth"], text="Smoothing settings:", fg_color="transparent", font=ctk.CTkFont(size=17, weight='bold'), anchor="w").grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=(10, 0))
        ctk.CTkLabel(self.tool_opt_frame["smooth"], text="Number of iterations", fg_color="transparent", anchor="w").grid(row=1, column=0, sticky="ew", padx=(10, 5), pady=(10, 2))
        self.smooth_iter_lbl = ctk.CTkLabel(self.tool_opt_frame["smooth"], text=str(self.smooth_iter), fg_color="transparent", anchor="e")
        self.smooth_iter_lbl.grid(row=1, column=1, sticky="ew", padx=(5, 10), pady=(10, 2))
        self.smooth_iter_slider = ctk.CTkSlider(self.tool_opt_frame["smooth"], from_=1, to=5,
                                                number_of_steps=4,
                                                command=lambda v: (setattr(self,"smooth_iter",int(v)), self.smooth_iter_lbl.configure(text=str(self.smooth_iter))))
        self.smooth_iter_slider.set(self.smooth_iter)
        self.smooth_iter_slider.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=0)
        ctk.CTkLabel(self.tool_opt_frame["smooth"], text="Number of erosion steps", fg_color="transparent", anchor="w").grid(row=3, column=0, sticky="ew", padx=(10, 5), pady=(10, 2))
        self.smooth_erosion_lbl = ctk.CTkLabel(self.tool_opt_frame["smooth"], text=str(self.smooth_n_erosions), fg_color="transparent", anchor="e")
        self.smooth_erosion_lbl.grid(row=3, column=1, sticky="ew", padx=(5, 10), pady=(10, 2))
        self.smooth_erosion_slider = ctk.CTkSlider(self.tool_opt_frame["smooth"], from_=0, to=10,
                                                   number_of_steps=10,
                                                   command=lambda v: (setattr(self,"smooth_n_erosions",int(v)), self.smooth_erosion_lbl.configure(text=str(self.smooth_n_erosions))))
        self.smooth_erosion_slider.set(self.smooth_n_erosions)
        self.smooth_erosion_slider.grid(row=4, column=0, columnspan=2, sticky="ew", padx=10, pady=0)
        ctk.CTkLabel(self.tool_opt_frame["smooth"], text="Number of dilation steps", fg_color="transparent", anchor="w").grid(row=5, column=0, sticky="ew", padx=(10, 5), pady=(10, 2))
        self.smooth_dilation_lbl = ctk.CTkLabel(self.tool_opt_frame["smooth"], text=str(self.smooth_n_dilations), fg_color="transparent", anchor="e")
        self.smooth_dilation_lbl.grid(row=5, column=1, sticky="ew", padx=(5, 10), pady=(10, 2))
        self.smooth_dilation_slider = ctk.CTkSlider(self.tool_opt_frame["smooth"], from_=0, to=10,
                                                    number_of_steps=10,
                                                    command=lambda v: (setattr(self,"smooth_n_dilations",int(v)), self.smooth_dilation_lbl.configure(text=str(self.smooth_n_dilations))))
        self.smooth_dilation_slider.set(self.smooth_n_dilations)
        self.smooth_dilation_slider.grid(row=6, column=0, columnspan=2, sticky="ew", padx=10, pady=0)

        splash.step(10)

        #%% Right panel: Navigation
        self.navigation_frame = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.navigation_frame.grid(row=3, column=0, sticky="n", padx=10, pady=(5, 10))

        self.sub_canvas_frames = {}
        preview_dim = self.slimtag_config["view"]["preview_dim"]
        
        image_only_frame = ctk.CTkFrame(self.navigation_frame)#, fg_color="transparent")
        image_only_frame.canvas = ctk.CTkCanvas(image_only_frame, bg="black", highlightthickness=0, width=preview_dim, height=preview_dim)
        image_only_frame.canvas.grid(row=0, column=0, sticky="sew", padx=5, pady=5)
        image_only_frame.grid_rowconfigure(0, weight=1)
        image_only_frame.grid_columnconfigure(0, weight=1)
        self.sub_canvas_frames["image"] = image_only_frame
        
        ortho_views_frame = ctk.CTkFrame(self.navigation_frame)#, fg_color="transparent")
        ortho_views_frame.view1 = ctk.CTkCanvas(ortho_views_frame, bg="black", highlightthickness=0, width=preview_dim, height=preview_dim)
        ortho_views_frame.view1.grid(row=0, column=0, sticky="sew", padx=5, pady=5)
        ortho_views_frame.view2 = ctk.CTkCanvas(ortho_views_frame, bg="black", highlightthickness=0, width=preview_dim, height=preview_dim)
        ortho_views_frame.view2.grid(row=1, column=0, sticky="sew", padx=5, pady=5)
        ortho_views_frame.grid_columnconfigure(0, weight=1)
        self.sub_canvas_frames["ortho"] = ortho_views_frame
        
        # TODO bind mouse click on minimap to pan?
        self.show_preview_frame("image") # default behaviour

        #%% Statusbar
        # Status
        self.status_icon = ctk.CTkLabel(self.statusbar, text=STATUS_SYMBOL, text_color=STATUS_COLOR["idle"], width=14)
        self.status_icon.grid(row=0, column=0, sticky="w", padx=(10, 0), pady=(0, 2))
        self.status_label = ctk.CTkLabel(self.statusbar, text="Initializing...")
        self.status_label.grid(row=0, column=1, sticky="w", padx=(4, 0))
        self.status_sam_label = ctk.CTkLabel(self.statusbar, text="") # for SAM asynchronous loading
        self.status_sam_label.grid(row=0, column=2, sticky="w", padx=(4, 0))
        
        # Mask appearance controls
        self.mask_appearance_frame = ctk.CTkFrame(self.statusbar, fg_color="transparent")
        self.mask_appearance_frame.grid(row=0, column=4, sticky="ew", padx=0)
        
        ctk.CTkLabel(self.mask_appearance_frame, text="Mask opacity", anchor="e").grid(row=0, column=0, sticky="ew", padx=10)
        self.mask_opacity_slider = ctk.CTkSlider(self.mask_appearance_frame, from_=0, to=255, command=lambda v: self.update_mask_opacity(int(v)))
        self.mask_opacity_slider.set(self.mask_opacity)
        self.mask_opacity_slider.grid(row=0, column=1, sticky="ew", padx=(0, 10))
        ctk.CTkLabel(self.mask_appearance_frame, text="Fill", anchor="e").grid(row=0, column=2, sticky="ew", padx=(0, 10))
        self.mask_outline_switch = ctk.CTkSwitch(self.mask_appearance_frame, text="Outline", variable=self.mask_outline,
                                                 command=lambda: self.update_display(update_image=False))
        self.mask_outline_switch.grid(row=0, column=3, sticky="ew", padx=(0, 10))
        self.mask_outline_switch.configure(state="disabled") # TODO remove when implemented
        
        # Position label
        self.pos_label = ctk.CTkLabel(self.statusbar, textvariable=self.pos_label_var, anchor="e", width=200)
        self.pos_label.grid(row=0, column=5, sticky="e", padx=10)
        
        # Zoom label
        self.zoom_label = ctk.CTkLabel(self.statusbar, textvariable=self.zoom_label_var)
        self.zoom_label.grid(row=0, column=6, sticky="e", padx=10)

        splash.step(30)
                
        #%% Bindings
        # Zoom via keyboard (Ctrl + / Ctrl -)
        self.bind("<Control-plus>", lambda e: self.zoom_in())
        self.bind("<Control-0>", self.reset_zoom)
        self.bind("<Control-KP_0>", self.reset_zoom) # also keypad for madmen like Oscar :)
        self.bind("<Control-space>", self.reset_zoom)
        self.bind("<Control-minus>", lambda e: self.zoom_out())
        
        # Move view:
        self.bind("<Up>", lambda e: self.pan_view(0, -20))
        self.bind("<Down>", lambda e: self.pan_view(0, 20)) 
        self.bind("<Left>", lambda e: self.pan_view(-20, 0))
        self.bind("<Right>", lambda e: self.pan_view(20, 0))
        
        # Bind "close window" to quit_program
        self.protocol("WM_DELETE_WINDOW", self.quit_program)
        
        # Bind "resizing window"
        self.bind("<Configure>", self.on_resize)
        
        if self.slimtag_config["modules"]["sam"]:
        # Fire SAM at Ctrl release
            self.bind("<KeyRelease-Control_L>", lambda e: self.sam_apply_release())
            self.bind("<KeyRelease-Control_R>", lambda e: self.sam_apply_release())
        
        # Shortcuts
        self.bind("<b>", lambda e: self.toggle_tool("brush"))
        self.bind("<B>", lambda e: self.toggle_tool("brush"))
        self.bind("<e>", lambda e: self.toggle_tool("eraser"))
        self.bind("<E>", lambda e: self.toggle_tool("eraser"))
        self.bind("<m>", lambda e: self.toggle_tool("wand"))
        self.bind("<M>", lambda e: self.toggle_tool("wand"))
        self.bind("<c>", lambda e: self.toggle_tool("cut"))
        self.bind("<C>", lambda e: self.toggle_tool("cut"))
        self.bind("<s>", lambda e: self.toggle_tool("smooth"))
        self.bind("<S>", lambda e: self.toggle_tool("smooth"))
        self.bind("<n>", lambda e: self.add_mask())
        self.bind("<N>", lambda e: self.add_mask())
        self.bind("<Control-z>", lambda e: self.undo())
        self.bind("<Control-Z>", lambda e: self.undo())
        self.bind("<Control-I>", lambda e: self.open_image())
        self.bind("<Control-i>", lambda e: self.open_image())
        #self.bind("<Control-F>", lambda e: self.load_folder()) # TODO reactivate load folder
        #self.bind("<Control-f>", lambda e: self.load_folder())
        self.bind("<Control-S>", lambda e: self.save_mask(switch_fast=True))
        self.bind("<Control-s>", lambda e: self.save_mask(switch_fast=True))
        self.bind("<Control-q>", lambda e: self.quit_program())
        self.bind("<Control-Q>", lambda e: self.quit_program())
        self.bind("<q>", lambda e: self.quit_program())
        self.bind("<Q>", lambda e: self.quit_program())
        
        self.bind("<Tab>", lambda e: self.tab())
        self.bind("<Shift-Tab>", lambda e: self.shift_tab())
        self.bind("<ISO_Left_Tab>", lambda e: self.shift_tab()) # for linux
        
        self.bind("<KeyPress-Shift_L>", lambda e: self.shift_key_pressed())
        self.bind("<KeyPress-Shift_R>", lambda e: self.shift_key_pressed())
        self.bind("<KeyRelease-Shift_L>", lambda e: self.shift_key_released())
        self.bind("<KeyRelease-Shift_R>", lambda e: self.shift_key_released())
        
        # Next image
        self.bind("<KeyPress-period>", lambda e: self.next_image())
        # TODO when folder navigation will be implemented, uncomment this
        #self.bind("<KeyPress-comma>", lambda e: self.prev_image())
        
        #%% Clean-up at the end of __init__
        
        # Deactivate all buttons -- must be done after defining switch_computed_magic_wand
        self.set_controls_state(False)

        # set appearance mode
        self.toggle_appearance()
        
        # Finally, set status to "Ready" and raise back main window
        self.set_status("ready", "Ready")
        splash.withdraw()
        self.update()
        self.deiconify()


        #%% TODO old code to be repurposed, DO NOT REMOVE UNTIL IMPLEMENTED BACK
        # Images in folder navigation frame
        # TODO move in main_canvas frame
        # self.images_in_folder_frame = ctk.CTkFrame(self.right_panel)
        # self.images_in_folder_frame.grid(row=3, column=0, sticky="nsew", padx=10, pady=5)
        # self.images_in_folder_label = ctk.CTkLabel(self.images_in_folder_frame, textvariable=self.images_num_label_var)
        # self.images_in_folder_label.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=(10, 5))

        # self.prev_image_btn = ctk.CTkButton(self.images_in_folder_frame, text="Previous image [,]", command=self.prev_image)
        # self.prev_image_btn.grid(row=1, column=0, sticky="ew", padx=(10, 5), pady=(5, 10))
        # self.prev_image_btn.configure(state="disabled")
        # self.next_image_btn = ctk.CTkButton(self.images_in_folder_frame, text="Next image [.]", command=self.next_image)
        # self.next_image_btn.grid(row=1, column=1, sticky="ew", padx=(5, 10), pady=(5, 10))
        # self.next_image_btn.configure(state="disabled")
        
        # self.images_in_folder_frame.grid_columnconfigure([0, 1], weight=1)

