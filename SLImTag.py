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
import time
import shutil

# Numerical arrays manipulation
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.special import expit # sigmoid

# Image manipulation and TkIntert (ImageTk)
from PIL import Image, ImageDraw, ImageTk

# TkInter and CustomTkInter GUI
import tkinter as tk
from tkinter import filedialog
import customtkinter as ctk

# Custom utils
from slimtag_utils import SplashScreen
from slimtag_utils import MultiButtonDialog, MaskEditDialog
from slimtag_utils import PreprocessingAdjustments, adjust_image
from slimtag_utils import Tooltip
from slimtag_color_utils import rgb_to_hex, hex_to_rgb

# Torch and SAM (Segment anything model)
import torch
from segment_anything import sam_model_registry, SamPredictor

# Asynchronous threading import
import threading

# Suppress specific PyTorch warnings
import warnings
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`"
)

#%% User selected parameters
# TODO move these into configuration file
MAX_DISPLAY = 800   # Maximum display size for resizing images
UNDO_DEPTH = 10     # Maximum number of undo steps

MAX_ZOOM_PIXEL = 32 # minimum number of pixels of orig image visible at max zoom level
MIN_ZOOM_PIXEL = 6144 # maximum number of pixels of orig image visible at max zoom level

REFRESH_RATE_BRUSH = 0.05    # Refresh rate for the brush

PREVIEW_DIM = 250 # max dimension of preview canvas

# predefined high contrast colors for masks
MAX_MASKS = 20 
HIGH_CONTRAST_COLORS = [
    (158, 31, 99), (24, 105, 204), (150, 99, 177), (21, 194, 210),
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (255, 128, 0), (128, 0, 255), (0, 255, 128),
    (128, 255, 0), (255, 0, 128), (0, 128, 255), (128, 128, 0),
    (128, 0, 0), (0, 128, 0), (0, 0, 128)
]
# OLD COLORS
#, (200, 200, 200), (255, 200, 200), (200, 255, 200), (200, 200, 255)

# TODO remove single button colors
# colors for different tool states
TOOL_OFF_COLOR = "#3A3A3A"   # neutral grey when tool is off
BRUSH_ON_COLOR = "#4CAF50"   # green
MAGIC_ON_COLOR = "#FF9800"   # orange
CC_ON_COLOR = "#9C27B0"      # purple
SMOOTH_ON_COLOR = "#2196F3"  # blue

# colors for tools panel (light, dark)
# obtained from color above by overlaying the standard background with the color above at 15% alpha
TOOL_PANEL_COLOR = {
    "brush":  ("#C5D4C6", "#303F31"),
    "wand":   ("#E0D0BA", "#4B3B25"),
    "ccomp":  ("#D1BFD4", "#3C2A3F"),
    "smooth": ("#BED0DE", "#293B49")
    }
# as above, but with color at 5% alpha
TOOL_PANEL_SUBCOLOR = {
    "brush":  ("#D3D8D4", "#2C312D"),
    "wand":   ("#DDD7D0", "#363029"),
    "ccomp":  ("#D7D1D9", "#302A32"),
    "smooth": ("#D2D8DD", "#2A3035")
    }
# color at 10% alpha, if needed
# TOOL_PANEL_SUBCOLOR = {
#     "brush":  ("#CDD7CE", "#2E382F"),
#     "wand":   ("#DFD4C5", "#413627"),
#     "ccomp":  ("#D4C8D7", "#362A39"),
#     "smooth": ("#C8D4DD", "#2A363F")
#     }

STATUS_SYMBOL = "●"
STATUS_COLOR = {
    "ready":  ("#2ECC71", "#2ECC71"),  # green
    "loading":("#F1C40F", "#F1C40F"),  # yellow
    "error":  ("#E74C3C", "#E74C3C"),  # red
    "idle":   ("#95A5A6", "#95A5A6"),  # gray
    }


#%% SAM parameters
# TODO rework magic wand
# Choose the model type
MODEL_TYPE = "vit_b"    # Lightweight

if MODEL_TYPE == "vit_b":       # Lightweight
    MODEL_WEIGHTS_PATH = "models/sam_vit_b_01ec64.pth"
elif MODEL_TYPE == "vit_l":     # Standard
    MODEL_WEIGHTS_PATH = "models/sam_vit_l_0b3195.pth"
elif MODEL_TYPE == "vit_h":     # Advanced
    MODEL_WEIGHTS_PATH = "models/sam_vit_h_4b8939.pth"
else:
    raise Exception("Warning: select a correct model type.")



#%% CTK parameters
#ctk.set_appearance_mode("System")   # System theme
#ctk.set_appearance_mode("dark") # force dark mode for testing
ctk.set_default_color_theme("color_palette.json") # CTK color theme
ctk.set_appearance_mode("dark")

HIGHLIGHT_COLOR = ctk.ThemeManager.theme["CTkButton"]["border_color"]

#%% SLImTAG main class
class SegmentationApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.appearance_mode = tk.StringVar(self, value="dark")
        
        # hide main window and open splash screen
        self.withdraw()
        splash = SplashScreen()

        self.title("SLImTAG")
        self.geometry("1300x900")
        self.iconphoto(False, ImageTk.PhotoImage(file=os.path.join("images", "main_icon.png")))
        
        # TODO move set_appearance_mode to preferences window
        # optionsmenu "dark", "light" with default value: "dark"
        
        #%% Attributes
        
        # Full image and mask
        self.image_orig = None
        self.mask_orig = None
        # Displayed image and mask
        self.image_disp = None
        self.mask_disp = None
        # current image preview (in sub canvas)
        self.current_preview_canvas = None
        self.preview_scale = 1.0
        # Matrix of locked masks
        self.mask_locked = None
        
        # aux display variables
        self.mask_pil = None
        self.tk_ov = None
        self.sam_preview_pil = None
        self.tk_sam_preview = None
        
        # to keep track of resizing window
        self.resizing_event = None
        
        # boolean switch to check if mask is modified and not saved
        # TODO: for multiple images import
        self.modified = False
        
        # zoom & pan status
        self.zoom = 1.0
        self.zoom_max = 1.0
        self.zoom_min = 1.0
        self._pan_start = None
        
        # labels for zoom and mouse position
        self.pos_label_var = tk.StringVar(self, value="| x: 0 | y: 0 | z: 0 |")
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
        self.sam_points = []
        self.sam_pt_labels = []
        self.sam_preview = None # boolean matrix for multipoint SAM preview
        
        # SAM preprocessing (for sliders: range -100 .. 100)
        self.wand_brightness = 0
        self.wand_contrast = 0
        self.wand_gamma = 0
        
        # Magic wand threshold (e.g. SAM model threshold), range 0.0 .. 1.0
        self.wand_threshold = 0.5 # SAM has 0.5 as default value

        # boolean to track if mouse buttons are pressed
        self.b3_pressed = False
        self.mid_pressed = False
        
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
        
        splash.step(5)
        
        
        #%% Top Menu
        self.menu_bar = tk.Menu(self)
        self.set_menu_theme(self.menu_bar, self.appearance_mode.get())

        self.config(menu=self.menu_bar)
        self.topmenu_items = {}

        # Menu File (top menu)
        file_menu = tk.Menu(self.menu_bar, tearoff=0)
        file_menu.add_command(label="Quit", command=self.quit_program, accelerator="Ctrl+Q")
        self.topmenu_items["file"] = file_menu
        self.menu_bar.add_cascade(label="File", menu=file_menu)

        # Menu Edit (top menu)
        edit_menu = tk.Menu(self.menu_bar, tearoff=0)
        edit_menu.add_command(label="Undo", command=self.undo, accelerator="Ctrl+Z")
        # TODO implement preferences window
        edit_menu.add_command(label="Preferences...", command=None, state="disabled")
        self.topmenu_items["edit"] = edit_menu
        self.menu_bar.add_cascade(label="Edit", menu=edit_menu)

        # Menu View (top menu)
        view_menu = tk.Menu(self.menu_bar, tearoff=0)
        view_menu.add_command(label="Zoom in", command=self.zoom_in, accelerator="Ctrl++")
        view_menu.add_command(label="Zoom out", command=self.zoom_out, accelerator="Ctrl+-")
        view_menu.add_command(label="Reset zoom", command=self.reset_zoom, accelerator="Ctrl+0")
        self.topmenu_items["view"] = view_menu
        self.menu_bar.add_cascade(label="View", menu=view_menu)

        # Menu Image (top menu)
        image_menu = tk.Menu(self.menu_bar, tearoff=0)
        image_menu.add_command(label="Import image", command=self.load_image, accelerator="Ctrl+I")
        image_menu.add_command(label="Import folder", command=self.load_folder, accelerator="Ctrl+F", state="disabled")
        # TODO reactivate import folder
        self.topmenu_items["image"] = image_menu
        self.menu_bar.add_cascade(label="Image", menu=image_menu)

        # Menu Mask (top menu)
        mask_menu = tk.Menu(self.menu_bar, tearoff=0)
        mask_menu.add_command(label="Load mask", command=self.load_mask)
        mask_menu.add_command(label="Save mask", command=lambda s=True: self.save_mask(switch_fast=s), accelerator="Ctrl+S")
        mask_menu.add_command(label="Save mask as...", command=lambda s=False: self.save_mask(switch_fast=s))
        mask_menu.add_separator()
        mask_menu.add_command(label="Clear active mask", command=self.clear_active_mask)
        mask_menu.add_command(label="Clear all masks", command=self.clear_all_masks)
        self.topmenu_items["mask"] = mask_menu
        self.menu_bar.add_cascade(label="Mask", menu=mask_menu)

        # Menu Magic Wand (top menu)
        # TODO implement load/save configuration
        wand_menu = tk.Menu(self.menu_bar, tearoff=0)
        wand_menu.add_command(label="Load configuration", command=None, state="disabled")
        wand_menu.add_command(label="Save configuration", command=None, state="disabled")
        self.topmenu_items["wand"] = wand_menu
        self.menu_bar.add_cascade(label="Magic wand", menu=wand_menu)
        
        # Menu Help (top menu)
        # TODO implement help functions
        help_menu = tk.Menu(self.menu_bar, tearoff=0)
        help_menu.add_command(label="Documentation", command=None, state="disabled")
        help_menu.add_command(label="About", command=None, state="disabled")
        self.topmenu_items["help"] = help_menu
        self.menu_bar.add_cascade(label="Help", menu=help_menu)
        
        #%% Main UI elements
        panels_width = 250
        # Left panel for tools
        self.left_panel = ctk.CTkFrame(self, width=panels_width)
        self.left_panel.grid(row=0, column=0, sticky="nsew")
        self.left_panel.grid_rowconfigure(5, weight=1)
        
        # Main canvas
        # TODO different frames with different widgets depending on load type
        # e.g. previous/next image for folder, slider with z-axis for medical...
        self.main_canvas_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_canvas_frame.grid(row=0, column=1, sticky="nsew")
        self.main_canvas_frame.grid_rowconfigure(0, weight=1)
        self.main_canvas_frame.grid_columnconfigure(0, weight=1)
        self.canvas = ctk.CTkCanvas(self.main_canvas_frame, bg="black", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        
        # Right panel for masks
        self.right_panel = ctk.CTkFrame(self, width=panels_width)
        self.right_panel.grid(row=0, column=2, sticky="nsew")
        self.right_panel.grid_rowconfigure(1, weight=1)
        self.right_panel.grid_rowconfigure(2, weight=2)
        
        # Statusbar
        self.statusbar = ctk.CTkFrame(self, height=32, fg_color=("gray92", "gray14"))
        self.statusbar.grid(row=1, column=0, columnspan=3, sticky="nsew", padx=0, pady=0)
        self.statusbar.grid_columnconfigure(3, weight=1)
        
        # Grid configuration for main window
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        
        #%% Left panel: Tools
        # Frames for buttons
        self.tools_btn_frame = {i: ctk.CTkFrame(self.left_panel, corner_radius=0) for i in range(5)}
        frame_paddings = [(0, 5)] + 3*[5] + [(5, 0)]
        for i in range(5):
            self.tools_btn_frame[i].grid(row=i, column=0, sticky="nsew", padx=0, pady=frame_paddings[i])
        
        # Buttons
        # TODO all commands, in particular add the "right-click" that are a different tool now
        # I ould keep the right-click behaviour in any case (for "pro users")
        self.create_tool_button("brush", 0, 0, 0, help_text="Brush [B]")
        self.create_tool_button("eraser", 0, 0, 1, help_text="Eraser")
        self.create_tool_button("polygon", 0, 1, 0)
        self.create_tool_button("bbox", 0, 1, 1)
        self.create_tool_button("cut", 0, 2, 0, help_text="Cut component [C]")
        self.create_tool_button("clean", 0, 2, 1, help_text="Keep component")
        self.create_tool_button("bucket", 0, 3, 0, last_row=True)
        self.create_tool_button("undo", 0, 3, 1, command=self.undo, last_row=True, help_text="Undo [Ctrl-Z]")
        self.create_tool_button("smooth", 1, 0, 0, help_text="Smooth [S]")
        self.create_tool_button("fill", 1, 0, 1)
        self.create_tool_button("denoise", 1, 1, 0, last_row=True)
        self.create_tool_button("interpolate", 1, 1, 1, last_row=True)
        self.create_tool_button("wand", 2, 0, 0, help_text="Magic wand [M]")
        self.create_tool_button("wand_all", 2, 0, 1)
        self.create_tool_button("wand_multi", 2, 1, 0, None, last_row=True)
        self.create_tool_button("wand_box", 2, 1, 1, None, last_row=True)
        self.create_tool_button("ruler", 3, 0, 0, None, last_row=True)
        self.create_tool_button("area", 3, 0, 1, None,last_row=True)
        self.create_tool_button("custom_1", 4, 0, 0, None)
        self.create_tool_button("custom_2", 4, 0, 1, None)
        self.create_tool_button("custom_3", 4, 1, 0, None, last_row=True)
        self.create_tool_button("custom_4", 4, 1, 1, None, last_row=True)
        
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
        self.brush_shape_btn = ctk.CTkSegmentedButton(self.tool_opt_frame["brush"], values=["Circle", "Square", "Line"], command=None)
        self.brush_shape_btn.set("Circle") # TODO implement shape
        self.brush_shape_btn.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=0)
        self.brush_shape_btn.configure(state="disabled") # TODO remove when implemented
        
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
                                              command=lambda v: (setattr(self,"brush_rot",int(v)), self.brush_rot_lbl.configure(text=f"{self.brush_rot}°")))
        self.brush_rot_slider.set(self.brush_rot) # TODO implement rotation
        self.brush_rot_slider.grid(row=6, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))
        self.brush_rot_slider.configure(state="disabled") # TODO remove when implemented
        
        # Magic wand options
        ctk.CTkLabel(self.tool_opt_frame["wand"], text="Magic wand settings:", fg_color="transparent", font=ctk.CTkFont(size=17, weight='bold'), anchor="w").grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=(10, 0))
        self.wand_model_menu = ctk.CTkOptionMenu(self.tool_opt_frame["wand"], values=["SAM (ViT-B)"], command=None) # TODO
        self.wand_model_menu.set("SAM (ViT-B)")
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

        self.wand_auto_update = ctk.CTkButton(self.wand_adj_frame, text="Auto", image=self.icons_dict["AutoUpdate"]["disabled"], command=None)
        self.wand_auto_update.grid(row=4, column=1, sticky="ew", padx=(5, 10), pady=(3,10))
        self.wand_auto_update.configure(state='disabled') # TODO implement auto update
        Tooltip(self.wand_auto_update, text="Auto compute preprocessing parameters")
        self.wand_manual_update = ctk.CTkButton(self.wand_adj_frame, text="Manual", image=self.icons_dict["ManualUpdate"]["disabled"], command=self.manual_wand_preprocessing)
        self.wand_manual_update.grid(row=4, column=0, sticky="ew", padx=(10, 5), pady=(3,10))
        Tooltip(self.wand_manual_update, text="Select preprocessing parameters")
        
        self.wand_adj_frame.grid_columnconfigure([0, 1], weight=1)
        
        ctk.CTkLabel(self.tool_opt_frame["wand"], text="Wand threshold", fg_color="transparent", anchor="w").grid(row=3, column=0, sticky="ew", padx=(10, 5), pady=(10, 2))
        self.wand_threshold_lbl = ctk.CTkLabel(self.tool_opt_frame["wand"], text=f"{self.wand_threshold:.2f}", fg_color="transparent", anchor="e")
        self.wand_threshold_lbl.grid(row=3, column=1, sticky="ew", padx=(5, 10), pady=(10, 2))
        self.wand_threshold_slider = ctk.CTkSlider(self.tool_opt_frame["wand"], from_=0.0, to=1.0,
                                                   command=lambda v: (setattr(self,"wand_threshold",float(v)), self.wand_threshold_lbl.configure(text=f"{self.wand_threshold:.2f}")))
        self.wand_threshold_slider.set(self.wand_threshold) # TODO
        self.wand_threshold_slider.grid(row=4, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))
        
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
        
        image_only_frame = ctk.CTkFrame(self.navigation_frame)#, fg_color="transparent")
        image_only_frame.canvas = ctk.CTkCanvas(image_only_frame, bg="black", highlightthickness=0, width=PREVIEW_DIM, height=PREVIEW_DIM)
        image_only_frame.canvas.grid(row=0, column=0, sticky="sew", padx=5, pady=5)
        image_only_frame.grid_rowconfigure(0, weight=1)
        image_only_frame.grid_columnconfigure(0, weight=1)
        self.sub_canvas_frames["image"] = image_only_frame
        
        ortho_views_frame = ctk.CTkFrame(self.navigation_frame)#, fg_color="transparent")
        ortho_views_frame.view1 = ctk.CTkCanvas(ortho_views_frame, bg="black", highlightthickness=0, width=PREVIEW_DIM, height=PREVIEW_DIM)
        ortho_views_frame.view1.grid(row=0, column=0, sticky="sew", padx=5, pady=5)
        ortho_views_frame.view2 = ctk.CTkCanvas(ortho_views_frame, bg="black", highlightthickness=0, width=PREVIEW_DIM, height=PREVIEW_DIM)
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
                                                 command=lambda: self.update_display(update_all="Mask"))
        self.mask_outline_switch.grid(row=0, column=3, sticky="ew", padx=(0, 10))
        self.mask_outline_switch.configure(state="disabled") # TODO remove when implemented
        
        # Position label
        self.pos_label = ctk.CTkLabel(self.statusbar, textvariable=self.pos_label_var, anchor="e", width=200)
        self.pos_label.grid(row=0, column=5, sticky="e", padx=10)
        
        # Zoom label
        self.zoom_label = ctk.CTkLabel(self.statusbar, textvariable=self.zoom_label_var)
        self.zoom_label.grid(row=0, column=6, sticky="e", padx=10)

        splash.step(10)
        
        
        #%% SAM
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[MODEL_TYPE](checkpoint=MODEL_WEIGHTS_PATH)
        sam.to(device).eval()
        self.sam = SamPredictor(sam)

        splash.step(20) 
        
        
        # Define the asynchronous mechanism to speed up image loading
        self.switch_computed_magic_wand = False     # True if SAM is loaded
        self.thread = None                          # Threading variable
        self.lock = threading.Lock()              # To protect shared varaibles
        
        self.set_controls_state(False) # Deactivate all buttons -- must be done after defining switch_computed_magic_wand

        #%% Bindings
        self.canvas.bind("<MouseWheel>", self.zoom_evt)
        self.canvas.bind("<Button-4>", self.zoom_in) # <Button-4> is scroll up for Linux
        self.canvas.bind("<Button-5>", self.zoom_out) # <Button-5> is scroll down for Linux
        self.canvas.bind("<Motion>", self.draw_brush_preview, add="+")
        self.canvas.bind("<Motion>", self.on_canvas_track, add="+")
        self.canvas.bind("<Button-1>", self.on_canvas_left)
        self.canvas.bind("<Button-2>", self.on_canvas_mid)
        self.canvas.bind("<Button-3>", self.on_canvas_right)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<B2-Motion>", self.on_canvas_drag)
        self.canvas.bind("<B3-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_left_release)
        self.canvas.bind("<ButtonRelease-2>", self.on_canvas_mid_release)
        self.canvas.bind("<ButtonRelease-3>", self.on_canvas_right_release)
        
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
        
        # Fire SAM at Ctrl release
        self.bind("<KeyRelease-Control_L>", lambda e: self.sam_apply_release())
        self.bind("<KeyRelease-Control_R>", lambda e: self.sam_apply_release())
        
        # Shortcuts
        self.bind("<b>", lambda e: self.toggle_tool("brush"))
        self.bind("<B>", lambda e: self.toggle_tool("brush"))
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
        self.bind("<Control-I>", lambda e: self.load_image())
        self.bind("<Control-i>", lambda e: self.load_image())
        #self.bind("<Control-F>", lambda e: self.load_folder()) # TODO reactivate load folder
        #self.bind("<Control-f>", lambda e: self.load_folder())
        self.bind("<Control-S>", lambda e: self.save_mask(switch_fast=True))
        self.bind("<Control-s>", lambda e: self.save_mask(switch_fast=True))
        self.bind("<Control-q>", lambda e: self.quit_program())
        self.bind("<Control-Q>", lambda e: self.quit_program())
        self.bind("<q>", lambda e: self.quit_program())
        self.bind("<Q>", lambda e: self.quit_program())
        
        self.bind("<Tab>", lambda e: self.tab())
        self.bind("<Shift-Tab>", lambda e: self.shiftTab())
        self.bind("<ISO_Left_Tab>", lambda e: self.shiftTab()) # for linux
        
        self.bind("<KeyPress-Shift_L>", lambda e: self.shiftPressed())
        self.bind("<KeyPress-Shift_R>", lambda e: self.shiftPressed())
        self.bind("<KeyRelease-Shift_L>", lambda e: self.shiftReleased())
        self.bind("<KeyRelease-Shift_R>", lambda e: self.shiftReleased())
        
        # Next image
        self.bind("<KeyPress-period>", lambda e: self.next_image())
        # TODO when folder navigation will be implemented, uncomment this
        #self.bind("<KeyPress-comma>", lambda e: self.prev_image())
        
        #%% Clean-up at the end of __init__
        
        # set appearance mode
        self.toggle_appearance()
        
        # Finally, set status to "Ready" and raise back main window
        self.set_status("ready", "Ready")
        splash.withdraw()
        self.update_idletasks()
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


    #%% AUX methods
    # Async method for efficient SAM loading
    def async_loader(self):
        #print("Loading SAM model")
        self.status_sam_label.configure(text="(Loading image into SAM...)")
        #  Thread-safe upload of shared variable
        with self.lock:
            self.switch_computed_magic_wand = False
            
            if len(self.mask_labels) == 0 or self.active_mask_id is None: # disable all buttons if there are no masks
                self.set_controls_state(False)
            else:
                self.set_controls_state(True)
            
            # SAM model inference on image
            image = adjust_image(np.array(self.image_orig), self.wand_brightness, self.wand_contrast, self.wand_gamma)
            self.sam.set_image(image)
            
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
        self.update_display()
    
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
                self.load_image(add_mask=False)
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
        ctk.set_appearance_mode(self.appearance_mode.get())
        self.set_menu_theme(self.menu_bar, self.appearance_mode.get())
        for menu in self.topmenu_items:
            self.set_menu_theme(self.topmenu_items[menu], self.appearance_mode.get())
        if hasattr(self, 'active_context_menu'):
            self.set_menu_theme(self.active_context_menu, self.appearance_mode.get())

    def set_menu_theme(self, menu, mode):
        if mode.lower() == 'dark':
            menu.configure(bd=0, background="#242424", fg="#999999", activebackground="#242424", activeforeground="white", activeborderwidth=0)
        else:
            menu.configure(bd=0, background="#d9d9d9", fg="#000000", activebackground="#d9d9d9", activeforeground="#242424", activeborderwidth=0)

    #%% NAVIGATION PANEL
    def show_preview_frame(self, preview):
        for nav in self.sub_canvas_frames:
            self.sub_canvas_frames[nav].grid_forget()
        # TODO implement for ortho
        # if hasattr(self.sub_canvas_frames[preview], "view1") ...
        if self.image_orig is not None:
            scale = max(self.orig_w, self.orig_h) / PREVIEW_DIM
            self.preview_scale = scale
            self.sub_canvas_frames["image"].canvas.configure(width=int(self.orig_w / scale), height=int(self.orig_h / scale))
            self.sub_canvas_image = ImageTk.PhotoImage(self.image_orig.resize((int(self.orig_w / scale), int(self.orig_h / scale)), Image.Resampling.LANCZOS))
            self.sub_canvas_frames["image"].canvas.create_image(0, 0, anchor="nw", image=self.sub_canvas_image, tag="image")
        self.current_preview_canvas = self.sub_canvas_frames["image"].canvas
        self.sub_canvas_frames[preview].grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.sub_canvas_frames[preview].tkraise()
    
    def update_preview_frame(self):
        self.current_preview_canvas.delete("rectangle")
        x = int(self.view_x / self.preview_scale)
        y = int(self.view_y / self.preview_scale)
        w = int(self.view_w / self.preview_scale)
        h = int(self.view_h / self.preview_scale)
        self.current_preview_canvas.create_rectangle(x, y, x+w, y+h, outline=HIGHLIGHT_COLOR, width=2, tag="rectangle")
    
    #%% UPDATE DISPLAY
    def update_display(self, update_all="Global"):
        '''
        Aux method to update display whenever a change occurs.
        
        Valid argument for update_all:
            - "Global" updates both the background image and the mask overlay
            - "Mask" updates only the mask overlay
        '''
        if self.image_orig is None:
            return
        
        self.zoom_label_var.set(f"Zoom: {round(100*self.zoom)}%")

        if update_all == "Global":
            # remove old info
            self.canvas.delete("background_image","mask")
            # create new image view and paste it on canvas
            self.image_disp = self.image_orig.crop([self.view_x, self.view_y, self.view_x+self.view_w, self.view_y+self.view_h]) \
                                             .resize((self.canvas.winfo_width(), self.canvas.winfo_height()), Image.NEAREST)
            self.tk_img = ImageTk.PhotoImage(self.image_disp)
            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img, tag="background_image")
        elif update_all == "Mask":
            # delete only the mask to speed up computations
            self.canvas.delete("mask")
        
        # compute new mask view margins
        top = max(0, self.view_y)
        bottom = min(max(self.view_y+self.view_h, 0), self.orig_h)
        left = max(0,self.view_x)
        right = min(max(self.view_x+self.view_w,0), self.orig_w)
        
        # create new mask view and populate
        cut_mask_orig = np.zeros((self.view_h, self.view_w), dtype=self.mask_orig.dtype)
        try:
            cut_mask_orig[top-self.view_y:bottom-self.view_y, left-self.view_x:right-self.view_x] = self.mask_orig[top:bottom, left:right]
        except ValueError: # in case we are out of image limits, in this case keep empty mask
            pass
        
        # TODO: if self.mask_outline.get(): change overlay as border only
        # else: do as below
        # create overlay object and convert it to be pasted on canvas
        overlay = np.zeros((self.view_h, self.view_w, 4), np.uint8)
        for mid, c in self.mask_colors.items():
            if not self.mask_widgets[mid].hidden:
                overlay[cut_mask_orig==mid] = [*c, self.mask_opacity if len(self.sam_points) == 0 else (self.mask_opacity // 3)]
        self.mask_pil = Image.fromarray(overlay)
        resized = self.mask_pil.resize((self.canvas.winfo_width(), self.canvas.winfo_height()), Image.NEAREST)
        self.tk_ov = ImageTk.PhotoImage(resized)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_ov, tag="mask")
        
        # if SAM is active, create also multipoint preview
        if any(self.tool_active[tool] for tool in ["wand", "wand_multi"]):
            # create new mask view and populate
            cut_mask_preview = np.full((self.view_h, self.view_w), False)
            try:
                cut_mask_preview[top-self.view_y:bottom-self.view_y, left-self.view_x:right-self.view_x] = self.sam_preview[top:bottom, left:right]
            except ValueError: # in case we are out of image limits, in this case keep empty mask
                pass
            
            # TODO: if self.mask_outline.get(): change overlay as border only
            # but maybe not for SAM preview?
            # create overlay object and convert it to be pasted on canvas
            overlay_prev = np.zeros((self.view_h, self.view_w, 4), np.uint8)
            if not self.mask_widgets[self.active_mask_id].hidden:
                overlay_prev[cut_mask_preview] = [*self.mask_colors[self.active_mask_id], (2 * self.mask_opacity) // 3]
            self.sam_preview_pil = Image.fromarray(overlay_prev)
            resized_prev = self.sam_preview_pil.resize((self.canvas.winfo_width(), self.canvas.winfo_height()), Image.NEAREST)
            self.tk_sam_preview = ImageTk.PhotoImage(resized_prev)
            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_sam_preview, tag="mask")
            
        
        # raise back SAM multipoints if any
        self.canvas.tag_raise("sam_pt")
    
    def update_mask_opacity(self, v):
        self.mask_opacity = v
        self.update_display(update_all="Mask")
    
    def update_display_after_resize(self):
        self.view_h = int(self.canvas.winfo_height()/self.zoom)
        self.view_w = int(self.canvas.winfo_width()/self.zoom)
        self.update_display(update_all="Global")

    #%% UI TOOLS METHODS
    def show_tool_frame(self, tool):
        frame = self.tool_opt_frame[tool]
        self.current_tool_frame = frame
        frame.tkraise()
        
    def deactivate_tools(self):
        '''
        Keep one tool button active at time.
        '''
        # TODO rework
        for tool in self.tools:
            self.tool_active[tool] = False
            self.tool_btn[tool].configure(border_width=0)
        self.show_tool_frame("empty")

    def set_controls_state(self, enabled: bool):
        '''
        Enable/disable all buttons.
        '''
        state = "normal" if enabled else "disabled"
        for tool in self.tool_btn:
            self.tool_btn[tool].configure(state=state, image=self.tool_icon[tool][state])
        if not self.switch_computed_magic_wand:
            for tool in ["wand", "wand_all", "wand_multi", "wand_box"]:
                self.tool_btn[tool].configure(state="disabled", image=self.tool_icon[tool]["disabled"])
        # TODO remove tool from 'always disabled' list when the corresponding function has been implemented
        always_disabled = ["polygon", "bbox", "bucket",
                           "fill", "denoise", "interpolate",
                           "wand_all", "wand_multi", "wand_box",
                           "ruler", "area",
                           "custom_1", "custom_2", "custom_3", "custom_4"]
        for tool in always_disabled:
            self.tool_btn[tool].configure(state="disabled", image=self.tool_icon[tool]["disabled"])
    
    def set_hide_lock_all_btns(self, enabled: bool, propagate=True):
        '''
        Hard set state for "hide all masks" and "lock all masks" buttons.
        
        Put them in the "non-hidden" and "non-locked" state, and enable/disable
        the buttons depending on state.
        
        If propagate, change also the state of all masks to "non-hidden" and "non-locked".
        '''
        state = "normal" if enabled else "disabled"
        if propagate:
            self.toggle_all_masks_hide(False)
            self.toggle_all_masks_lock(False)
        else:
            self.hide_all_mask_btn.hidden = False
            self.lock_all_mask_btn.locked = False
        self.hide_all_mask_btn.configure(state=state, image=self.icons_dict["EyeOpen"][state])
        self.lock_all_mask_btn.configure(state=state, image=self.icons_dict["LockOpen"][state])

    #%% TOOL BUTTONS
    def create_tool_button(self, tool, btn_frame, row, col, command=None, last_row=False, help_text=''):
        """
        Aux function to create button object from tool
        """
        assert tool in self.tools
        self.tool_btn[tool] = ctk.CTkButton(self.tools_btn_frame[btn_frame],
                                            width=44, height=44,
                                            text="", image=self.tool_icon[tool]["normal"],
                                            fg_color="transparent",
                                            command=(lambda: self.toggle_tool(tool)) if command is None else command)
        padx = (4, 2) if col == 0 else (2, 4) # col == 1
        pady = (4 if row == 0 else 2, 4 if last_row else 2)
        self.tool_btn[tool].grid(row=row, column=col, sticky="nsew", padx=padx, pady=pady)
        self.tool_btn[tool].help = Tooltip(self.tool_btn[tool], text=help_text)
    
    def toggle_tool(self, tool):
        if not self.image_is_loaded():
            return
        if tool == "undo": # just as a safeguard, "toggle" should not be defined for undo
            return
        
        assert tool in self.tools
        
        if self.tool_btn[tool].cget('state') != "disabled":
            if not self.tool_active[tool]:
                self.deactivate_tools()
                self.tool_active[tool] = True
                self.tool_btn[tool].configure(border_width=2)
                self.show_tool_frame(self.tool_opt_map[tool])
            else:
                self.tool_active[tool] = False
                self.tool_btn[tool].configure(border_width=0)
                self.show_tool_frame("empty")
        
        if tool in ["brush", "eraser"]:
            self._draw_brush_preview(self.mouse['x'], self.mouse['y'])
    
    #%% UNDO
    def push_undo(self):
        '''
        Saves a copy of the current mask (mask_orig) into the undo_stack.
        '''
        if self.mask_orig is not None:
            self.undo_stack.append(self.mask_orig.copy())
            if len(self.undo_stack) > UNDO_DEPTH:
                self.undo_stack.pop(0)

    def undo(self):
        '''
        Pops the last saved state from undo_stack and restores it as the 
        current mask_orig. Calls update_display() to refresh the canvas so the 
        user sees the previous mask.
        '''
        # Check if an image is loaded
        if not self.image_is_loaded():
            return
        
        if self.undo_stack:
            self.mask_orig = self.undo_stack.pop()
            self.update_lock()
            self.update_display(update_all="Mask")


    #%% MASK MANAGEMENT
    def add_mask(self, name=None):
        '''
        Creates a new mask, asks the user for a name via dialog, assigns it a 
        unique ID and a color, and updates the UI accordingly.
        '''
        # Check if an image is loaded
        if not self.image_is_loaded():
            return
        
        self.deactivate_tools()
        if len(self.mask_labels) >= MAX_MASKS:
            return
        
        default_color = [c for c in HIGH_CONTRAST_COLORS if c not in self.mask_colors.values()][0] # get first non-used color
        color = None
        
        # candidate mask id
        mid = min([i+1 for i in range(MAX_MASKS) if i+1 not in self.mask_labels.keys()])
        
        if name is None:
            # Ask user for mask name
            color_in_use = True
            while color_in_use:
                name, color = MaskEditDialog(self, title="New mask", initial_color=rgb_to_hex(default_color), mask_name=f"mask_{mid}").get()
                if color is not None and hex_to_rgb(color) in self.mask_colors.values():
                    MultiButtonDialog(self, message=f"Color {color} already in use. Please choose another one.", buttons=(("OK", None),))
                else:
                    color_in_use = False

        if not name:  # User cancelled or empty
            return
        color = default_color if not color else hex_to_rgb(color)
        
        self.mask_labels[mid] = name
        self.mask_colors[mid] = color
        
        self.mask_widgets[mid] = self.create_mask_widget(mid)
        self.mask_widgets[mid].pack(fill="x", expand=True, padx=(6, 2), pady=(3, 0))
        self.change_mask(target_id=mid) # this also sets self.active_mask_id
        
        # since new masks are created non-locked and non-hidden,
        # revert state of "lock all" and "hide all" buttons but do not change
        # status of other masks
        self.toggle_all_masks_hide(set_hide=False, enabled=True, propagate=False)
        self.toggle_all_masks_lock(set_lock=False, enabled=True, propagate=False)
        self.set_controls_state(True) # activate buttons if there is at least one mask
    
    def _crc(self, mid, circle_size = 21):
        # aux function that draws a circle for the mask with ID mid
        color_circle = Image.new("RGBA", (circle_size+1, circle_size+1), (0, 0, 0, 0))
        color_circle_draw = ImageDraw.Draw(color_circle)
        color_circle_draw.ellipse((0, 0, circle_size, circle_size), fill=self.mask_colors[mid])
        return ctk.CTkImage(color_circle, size=(circle_size+1, circle_size+1))

    def create_mask_widget(self, mid):
        mask_frame = ctk.CTkFrame(self.mask_list_frame)
        mask_frame._default_fg_color = mask_frame.cget("fg_color")
        mask_frame.crc = ctk.CTkLabel(mask_frame, text="", image=self._crc(mid))
        mask_frame.crc.grid(row=0, column=0, padx=(10, 5), pady=5)
        mask_frame.lbl = ctk.CTkLabel(mask_frame, text=f"{mid}: {self.mask_labels[mid]}", anchor="w")
        mask_frame.lbl.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        mask_frame._default_text_color = mask_frame.lbl.cget("text_color")
        mask_frame.hide = ctk.CTkButton(mask_frame, text="",
                                        image=self.icons_dict["EyeOpen"]["normal"],
                                        width=34, height=34,
                                        fg_color="transparent",
                                        command=lambda: self.toggle_mask_hide(mid, not mask_frame.hidden))
        mask_frame.hide.grid(row=0, column=2, padx=(5,2), pady=5)
        mask_frame.hidden = False
        mask_frame.lock = ctk.CTkButton(mask_frame, text="",
                                        image=self.icons_dict["LockOpen"]["normal"],
                                        width=34, height=34,
                                        fg_color="transparent",
                                        command=lambda: self.toggle_mask_lock(mid, not mask_frame.locked))
        mask_frame.lock.grid(row=0, column=3, padx=2, pady=5)
        mask_frame.locked = False
        clear_btn = ctk.CTkButton(mask_frame, text="×",
                                  font=ctk.CTkFont(size=24, weight="bold"),
                                  width=34, height=34,
                                  fg_color="transparent",
                                  text_color="#AB2B22",
                                  command=lambda: self.clear_mask(mid))
        clear_btn.grid(row=0, column=4, padx=(2,5), pady=5)
        clear_btn.bind("<Enter>", lambda e: clear_btn.configure(fg_color="#AB2B22", text_color="white"))
        clear_btn.bind("<Leave>", lambda e: clear_btn.configure(fg_color="transparent", text_color="#AB2B22"))
        mask_frame.grid_columnconfigure(1, weight=1)
        mask_frame.bind("<Button-1>", lambda e: self.change_mask(mid))
        mask_frame.bind("<Button-3>", lambda e: self.update_mask(e, mid))
        mask_frame.crc.bind("<Button-1>", lambda e: self.change_mask(mid))
        mask_frame.crc.bind("<Button-3>", lambda e: self.update_mask(e, mid))
        mask_frame.lbl.bind("<Button-1>", lambda e: self.change_mask(mid))
        mask_frame.lbl.bind("<Button-3>", lambda e: self.update_mask(e, mid))
        if 1 <= mid <= 9:
            self.bind(f"<Key-{mid}>", lambda e: self.change_mask(mid))
        return mask_frame

    def change_mask(self, target_id=None):
        '''
        Changes the currently active mask based on the user selection in the 
        combo box.
        '''
        # Retrieves the mask ID corresponding to the current selection
        if self.active_mask_id:
            self.mask_widgets[self.active_mask_id].configure(border_width=0, fg_color=self.mask_widgets[self.active_mask_id]._default_fg_color)
            self.mask_widgets[self.active_mask_id].hide.configure(hover_color=ctk.ThemeManager.theme["CTkButton"]["hover_color"])
            self.mask_widgets[self.active_mask_id].lock.configure(hover_color=ctk.ThemeManager.theme["CTkButton"]["hover_color"])
        self.active_mask_id = target_id
        # Change appearance of mask row in mask list
        self.mask_widgets[target_id].configure(border_width=3, border_color=HIGHLIGHT_COLOR, fg_color=ctk.ThemeManager.theme["CTkButton"]["hover_color"])
        self.mask_widgets[self.active_mask_id].hide.configure(hover_color=self.mask_widgets[self.active_mask_id]._default_fg_color)
        self.mask_widgets[self.active_mask_id].lock.configure(hover_color=self.mask_widgets[self.active_mask_id]._default_fg_color)
        self.set_controls_state(True)
        self._draw_brush_preview(self.mouse['x'], self.mouse['y'])

    def update_mask(self, e, target_id):
        if hasattr(self, 'active_context_menu'):
            self.active_context_menu.destroy()

        context_menu = tk.Menu(self, tearoff=0)
        context_menu.add_command(label="Edit mask", command=lambda: self.edit_mask(target_id))
        context_menu.add_command(label="Set as active mask", command=lambda: self.change_mask(target_id))
        context_menu.add_command(label="Delete mask", command=lambda: self.clear_mask(target_id))
        context_menu.add_separator()
        context_menu.add_command(label="Close this menu", command=lambda: context_menu.destroy())
        context_menu.post(e.x_root, e.y_root)

        self.active_context_menu = context_menu
        self.set_menu_theme(self.active_context_menu, self.appearance_mode.get())
    
    def edit_mask(self, target_id): 
        self.deactivate_tools()
        name, color = MaskEditDialog(self,
                                     initial_color=rgb_to_hex(self.mask_colors[target_id]),
                                     mask_name=self.mask_labels[target_id]
                                     ).get()
        
        # Update Widget
        if color is not None:
            self.mask_colors[target_id] = hex_to_rgb(color)
            self.mask_widgets[target_id].crc.configure(image=self._crc(target_id))
        if name != "":
            self.mask_labels[target_id] = name
            self.mask_widgets[target_id].lbl.configure(text=f"{target_id}: {self.mask_labels[target_id]}")
        self.update_display(update_all="Mask")
    
    def clear_mask(self, mid):
        '''
        Deletes all pixels of the mask with mask id mid, removes its label and 
        color, updates the combo box to reflect remaining masks, and refreshes 
        the display and color preview.
        '''
        # Check if an image is loaded
        if not self.image_is_loaded():
            return
        
        self.deactivate_tools()
        
        if self.mask_orig is None:
            return
        
        self.push_undo()
        self.mask_locked[self.mask_orig == mid] = False # free locks
        self.mask_orig[self.mask_orig == mid] = 0
        del self.mask_labels[mid]
        del self.mask_colors[mid]
        self.mask_widgets[mid].destroy()
        del self.mask_widgets[mid]
        
        if 1 <= mid <= 9:
            self.unbind(f"<Key-{mid}>")

        if self.active_mask_id == mid:
            self.active_mask_id = None
        
        if len(self.mask_labels) == 0 or self.active_mask_id is None: # disable all buttons if there are no masks
            self.set_controls_state(False)
        self.update_display(update_all="Mask")
    
    def clear_active_mask(self):
        if self.active_mask_id is None:
            return
        self.clear_mask(self.active_mask_id)
    
    def clear_all_masks(self):
        mask_ids = list(self.mask_labels.keys())
        for mid in mask_ids:
            self.clear_mask(mid)
        # TODO add warning
    
    def toggle_mask_hide(self, mid, set_hide: bool, update_display=True):
        # change mid mask hidden status to set_hide
        self.mask_widgets[mid].hidden = set_hide
        self.mask_widgets[mid].hide.configure(image=self.icons_dict["EyeClosed" if set_hide else "EyeOpen"]["normal"])
        # if all the statuses of the single masks are the same, change the "all" button as well
        all_statuses = set([self.mask_widgets[m].hidden for m in list(self.mask_labels.keys())])
        if len(all_statuses) == 1:
            self.toggle_all_masks_hide(list(all_statuses)[0], propagate=False)
        else:
            self.toggle_all_masks_hide(False, propagate=False)
        if update_display:
            self.update_display(update_all="Mask")
    
    def toggle_mask_lock(self, mid, set_lock: bool):
        # change mid mask locked status to set_lock
        self.mask_widgets[mid].locked = set_lock
        self.mask_widgets[mid].lock.configure(image=self.icons_dict["LockClosed" if set_lock else "LockOpen"]["normal"])
        self.mask_locked[self.mask_orig==mid] = set_lock
        # if all the statuses of the single masks are the same, change the "all" button as well
        all_statuses = set([self.mask_widgets[m].locked for m in list(self.mask_labels.keys())])
        if len(all_statuses) == 1:
            self.toggle_all_masks_lock(list(all_statuses)[0], propagate=False)
        else:
            self.toggle_all_masks_lock(False, propagate=False)
    
    def toggle_all_masks_hide(self, set_hide: bool, enabled=True, propagate=True):
        state = "normal" if enabled else "disabled"
        icon = "EyeClosed" if set_hide else "EyeOpen"
        self.hide_all_mask_btn.hidden = set_hide
        self.hide_all_mask_btn.configure(state=state, image=self.icons_dict[icon][state])
        if propagate:
            mask_ids = list(self.mask_labels.keys())
            for mid in mask_ids:
                self.toggle_mask_hide(mid, set_hide, update_display=False)
            self.update_display(update_all="Mask")

    def toggle_all_masks_lock(self, set_lock: bool, enabled=True, propagate=True):
        state = "normal" if enabled else "disabled"
        icon = "LockClosed" if set_lock else "LockOpen"
        self.lock_all_mask_btn.locked = set_lock
        self.lock_all_mask_btn.configure(state=state, image=self.icons_dict[icon][state])
        if propagate:
            mask_ids = list(self.mask_labels.keys())
            for mid in mask_ids:
                self.toggle_mask_lock(mid, set_lock)
    
    def update_lock(self):
        # update self.mask_locked with current locked masks
        if self.mask_orig is None:
            return
        self.mask_locked = np.full(self.mask_orig.shape, False)
        mask_ids = list(self.mask_labels.keys())
        for mid in mask_ids:
            if self.mask_widgets[mid].locked:
                self.mask_locked[self.mask_orig==mid] = True

    #%% LOAD & SAVE METHODS
    def load_image(self, path=None, add_mask=True):
        '''
        Load a .png or .jpg image and define an empty mask on it.
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
            
            p = filedialog.askopenfilename(filetypes=[("Image files", ("*.png", "*.jpg", "*.jpeg"))])
            if not p:
                return
            
        else:
            p = path
        
        self.path_original_image = p

        self.set_status("loading", "Loading image...")
        img = Image.open(p).convert("RGB")
        self.orig_w, self.orig_h = img.size
        self.image_orig = img
        self.mask_orig = np.zeros((self.orig_h, self.orig_w), np.uint8)
        self.mask_locked = np.full(self.mask_orig.shape, False)
        self.sam_preview = np.full(self.mask_orig.shape, False)
        
        self.update_title()
        # Async load of the SAM model to avoid freezed interface
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self.async_loader, daemon=True)
            self.thread.start()
        
        # reset masks
        self.clear_all_masks()
        
        # reset zoom
        self.zoom = 1.0
        # Define a max and min zoom
        self.zoom_max = max(self.canvas.winfo_width() / MAX_ZOOM_PIXEL, self.canvas.winfo_height() / MAX_ZOOM_PIXEL)
        self.zoom_min = min(self.canvas.winfo_width() / MIN_ZOOM_PIXEL, self.canvas.winfo_height() / MIN_ZOOM_PIXEL)

        # reset history
        self.undo_stack.clear()
        
        # Define the view parameters (equal to the canvas size):
        self.update_idletasks()
        self.view_x = 0
        self.view_y = 0
        self.view_w = self.canvas.winfo_width()
        self.view_h = self.canvas.winfo_height()
        
        if add_mask:
            self.add_mask("mask_1")
        self.update_display()
        
        self.show_preview_frame("image")
        self.update_preview_frame()
        
        if self.list_images is None:
            self.images_num_label_var.set("Image 1 of 1")
            #self.next_image_btn.configure(state="disabled")
            # TODO image navigation

        self.toggle_all_masks_hide(set_hide=False, enabled=True)
        self.toggle_all_masks_lock(set_lock=False, enabled=True)
        
        self.set_status("ready", "Ready")

        
    def load_folder(self):
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

        
        self.load_image(path=self.list_images[self.list_index])
        
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
        
            self.load_image(path=self.list_images[self.list_index])
            
            self.images_num_label_var.set(f"Image {self.list_index+1} of {len(self.list_images)}")
            
            self.next_image_btn.configure(state="disabled") # Disable next img
            self.switch_computed_magic_wand = False # Disable MAGIC WAND
            self.magic_btn.configure(state="disabled")
            
            self.set_status("ready", "Ready")

    def load_mask(self):
        """
        Upload an existing mask
        - Indexed PNG (mode "P"): direct recovery of mask indices.
        - RGB PNG: legacy color-based reconstruction.
        """
        if not self.image_is_loaded():
            return
        
        self.deactivate_tools()
        p = filedialog.askopenfilename(filetypes=[("PNG (indexed or RGB)", "*.png")])
        if not p:
            return
        
        self.set_status("loading", "Loading mask...")
        
        self.push_undo()
        ext = os.path.splitext(p)[1].lower()
        if ext != ".png":
            return
        
        # RESET ALL MASKS
        self.clear_all_masks()
        
        img = Image.open(p)
        
        # CASE 1: Indexed PNG
        if img.mode == "P":
            arr = np.array(img, dtype=np.uint8)
            self.mask_orig = arr.copy()
            labels = np.unique(arr)
            labels = labels[labels != 0][:MAX_MASKS]
            palette = img.getpalette()
            
            for l in labels:
                self.mask_labels[l] = f"mask_{l}"
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
                    if len(unique_colors) >= MAX_MASKS:
                        break
            
            mask = np.zeros((h, w), np.uint8)
            for i, color in enumerate(unique_colors, 1):
                mask[np.all(arr == color, axis=-1)] = i
                self.mask_labels[i] = f"mask_{i}"
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
        
        self.update_display()
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
            p = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG (indexed)", "*.png")])
            if not p:
                return
        else: # Save()
            # If working with a folder
            if self.list_images != None:
                p = os.path.join(self.path_aux_save, os.path.splitext(os.path.basename(self.list_images[self.list_index]))[0]+".png")
            # Otherwise
            else:
                p = os.path.splitext(self.path_original_image)[0] + "_mask.png"
        
        self.set_status("loading", "Saving mask...")
    
        # SCALE
        mask_to_save = Image.fromarray(self.mask_orig, mode="P")
        mask_to_save = mask_to_save.resize((self.orig_w, self.orig_h), Image.NEAREST)
    
        
        palette = [0, 0, 0] * 256  # index 0 = background nero
        for mid, color in self.mask_colors.items():
            palette[mid*3:mid*3+3] = list(color)
    
        mask_to_save.putpalette(palette)
        mask_to_save.save(p)
        
        self.set_modified(False)
        self.set_status("ready", "Ready")

    #%% MOUSE EVENTS
    def on_canvas_left(self, e):
        '''
        Handles left-clicks on the canvas, performing the active tool's action 
        (brush, magic wand, connected component, or smoothing) and using Shift 
        to modify behaviour.
        '''
        shift_pressed = (e.state & 0x0001) != 0 or self.b3_pressed
        ctrl_pressed = (e.state & 0x0004) != 0
        self._prev_brush_pos = None
        
        if not any(self.tool_active[tool] for tool in self.tool_active):
            self._pan_start = (e.x, e.y, self.view_x, self.view_y)
            return
        
        # Check position
        x_check = int((e.x)*(self.view_w/self.canvas.winfo_width())) + self.view_x
        y_check = int((e.y)*(self.view_h/self.canvas.winfo_height())) + self.view_y
        
        if (x_check < 0) or (x_check > self.orig_w) or (y_check < 0) or (y_check > self.orig_h):
            check_inside_image = False
        else:
            check_inside_image = True
        
        if self.tool_active["smooth"] and check_inside_image:
            x = int((e.x)*(self.view_w/self.canvas.winfo_width())) + self.view_x
            y = int((e.y)*(self.view_h/self.canvas.winfo_height())) + self.view_y
            op = "erosion" if shift_pressed else "dilation"
            self.apply_smoothing(y, x, operation=op)
            return
    
        if (self.tool_active["cut"] or self.tool_active["clean"]) and check_inside_image:
            self.connected_component_click(e, remove_only=(self.tool_active["cut"] and not shift_pressed))
            return
        
        if self.tool_active["wand"] and check_inside_image:
            self.sam_add_point(e, add=not shift_pressed, multipoint=ctrl_pressed)
            return
        
        if self.tool_active["wand_multi"] and check_inside_image:
            self.sam_add_point(e, add=not shift_pressed, multipoint=True)
            return
        
        if (self.tool_active["brush"] or self.tool_active["eraser"]) and check_inside_image:
            x = int((e.x)*(self.view_w/self.canvas.winfo_width())) + self.view_x
            y = int((e.y)*(self.view_h/self.canvas.winfo_height())) + self.view_y
            self.push_undo()
            self.brush_at(x, y, add=(self.tool_active["brush"] and not shift_pressed))
            self.update_display(update_all="Mask")
            self.draw_brush_preview(e)
            return

        
    def on_canvas_mid(self, e):
        self.mid_pressed = True
        self._pan_start = (e.x, e.y, self.view_x, self.view_y)
   
    def on_canvas_mid_release(self, e):
        self.mid_pressed = False
        self._pan_start = None 

    def on_canvas_left_release(self, e):
        self.last_brush_pos = None
        self._pan_start = None
        self._drag_counter = 0
        self.update_display()
        self.draw_brush_preview(e)
     

    def on_canvas_right(self, e):
        '''
        Handles right-clicks on the canvas, applying the active tool's removal 
        or erosion action without toggling tools.
        '''
        self.b3_pressed = True
        self.on_canvas_left(e)
        return;

    def on_canvas_right_release(self, e):
        self.b3_pressed = False
        self.on_canvas_left_release(e)

    def on_canvas_drag(self, e):
        if self.image_orig is None:
            return
        '''
        Updates the brush continuously while dragging the mouse.
        Draws one circle per event, using add/subtract depending on Shift.
        No interpolation between points to avoid undesired smoothing.
        '''
        shift_pressed = (e.state & 0x0001) != 0 or self.b3_pressed
        
        # Move the canvas if not tools selected
        if not any(self.tool_active[tool] for tool in self.tool_active) or self.mid_pressed:
            
            if self._pan_start is not None:
                x0, y0, ox0, oy0 = self._pan_start
                self.view_x = ox0 -int((e.x - x0)*(self.view_w/self.canvas.winfo_width()))
                self.view_y = oy0- int((e.y - y0)*(self.view_h/self.canvas.winfo_height()))
                self.update_display()
                self.update_preview_frame()
            return
        
        # Check if the brush is not active (only draggable tool)
        # TODO implement other tools
        if not (self.tool_active["brush"] or self.tool_active["eraser"]):
            return
        
        # Define the brush drag
        x1 = int((e.x)*(self.view_w/self.canvas.winfo_width())) + self.view_x
        y1 = int((e.y)*(self.view_h/self.canvas.winfo_height())) + self.view_y
        
        if not hasattr(self, "_prev_brush_pos") or self._prev_brush_pos is None:
            self._prev_brush_pos = (x1, y1)
            self.push_undo() # TODO: Check problem for undo
            self.brush_at(x1, y1, add=(self.tool_active["brush"] and not shift_pressed))
            self.update_display(update_all="Mask")
            self.draw_brush_preview(e)
            
            return
        
        # Skip some updates when zooming
        now = time.monotonic()
        
        if not hasattr(self, "_last_brush_update"):
            self._last_brush_update = 0.0
        
        if now - self._last_brush_update >= REFRESH_RATE_BRUSH:
            x0, y0 = self._prev_brush_pos
            dx = x1 - x0
            dy = y1 - y0
            dist = max(1, int(np.hypot(dx, dy))) # Distance between previous and current point (in pixel)
            r = max(1, self.brush_size // 2)
            steps = max(3, dist*3 // r) # draw this number of circles along (x0, y0) and (x1, y1)
            for i in np.linspace(0, dist + 1, steps):
                xi = int(x0 + dx * i / dist)
                yi = int(y0 + dy * i / dist)
                self.brush_at(xi, yi, add=(self.tool_active["brush"] and not shift_pressed))

            self.update_display(update_all="Mask") # update only mask
            self.draw_brush_preview(e)
            self._last_brush_update = now
            self._prev_brush_pos = (x1, y1)
    
    def on_canvas_track(self, e):
        '''
        Update label in statusbar depending on mouse position
        '''
        if self.image_orig is None:
            return
        
        x1 = int((e.x)*(self.view_w/self.canvas.winfo_width())) + self.view_x
        y1 = int((e.y)*(self.view_h/self.canvas.winfo_height())) + self.view_y
        
        # TODO implement z change
        self.pos_label_var.set(f"| x: {x1} | y: {y1} | z: 0 |")

    #%% KEYBOARD EVENTS
    def shiftPressed(self):
        # in case brush is active, set preview to "dashed"
        self._draw_brush_preview(self.mouse['x'], self.mouse['y'], True)

    def shiftReleased(self):
        # in case brush is active, set preview to "solid"
        self._draw_brush_preview(self.mouse['x'], self.mouse['y'])

    def tab(self):
        return self._tab(-1, 0, 1)

    def shiftTab(self):
        return self._tab(0, -1, -1)

    def _tab(self, id_key_to_check, id_key_to_get, increment):
        """
        Use TAB to cycle through masks
        """
        if len(self.mask_labels) == 0: # if there are no mask, do nothing
            return

        keys = list(self.mask_labels.keys())

        if len(self.mask_labels) == 1: 
            self.change_mask(keys[0])
            return

        if self.active_mask_id == keys[id_key_to_check]:
            self.change_mask(keys[id_key_to_get])
            return

        newIndex = keys.index(self.active_mask_id) + increment
        self.change_mask(keys[newIndex])

    #%% WINDOW EVENTS
    def on_resize(self, e):
        """
        Redraw canvas after window resize
        """
        if e.widget is self: # prevent firing during other events
            # while resizing, cancel the scheduled update_display event
            if self.resizing_event is not None:
                self.after_cancel(self.resizing_event)
            # if still resizing, schedule a new event
            self.resizing_event = self.after(300, self.update_display_after_resize)
    
    #%% PAN & ZOOM
    def pan_view(self, dx, dy):
        '''
        Pan view when distance is fixed. Used e.g. to bind keyboard arrows
        '''
        if not self.image_is_loaded():
            return
        self.view_x += dx
        self.view_y += dy
        self.update_display()
        self.update_preview_frame()
    
    def zoom_evt(self, e):
        '''
        Adjusts the zoom level of the displayed image based on mouse wheel 
        input and refreshes the display.
        '''
        if e.delta > 0:
            self.zoom_in(e)
        else:
            self.zoom_out(e)
        
    def zoom_in(self, e=None):
        '''
        Adjust zoom level (zoom in).
        '''
        # Check if an image is loaded
        if not self.image_is_loaded():
            return
        
        # change status while zoom function is inefficient, so that user is aware
        self.set_status("loading", "Zooming in...")
        
        # apply zoom
        if (self.zoom * 1.1) < self.zoom_max:
            self.zoom *= 1.1
        

        old_h = self.view_h
        old_w = self.view_w
        self.view_h = int(self.canvas.winfo_height()/self.zoom)
        self.view_w = int(self.canvas.winfo_width()/self.zoom)
        
        if e is not None:
            x = (e.x)/self.canvas.winfo_width()
            y = (e.y)/self.canvas.winfo_height()
        else:
            x = 0.5
            y = 0.5
        dx = round(x * (old_w - self.view_w))
        dy = round(y * (old_h - self.view_h))
        self.view_x += dx
        self.view_y += dy
            
            

        self.update_display()
        self.update_preview_frame()
        self.set_status("ready", "Ready")


    def zoom_out(self, e=None):
        '''
        Adjust zoom level (zoom out).
        '''
        # Check if an image is loaded
        if not self.image_is_loaded():
            return
        
        # change status while zoom function is inefficient, so that user is aware
        self.set_status("loading", "Zooming out...")
        
        # apply zoom
        if (self.zoom * 0.9) > self.zoom_min:
            self.zoom *= 0.9
        
        old_h = self.view_h
        old_w = self.view_w
        self.view_h = int(self.canvas.winfo_height()/self.zoom)
        self.view_w = int(self.canvas.winfo_width()/self.zoom)
        
        if e is not None:
            x = (e.x)/self.canvas.winfo_width()
            y = (e.y)/self.canvas.winfo_height()
        else: # use center of canvas
            x = 0.5
            y = 0.5
        dx = round(x * (old_w - self.view_w))
        dy = round(y * (old_h - self.view_h))
        self.view_x += dx
        self.view_y += dy

        
        self.update_display()
        self.update_preview_frame()
        self.set_status("ready", "Ready")
        
        
    def reset_zoom(self, e=None):
        '''
        Reset zoom (Ctrl-0, Ctrl-Space).
        '''
        if not self.image_is_loaded():
            return
        self.zoom = 1.0
        
        self.view_x = 0
        self.view_y = 0
        self.view_w = self.canvas.winfo_width()#min(self.canvas.winfo_width(), self.orig_w)
        self.view_h = self.canvas.winfo_height()#min(self.canvas.winfo_height(), self.orig_h)
        
        self.update_display()
        self.update_preview_frame()


    #%% TOOLS
    # BRUSH
    # def brush(self, e, add=True):
    #     '''
    #     Paints or erases a circular area on the active mask at the mouse 
    #     position, saving the previous state for undo and updating the display.
    #     '''
    #     if self.mask_orig is None or self.active_mask_id is None:
    #         return
    #     self.push_undo()
    
    #     # Mouse position in image coordinates
    #     x = int((e.x)*(self.view_w/self.canvas.winfo_width())) + self.view_x
    #     y = int((e.y)*(self.view_h/self.canvas.winfo_height())) + self.view_y
        
    #     r = self.brush_size // 2
    
    #     # Extend bounding box slightly to avoid gaps
    #     buffer = max(1, r // 2)
    #     y0, y1 = max(0, y - r - buffer), min(self.mask_orig.shape[0], y + r + buffer)
    #     x0, x1 = max(0, x - r - buffer), min(self.mask_orig.shape[1], x + r + buffer)
    
    #     # Create circular mask
    #     yy, xx = np.ogrid[y0:y1, x0:x1]
    #     circle = (yy - y)**2 + (xx - x)**2 <= r*r
    
    #     if add:
    #         if self.only_on_empty.get():
    #             mask_area = self.mask_orig[y0:y1, x0:x1]
    #             mask_area[circle & (mask_area==0)] = self.active_mask_id
    #         else:
    #             self.mask_orig[y0:y1, x0:x1][circle] = self.active_mask_id
    #     else:
    #         erase_mask = circle & (self.mask_orig[y0:y1, x0:x1] == self.active_mask_id)
    #         self.mask_orig[y0:y1, x0:x1][erase_mask] = 0
        
    #     self.set_modified(True)
    #     self.update_display(update_all="Mask")
    # TODO check if "brush" is still needed
    def brush_at(self, x, y, add=True):
        '''
        Aux method to define brush position without updating display or undo.
        '''
        # Return immediately if no mask or no active label
        if self.mask_orig is None or self.active_mask_id is None:
            return
        
        # Brush radius
        r = self.brush_size // 2
        buffer = max(1, r // 2)
        
        # Define bounding box of the brush, clamped to image edges
        y0 = max(0, y - r - buffer)
        y1 = min(self.mask_orig.shape[0], y + r + buffer)
        x0 = max(0, x - r - buffer)
        x1 = min(self.mask_orig.shape[1], x + r + buffer)
        
        # Create small local coordinate arrays (only the bounding box, not the whole mask)
        ys = np.arange(y0, y1)
        xs = np.arange(x0, x1)
        
        # Efficient broadcasting to create circle mask
        dy = ys[:, None] - y  # shape (height, 1)
        dx = xs[None, :] - x  # shape (1, width)
        circle = dx**2 + dy**2 <= r*r + 4 # boolean array, shape (y1-y0, x1-x0)
        
        # Slice of the mask corresponding to the bounding box
        mask_area = self.mask_orig[y0:y1, x0:x1]
        lock_area = self.mask_locked[y0:y1, x0:x1]
        
        if add:
            # Paint only on non-locked pixels
            mask_area[circle & (~lock_area)] = self.active_mask_id
        else:
            # Erase only pixels that match the active mask label
            # (independently from their locked status)
            erase_mask = circle & (mask_area == self.active_mask_id)
            mask_area[erase_mask] = 0
        
        # Update locked status
        # only on slices for performance
        lock_area[mask_area==self.active_mask_id] = self.mask_widgets[self.active_mask_id].locked
        lock_area[mask_area==0] = False
        # Mark mask as modified for later saving or GUI update
        self.set_modified(True)

    def draw_brush_preview(self, e):
        '''
        Draws a semi-transparent circle on the canvas to show the brush size
        and position before painting. The circle is solid in 'add mask' mode
        and dashed in 'remove mask' mode.
        '''
        self.mouse['x'], self.mouse['y'] = e.x, e.y # store mouse position
        x, y = e.x, e.y

        shift_pressed = (e.state & 0x0001) != 0 or self.b3_pressed
        self._draw_brush_preview(x, y, shift_pressed)

    def _draw_brush_preview(self, x, y, shift_pressed=False):
        self.canvas.delete("brush")
        
        if not (self.tool_active["brush"] or self.tool_active["eraser"]):
            return

        r = int(self.brush_size * self.zoom / 2)
        outline_color = "#" + "".join([f"{c:02x}" for c in self.mask_colors[self.active_mask_id]])
        dash = (5,10) if (shift_pressed or self.tool_active["eraser"]) else None

        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="", outline=outline_color, dash=dash, width=2, tag="brush")
        
    # SAM
    def sam_add_point(self, e, add=True, multipoint=False):
        """
        Add the clicked point to the list of points to be fed to SAM.
        
        If multipoint=False, add=True means "use only this point to compute the
        mask, and add it", while add=False means "use only this point to
        compute the mask, and remove it".
        If multipoint=True, add=True means "mark this point as foreground",
        while add=False means "mark this point as background"
        """
        if self.image_orig is None or self.active_mask_id is None:
            return
        x = int((e.x)*(self.view_w/self.canvas.winfo_width())) + self.view_x
        y = int((e.y)*(self.view_h/self.canvas.winfo_height())) + self.view_y
        self.sam_points.append([x, y])
        if multipoint:
            self.sam_pt_labels.append(1 if add else 0)
            if add:
                # fg point: fill with active mask color, outline black
                pt_fill = "#" + "".join([f"{c:02x}" for c in self.mask_colors[self.active_mask_id]])
                pt_out = "black"
            else:
                # bg point: fill black, use the inverted active mask color for outline
                pt_fill = "black"
                pt_out = "#" + "".join([f"{255-c:02x}" for c in self.mask_colors[self.active_mask_id]])
            self.canvas.create_oval(e.x-3, e.y-3, e.x+3, e.y+3, fill=pt_fill, outline=pt_out, width=1, tag="sam_pt")
            self.sam_compute(multipoint=True)
        else:
            self.sam_pt_labels.append(1)
            self.sam_compute(multipoint=False)
            self.sam_apply(add=add)

    def sam_compute(self, multipoint=False):
        """
        Use SAM points and labels lists to compute mask, and store it in the
        preview matrix.
        
        multipoint determines if one or three masks are computed
        """
        if (self.image_orig is None) or (self.active_mask_id is None) or (not self.sam_points):
            return
        self.set_status("loading", "SAM computing...")
            
        masks, scores, _ = self.sam.predict(np.array(self.sam_points),
                                            np.array(self.sam_pt_labels),
                                            multimask_output=not multipoint,
                                            return_logits=True)

        masks = expit(masks) > self.wand_threshold
        i = np.argmax(scores)
        self.sam_preview[masks[i] & (~self.mask_locked)] = True
        
        if multipoint: # to show preview
            self.update_display(update_all="Mask")
        self.set_status("ready", "Ready")

    def sam_apply(self, add=True, cancel=False):
        """
        Apply the mask in SAM_preview to definitive mask, and empty SAM points
        and labels lists.
        
        If cancel=True, don't apply the computed mask and only empty SAM infos.
        """
        if self.image_orig is None or self.active_mask_id is None:
            return
        if not cancel:
            self.push_undo()
            if add:
                self.mask_orig[self.sam_preview] = self.active_mask_id
            else:
                self.mask_orig[self.sam_preview & (self.mask_orig==self.active_mask_id)] = 0
            self.set_modified(True)
        self.sam_preview = np.full(self.mask_orig.shape, False) # reset preview
        self.sam_points = []
        self.sam_pt_labels = []
        self.canvas.delete("sam_pt")
        # Update locked status (we don't use self.update_lock() for performances)
        self.mask_locked[self.mask_orig==self.active_mask_id] = self.mask_widgets[self.active_mask_id].locked
        self.mask_locked[self.mask_orig==0] = False
        self.update_display(update_all="Mask")

    def sam_apply_release(self):
        """
        Event bound to the release of the "Multipoint" key
        """
        # TODO check other active tools
        if (not self.tool_active["wand"]) or (not self.sam_points): # empty lists are false
            return
        self.sam_apply(add=True) # multipoint only adds mask

    def manual_wand_preprocessing(self):
        values = PreprocessingAdjustments(self).values
        if values is not None:
            self.wand_brightness, self.wand_contrast, self.wand_gamma = values
            self.wand_brightness_lbl.configure(text=str(self.wand_brightness))
            self.wand_contrast_lbl.configure(text=str(self.wand_contrast))
            self.wand_gamma_lbl.configure(text=str(self.wand_gamma))
            # reload SAM image
            # deactivate tools
            self.deactivate_tools()
            if self.thread is None or not self.thread.is_alive():
                self.thread = threading.Thread(target=self.async_loader, daemon=True)
                self.thread.start()

    # CONNECTED COMPONENT
    def get_connected_component(self, mask, start_y, start_x, target_id):
        '''
        Computes and returns a boolean mask of all pixels connected to a 
        starting point that belong to the given mask ID, using a depth-first 
        search.
        '''
        if mask[start_y, start_x] != target_id:
            return np.zeros_like(mask, dtype=bool)
        h, w = mask.shape
        visited = np.zeros_like(mask, dtype=bool)
        component = np.zeros_like(mask, dtype=bool)
        stack = [(start_y, start_x)]
        while stack:
            y, x = stack.pop()
            if visited[y, x]:
                continue
            visited[y, x] = True
            if mask[y, x] == target_id:
                component[y, x] = True
                for ny, nx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                    if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                        stack.append((ny,nx))
        return component

    def connected_component_click(self, e, remove_only=True):
        '''
        Handles a click for the connected component tool, removing either only 
        the clicked component or all other pixels of the active mask, saving 
        the previous state for undo, and updating the display.
        '''
        if self.mask_orig is None or self.active_mask_id is None:
            return
        
        x = int((e.x)*(self.view_w/self.canvas.winfo_width())) + self.view_x
        y = int((e.y)*(self.view_h/self.canvas.winfo_height())) + self.view_y
        
        comp = self.get_connected_component(self.mask_orig, y, x, self.active_mask_id)
        self.push_undo()
        
        # notice that "connected component" only acts on active mask,
        # so the lock check is not needed
        if remove_only:
            self.mask_orig[comp] = 0
        else:
            self.mask_orig[(self.mask_orig==self.active_mask_id) & (~comp)] = 0
        
        self.set_modified(True)
        # Update locked status (we don't use self.update_lock() for performances)
        self.mask_locked[self.mask_orig==self.active_mask_id] = self.mask_widgets[self.active_mask_id].locked
        self.mask_locked[self.mask_orig==0] = False
        self.update_display(update_all="Mask")
        

    # SMOOTHING (EROSION + DILATION)
    def apply_smoothing(self, y, x, operation="dilation", size=3):
        '''
        Applies dilation or erosion to the connected component under the 
        clicked point, saving the previous state for undo, updating the mask 
        with the smoothed result, and refreshing the display.
        '''
        if self.mask_orig is None or self.active_mask_id is None:
            return
        
        self.set_status("loading", "Applying smoothing...")
        # Identify the connected component
        comp = self.get_connected_component(self.mask_orig, y, x, self.active_mask_id)
        if not comp.any():
            self.set_status("ready", "Ready")
            return
    
        self.push_undo()
    
        struct = np.ones((size, size), dtype=bool)
        
        comp_smooth = comp.copy()
        
        if operation == "dilation":
            for _ in range(self.smooth_iter):
                if self.smooth_n_erosions > 0:
                    comp_smooth = binary_erosion(comp_smooth, structure=struct, iterations=self.smooth_n_erosions)
                if self.smooth_n_dilations > 0:
                    comp_smooth = binary_dilation(comp_smooth, structure=struct, iterations=self.smooth_n_dilations)
        elif operation == "erosion":
            for _ in range(self.smooth_iter):
                if self.smooth_n_dilations > 0:
                    comp_smooth = binary_dilation(comp_smooth, structure=struct, iterations=self.smooth_n_dilations)
                if self.smooth_n_erosions > 0:
                    comp_smooth = binary_erosion(comp_smooth, structure=struct, iterations=self.smooth_n_erosions)
        else:
            return

        
        self.mask_orig[comp] = 0
        # (comp|(~self.mask_locked)) means: "during erosion, allow changes on
        # the old component even if active mask is locked"
        self.mask_orig[comp_smooth & (comp|(~self.mask_locked))] = self.active_mask_id
        
        self.set_modified(True)
        # Update locked status (we don't use self.update_lock() for performances)
        self.mask_locked[self.mask_orig==self.active_mask_id] = self.mask_widgets[self.active_mask_id].locked
        self.mask_locked[self.mask_orig==0] = False
        self.update_display(update_all="Mask")
        self.set_status("ready", "Ready")

#%% Main cycle
if __name__ == "__main__":
    SegmentationApp().mainloop()
