'''
SLImTAG gui. Version 1.0

20 Jan 2026

Giulio Del Corso & Oscar Papini
'''


#%% Libraries
import os
import time

# Numerical arrays manipulation
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

# Image manipulation and TkIntert (ImageTk)
from PIL import Image, ImageDraw, ImageTk

# TkInter and CustomTkInter GUI
import tkinter as tk
from tkinter import filedialog#, simpledialog, messagebox
import customtkinter as ctk

# Custom utils
from slimtag_utils import MultiButtonDialog, EntryDialog

# Torch and SAM (Segment anything model)
import torch
from segment_anything import sam_model_registry, SamPredictor

# Asynchronous threading import
import threading



#%% Suppress specific PyTorch warnings
import warnings

warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`"
)



#%% User selected parameters
MAX_DISPLAY = 800   # Maximum display size for resizing images
UNDO_DEPTH = 10     # Maximum number of undo steps

MAX_ZOOM_PIXEL = 32 # minimum number of pixels of orig image visible at max zoom level
MIN_ZOOM_PIXEL = 8192 # maximum number of pixels of orig image visible at max zoom level
# TODO upgrade min zoom management
#MIN_ZOOM_CANVAS = 0.9 # for images NOT smaller than (MIN_ZOOM_CANVAS)*(canvas dimension), this is the maximum value that the ratio (canvas dimension)/(image dimension) can achieve
#MIN_ZOOM_FACTOR = 0.05 # minimum zoom value IN ANY CASE

REFRESH_RATE_BRUSH = 0.05    # Refresh rate for the brush

# predefined high contrast colors for masks
MAX_MASKS = 20 
HIGH_CONTRAST_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (255, 128, 0), (128, 0, 255), (0, 255, 128),
    (128, 255, 0), (255, 0, 128), (0, 128, 255), (128, 128, 0),
    (128, 0, 0), (0, 128, 0), (0, 0, 128), (200, 200, 200),
    (255, 200, 200), (200, 255, 200), (200, 200, 255)
]

# colors for different tool states
TOOL_OFF_COLOR = "#3A3A3A"   # neutral grey when tool is off
BRUSH_ON_COLOR = "#4CAF50"   # green
MAGIC_ON_COLOR = "#FF9800"   # orange
CC_ON_COLOR = "#9C27B0"      # purple
SMOOTH_ON_COLOR = "#2196F3"  # blue

STATUS_SYMBOL = "●"
STATUS_COLOR = {
    "ready":  ("#2ECC71", "#2ECC71"),  # green
    "loading":("#F1C40F", "#F1C40F"),  # yellow
    "error":  ("#E74C3C", "#E74C3C"),  # red
    "idle":   ("#95A5A6", "#95A5A6"),  # gray
    }


#%% SAM parameters
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
ctk.set_appearance_mode("System")   # System theme
#ctk.set_appearance_mode("dark") # force dark mode for testing
ctk.set_default_color_theme("blue") # CTK color theme

#%% Main class
class SegmentationApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title(f"SLImTAG ({MODEL_TYPE})")
        self.geometry("1300x900")

        # STATE ---------------------------------------------------------------
        # Full image and mask
        self.image_orig = None
        self.mask_orig = None
        # Displayed image and mask
        self.image_disp = None
        self.mask_disp = None
        
        # aux display variables
        self.mask_pil = None
        self.tk_ov = None
        
        # to keep track of resizing window
        self.resizing_event = None
        
        # boolean switch to check if mask is modified and not saved
        self.modified = False
        
        # zoom & pan status
        self.zoom = 1.0
        self.zoom_max = 1.0
        self.zoom_min = 1.0
        self._pan_start = None
        
        self.zoom_label_var = tk.StringVar(self, value="Zoom: 100%")
        
        # Original values for rescale
        self.orig_h = None
        self.orig_w = None

        # Masks stuff
        self.mask_labels = {}
        self.mask_colors = {}
        self.mask_widgets = {}
        self.active_mask_id = None
        
        # switch for locking mask
        self.only_on_empty = tk.BooleanVar(self, value=False)

        # tools status
        self.brush_active = False
        self.magic_mode = False
        self.cc_mode = False 
        self.smoothing_active = False
        
        # brush control
        self.last_brush_pos = None
        self.brush_size = 30
        self.mouse = {'x': None, 'y': None}
        
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

        # boolean to track if right mouse button is pressed
        self.b3_pressed = False

        # UI ------------------------------------------------------------------
        
        # Top Menu
        self.menu_bar = tk.Menu(self)
        self.config(menu=self.menu_bar)
        # Menu File (top menu)
        file_menu = tk.Menu(self.menu_bar, tearoff=0)
        file_menu.add_command(label="Quit", command=self.quit_program, accelerator="Ctrl+Q")
        self.menu_bar.add_cascade(label="File", menu=file_menu)
        # Menu Edit (top menu)
        edit_menu = tk.Menu(self.menu_bar, tearoff=0)
        edit_menu.add_command(label="Undo", command=self.undo, accelerator="Ctrl+Z")
        self.menu_bar.add_cascade(label="Edit", menu=edit_menu)
        # Menu View (top menu)
        view_menu = tk.Menu(self.menu_bar, tearoff=0)
        view_menu.add_command(label="Zoom in", command=self.zoom_in, accelerator="Ctrl++")
        view_menu.add_command(label="Zoom out", command=self.zoom_out, accelerator="Ctrl+-")
        view_menu.add_command(label="Reset zoom", command=self.reset_zoom, accelerator="Ctrl+0")
        self.menu_bar.add_cascade(label="View", menu=view_menu)
        # Menu Image (top menu)
        image_menu = tk.Menu(self.menu_bar, tearoff=0)
        image_menu.add_command(label="Import image", command=self.load_image, accelerator="Ctrl+I")
        self.menu_bar.add_cascade(label="Image", menu=image_menu)
        # Menu Mask (top menu)
        mask_menu = tk.Menu(self.menu_bar, tearoff=0)
        mask_menu.add_command(label="Load mask", command=self.load_mask)
        mask_menu.add_command(label="Save mask", command=self.save_mask, accelerator="Ctrl+S")
        mask_menu.add_separator()
        mask_menu.add_command(label="Clear active mask", command=self.clear_active_mask)
        mask_menu.add_command(label="Clear all masks", command=self.clear_all_masks)
        self.menu_bar.add_cascade(label="Mask", menu=mask_menu)
        
        panels_width = 250
        # Left panel for tools
        self.tools_panel = ctk.CTkFrame(self, width=panels_width)
        self.tools_panel.grid(row=0, column=0, sticky="nsew")
        
        # Main canvas
        self.canvas = ctk.CTkCanvas(self, bg="black")
        self.canvas.grid(row=0, column=1, sticky="nsew")
        
        # Right panel for masks
        self.masks_panel = ctk.CTkFrame(self, width=panels_width)
        self.masks_panel.grid(row=0, column=2, sticky="nsew")
        
        # Statusbar
        self.statusbar = ctk.CTkFrame(self, height=24, fg_color=("gray92", "gray14"))
        self.statusbar.grid(row=1, column=0, columnspan=3, sticky="nsew", padx=0, pady=0)

        self.status_icon = ctk.CTkLabel(self.statusbar, text=STATUS_SYMBOL, text_color=STATUS_COLOR["idle"], width=14)
        self.status_icon.grid(row=0, column=0, sticky="w", padx=(10, 0), pady=(0, 2))
        self.status_label = ctk.CTkLabel(self.statusbar, text="Initializing...")
        self.status_label.grid(row=0, column=1, sticky="w", padx=(4, 0))
        
        self.zoom_label = ctk.CTkLabel(self.statusbar, textvariable=self.zoom_label_var)
        self.zoom_label.grid(row=0, column=3, sticky="e", padx=10)
        
        self.statusbar.grid_columnconfigure(2, weight=1)
        
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        
        # Mask controls
        # Add mask button
        ctk.CTkButton(self.masks_panel, text="Add new mask [N]", command=self.add_mask).grid(row=0, column=0, sticky="ew", padx=10, pady=(10,5))
        
        # ScrollFrame for mask list
        self.mask_list_frame = ctk.CTkScrollableFrame(self.masks_panel)
        self.mask_list_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(5, 10))
        
        self.masks_panel.grid_rowconfigure(1, weight=1)

        # Tool buttons
        # Frame for buttons
        self.tools_btn_frame = ctk.CTkFrame(self.tools_panel)
        self.tools_btn_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 5))
        # Frame for tool options
        self.tool_opt_frame = ctk.CTkFrame(self.tools_panel)
        self.tool_opt_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        
        self.brush_btn = ctk.CTkButton(self.tools_btn_frame, text="Brush [B]", command=self.toggle_brush)
        self.brush_btn.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        
        ctk.CTkLabel(self.tool_opt_frame, text="Brush size", font=ctk.CTkFont(size=11), fg_color="transparent", anchor="w").grid(row=0, column=0, sticky="ew", padx=10, pady=(8,2))
        self.brush_slider = ctk.CTkSlider(self.tool_opt_frame, from_=5, to=100, command=lambda v: setattr(self,"brush_size",int(v)))
        self.brush_slider.set(self.brush_size)
        self.brush_slider.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 8))
        

        self.magic_btn = ctk.CTkButton(self.tools_btn_frame, text="Magic wand [M]", command=self.toggle_magic)
        self.magic_btn.grid(row=1, column=0, sticky="ew", padx=10, pady=5)

        self.smoothing_btn = ctk.CTkButton(self.tools_btn_frame, text="Smoothing [S]", command=self.toggle_smoothing)
        self.smoothing_btn.grid(row=2, column=0, sticky="ew", padx=10, pady=5)

        self.cc_btn = ctk.CTkButton(self.tools_btn_frame, text="Connected component [C]", command=self.toggle_cc_mode)
        self.cc_btn.grid(row=3, column=0, sticky="ew", padx=10, pady=(5, 10))
        
        self.tools_btn_frame.grid_columnconfigure(0, weight=1)
        self.tool_opt_frame.grid_columnconfigure(0, weight=1)
        
        # Buttons configuration
        self.brush_btn.configure(fg_color=TOOL_OFF_COLOR)
        self.magic_btn.configure(fg_color=TOOL_OFF_COLOR)
        self.smoothing_btn.configure(fg_color=TOOL_OFF_COLOR)
        self.cc_btn.configure(fg_color=TOOL_OFF_COLOR)
        self.all_action_buttons = [self.brush_btn, self.magic_btn, self.cc_btn, self.smoothing_btn]
        
        # Toggle for only empty
        ctk.CTkSwitch(self.tools_panel, text="Only add on empty", variable=self.only_on_empty).grid(row=2, column=0, sticky="ew", padx=10, pady=5)

        # Undo button
        ctk.CTkButton(self.tools_panel, text="Undo [Ctrl-Z]", command=self.undo).grid(row=3, column=0, sticky="ew", padx=10, pady=(5, 10))

        self.tools_panel.grid_rowconfigure(1, weight=1)

        # SAM -----------------------------------------------------------------
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[MODEL_TYPE](checkpoint=MODEL_WEIGHTS_PATH)
        sam.to(device).eval()
        self.sam = SamPredictor(sam)

        # Define the asynchronous mechanism to speed up image loading
        self.switch_computed_magic_wand = False     # True if SAM is loaded
        self.thread = None                          # Threading variable
        self.lock = threading.Lock()              # To protect shared varaibles
        
        self.set_controls_state(False) # Deactivate all buttons -- must be done after defining switch_computed_magic_wand

        # BINDINGS ------------------------------------------------------------
        self.canvas.bind("<MouseWheel>", self.zoom_evt)
        self.canvas.bind("<Button-4>", self.zoom_in) # <Button-4> is scroll up for Linux
        self.canvas.bind("<Button-5>", self.zoom_out) # <Button-5> is scroll down for Linux
        self.canvas.bind("<Motion>", self.draw_brush_preview)
        self.canvas.bind("<Button-1>", self.on_canvas_left)
        self.canvas.bind("<Button-3>", self.on_canvas_right)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<B3-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_left_release)
        self.canvas.bind("<ButtonRelease-3>", self.on_canvas_right_release)
        
        # Zoom via keyboard (Ctrl + / Ctrl -)
        self.bind("<Control-plus>", lambda e: self.zoom_in())
        #self.bind("<Control-equal>", self.reset_zoom) # usually it is Ctrl-0
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
        
        # SHORTCUT KEYS -------------------------------------------------------
        self.bind("<b>", lambda e: self.toggle_brush())
        self.bind("<B>", lambda e: self.toggle_brush())
        self.bind("<m>", lambda e: self.toggle_magic())
        self.bind("<M>", lambda e: self.toggle_magic())
        self.bind("<c>", lambda e: self.toggle_cc_mode())
        self.bind("<C>", lambda e: self.toggle_cc_mode())
        self.bind("<s>", lambda e: self.toggle_smoothing())
        self.bind("<S>", lambda e: self.toggle_smoothing())
        #self.bind("<z>", lambda e: self.undo()) # I actually prefer Ctrl-Z, but
        #self.bind("<Z>", lambda e: self.undo()) # we can discuss about this. -Oscar
        self.bind("<n>", lambda e: self.add_mask())
        self.bind("<N>", lambda e: self.add_mask())
        self.bind("<Control-z>", lambda e: self.undo())
        self.bind("<Control-Z>", lambda e: self.undo())
        self.bind("<Control-I>", lambda e: self.load_image())
        self.bind("<Control-i>", lambda e: self.load_image())
        self.bind("<Control-S>", lambda e: self.save_mask())
        self.bind("<Control-s>", lambda e: self.save_mask())
        self.bind("<Control-q>", lambda e: self.quit_program())
        self.bind("<Control-Q>", lambda e: self.quit_program())
        
        self.bind("<KeyPress-Shift_L>", lambda e: self.shiftPressed())
        self.bind("<KeyPress-Shift_R>", lambda e: self.shiftPressed())
        self.bind("<KeyRelease-Shift_L>", lambda e: self.shiftReleased())
        self.bind("<KeyRelease-Shift_R>", lambda e: self.shiftReleased())
        
        
        # Finally, set status to "Ready"
        self.set_status("ready", "Ready")
        
        
        
    #%% ASYNC METHOD FOR EFFICIENT SAM UPLOAD ---------------------------------
    def async_loader(self):
        print("Load SAM model")
        
        #  Thread-safe upload of shared variable 
        with self.lock:
            self.switch_computed_magic_wand = False
            # SAM model inference on image
            self.sam.set_image(np.array(self.image_orig))
            
            # Turn on swithc
            self.switch_computed_magic_wand = True
            
        print("Loaded SAM model")
        self.set_status("ready", "Ready")
        # Refresh and update display
        self.update_display()
        
        
        
    #%% AUX METHODS  ----------------------------------------------------------
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
        cut_mask_orig[top-self.view_y:bottom-self.view_y, left-self.view_x:right-self.view_x] = self.mask_orig[top: bottom, left:right]
        
        # create overlay object and convert it to be pasted on canvas
        overlay = np.zeros((self.view_h, self.view_w, 4), np.uint8)
        for mid ,c in self.mask_colors.items():
            overlay[cut_mask_orig==mid] = [*c, 150]
        self.mask_pil = Image.fromarray(overlay)
        resized = self.mask_pil.resize((self.canvas.winfo_width(), self.canvas.winfo_height()), Image.NEAREST)
        self.tk_ov = ImageTk.PhotoImage(resized)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_ov, tag="mask")
        
        
        # raise back SAM multipoints if any
        self.canvas.tag_raise("sam_pt")

    def update_button_colors(self):
        '''
        Update button colors based on active tool.
        '''
        self.brush_btn.configure(fg_color=BRUSH_ON_COLOR if self.brush_active else TOOL_OFF_COLOR)
        self.magic_btn.configure(fg_color=MAGIC_ON_COLOR if self.magic_mode else TOOL_OFF_COLOR)
        self.cc_btn.configure(fg_color=CC_ON_COLOR if self.cc_mode else TOOL_OFF_COLOR)
        self.smoothing_btn.configure(fg_color=SMOOTH_ON_COLOR if self.smoothing_active else TOOL_OFF_COLOR)
        
        
    def deactivate_tools(self):
        '''
        Keep one tool button active at time.
        '''
        self.brush_active = False
        self.magic_mode = False
        self.cc_mode = False
        self.smoothing_active = False
        self.update_button_colors()
        
        
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
                self.load_image()
            else:
                return False
        return True

    def set_controls_state(self, enabled: bool):
        '''
        Enable/disable all buttons.
        '''
        state = "normal" if enabled else "disabled"
        for b in self.all_action_buttons:
            b.configure(state=state)
        if not self.switch_computed_magic_wand:
            self.magic_btn.configure(state="disabled")

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
            title = self.title()
            if state == True:
                self.modified = True
                self.title("*"+title)
            else: # state == False
                self.modified = False
                self.title(title[1:]) # remove '*'

    def shiftPressed(self):
        # in case brush is active, set preview to "dashed"
        self._draw_brush_preview(self.mouse['x'], self.mouse['y'], True)

    def shiftReleased(self):
        # in case brush is active, set preview to "solid"
        self._draw_brush_preview(self.mouse['x'], self.mouse['y'])

    # TODO
    # def compute_zoom_limits(self):
    #     """
    #     Compute new zoom limits when an image is loaded or when the canvas is resized,
    #     depending on the image size and the (current/updated) canvas size
    #     """
    #     if an image is not loaded: return
    #     compute limits and set self.zoom_min and self.zoom_max
    #     call this function from load_image and aso bind it to the function called at window resize

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
    
    def update_display_after_resize(self):
        self.view_h = int(self.canvas.winfo_height()/self.zoom)
        self.view_w = int(self.canvas.winfo_width()/self.zoom)
        self.update_display(update_all="Global")

    def quit_program(self):
        """
        Quit program.
        """
        if self.modified:
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

    #%% TOOL BUTTONS ----------------------------------------------------------
    # UNDO --------------------------------------------------------------------
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
            self.update_display(update_all="Mask")


    # MASK MANAGEMENT ---------------------------------------------------------
    def add_mask(self):
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
        
        # Ask user for mask name
        name_dialog = EntryDialog(self, message="New mask name:")
        name = name_dialog.value
        if not name:  # User cancelled or empty
            return
        
        mid = min([i+1 for i in range(MAX_MASKS) if i+1 not in self.mask_labels.keys()])
        self.mask_labels[mid] = name
        self.mask_colors[mid] = HIGH_CONTRAST_COLORS[mid-1]
        
        self.mask_widgets[mid] = self.create_mask_widget(mid)
        self.mask_widgets[mid].pack(fill="x", expand=True)
        self.change_mask(target_id=mid) # this also sets self.active_mask_id

        self.set_controls_state(True) # activate buttons if there is at least one mask
    
    def create_mask_widget(self, mid):
        circle_size = 21
        mask_frame = ctk.CTkFrame(self.mask_list_frame)
        mask_frame._default_fg_color = mask_frame.cget("fg_color")
        color_circle = Image.new("RGBA", (circle_size+1, circle_size+1), (0, 0, 0, 0))
        color_circle_draw = ImageDraw.Draw(color_circle)
        color_circle_draw.ellipse((0, 0, circle_size, circle_size), fill=self.mask_colors[mid])
        mask_crc = ctk.CTkLabel(mask_frame, text="", image=ctk.CTkImage(color_circle, size=(circle_size+1, circle_size+1)))
        mask_crc.grid(row=0, column=0, padx=(10,5), pady=10)
        mask_frame.lbl = ctk.CTkLabel(mask_frame, text=f"{mid}: {self.mask_labels[mid]}", anchor="w")
        mask_frame.lbl.grid(row=0, column=1, sticky="ew", padx=5, pady=10)
        mask_frame._default_text_color = mask_frame.lbl.cget("text_color")
        clear_btn = ctk.CTkButton(mask_frame, text="×",
                                  font=ctk.CTkFont(size=18, weight="bold"),
                                  width=12, height=12,
                                  fg_color="transparent",
                                  text_color="red",
                                  command=lambda mid=mid: self.clear_mask(mid))
        clear_btn.grid(row=0, column=2, padx=(5,10), pady=10)
        clear_btn.bind("<Enter>", lambda e: clear_btn.configure(fg_color="#CC0000", text_color="white"))
        clear_btn.bind("<Leave>", lambda e: clear_btn.configure(fg_color="transparent", text_color="red"))
        mask_frame.grid_columnconfigure(1, weight=1)
        mask_frame.bind("<Button-1>", lambda e, mid=mid: self.change_mask(e, mid))
        mask_crc.bind("<Button-1>", lambda e, mid=mid: self.change_mask(e, mid))
        mask_frame.lbl.bind("<Button-1>", lambda e, mid=mid: self.change_mask(e, mid))
        if 1 <= mid <= 9:
            self.bind(f"<Key-{mid}>", lambda e, tid=mid: self.change_mask(e, tid))
        return mask_frame

    def change_mask(self, e=None, target_id=None):
        '''
        Changes the currently active mask based on the user selection in the 
        combo box.
        '''
        # Retrieves the mask ID corresponding to the current selection
        if self.active_mask_id:
            self.mask_widgets[self.active_mask_id].configure(fg_color=self.mask_widgets[self.active_mask_id]._default_fg_color)
            self.mask_widgets[self.active_mask_id].lbl.configure(text_color=self.mask_widgets[self.active_mask_id]._default_text_color)
        self.active_mask_id = target_id
        # Change appearance of mask row in mask list
        self.mask_widgets[target_id].configure(fg_color="white")
        self.mask_widgets[target_id].lbl.configure(text_color="black")
        # Updates the UI buttons’ colors
        self.update_button_colors()
        self.set_controls_state(True)

    # CLEAR MASK --------------------------------------------------------------
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
        if self.mask_orig is None: return
        self.push_undo()
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
        if self.active_mask_id is None: return
        self.clear_mask(self.active_mask_id)
    
    def clear_all_masks(self):
        mask_ids = list(self.mask_labels.keys())
        for mid in mask_ids:
            self.clear_mask(mid)


    # BUTTON TOGGLES ----------------------------------------------------------
    def toggle_brush(self):
        '''
        Activates or deactivates the brush tool.
        '''
        # Check if an image is loaded
        if not self.image_is_loaded():
            return
        
        if self.brush_btn.cget('state') != "disabled":
            if not self.brush_active:
                self.deactivate_tools()
                self.brush_active = True
            else:
                self.brush_active = False
            self.update_button_colors()

        self._draw_brush_preview(self.mouse['x'], self.mouse['y'])

    def toggle_cc_mode(self):
        '''
        Activates or deactivates the connected component tool.
        '''
        # Check if an image is loaded
        if not self.image_is_loaded():
            return
        
        if self.cc_btn.cget('state') != "disabled":
            if not self.cc_mode:
                self.deactivate_tools()
                self.cc_mode = True
            else:
                self.cc_mode = False
            self.update_button_colors()


    def toggle_magic(self):
        '''
        Activates or deactivates the magic wand tool.
        '''
        # Check if an image is loaded
        if not self.image_is_loaded():
            return
        
        if self.magic_btn.cget('state') != "disabled":
            if not self.magic_mode and self.switch_computed_magic_wand:
                self.deactivate_tools()
                self.magic_mode = True
            else:
                self.magic_mode = False
            self.update_button_colors()

    def toggle_smoothing(self):
        '''
        Activates or deactivates the smoothing tool.
        '''
        # Check if an image is loaded
        if not self.image_is_loaded():
            return
        
        if self.smoothing_btn.cget('state') != "disabled":
            if not self.smoothing_active:
                self.deactivate_tools()
                self.smoothing_active = True
            else:
                self.smoothing_active = False
            self.update_button_colors()



    #%% IMAGE AND MASK --------------------------------------------------------
    def load_image(self):
        '''
        Load a .png or .jpg image and define and empty mask on it.
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
        p = filedialog.askopenfilename(filetypes=[("Image files", ("*.png", "*.jpg", "*.jpeg"))])
        if not p:
            return

        self.set_status("loading", "Loading image...")
        img = Image.open(p).convert("RGB")
        self.orig_w, self.orig_h = img.size
        self.image_orig = img
        self.mask_orig = np.zeros((self.orig_h, self.orig_w), np.uint8)
        
        # Async load of the SAM model to avoid freezed interface
        if self.thread is None or not self.thread.is_alive():
               self.thread = threading.Thread(target=self.async_loader, daemon=True)
               self.thread.start()
        
        # reset masks
        self.clear_all_masks()
        
        # TODO refine zoom_min
        
        # reset zoom
        self.zoom = 1.0
        # Define a max and min zoom
        self.zoom_max = max(self.canvas.winfo_width() / MAX_ZOOM_PIXEL, self.canvas.winfo_height() / MAX_ZOOM_PIXEL)
        self.zoom_min = min(self.canvas.winfo_width() / MIN_ZOOM_PIXEL, self.canvas.winfo_height() / MIN_ZOOM_PIXEL)
        #self.zoom_max = max(self.orig_w // MAX_ZOOM_PIXEL, self.orig_h // MAX_ZOOM_PIXEL, 1)
        #self.zoom_min = min(max(min(self.canvas.winfo_width()/self.orig_w, self.canvas.winfo_height()/self.orig_h)*0.9, MIN_ZOOM_FACTOR), 0.9)
        # min(...)*0.9 to be able to potentially see all image on the current canvas
        # MIN_ZOOM_FACTOR hard-coded if the canvas is small or the image is too big
        # min(..., 0.9) to prevent rescaling of very small images w.r.t. canvas

        # RESET HISTORY
        self.undo_stack.clear()
        
        # Define the view parameters (equal to the canvas size):
        self.update_idletasks()
        self.view_x = 0
        self.view_y = 0
        self.view_w = self.canvas.winfo_width()#min(self.canvas.winfo_width(), self.orig_w)
        self.view_h = self.canvas.winfo_height()#min(self.canvas.winfo_height(), self.orig_h)
            
        self.update_display()
        
        if self.switch_computed_magic_wand:
            self.set_status("ready", "Ready")
        else:
            self.set_status("ready", "Ready (Loading image into SAM...)")

    def load_mask(self):
        """
        Upload an existing mask
        - Indexed PNG (mode "P"): direct recovery of mask indices.
        - RGB PNG: legacy color-based reconstruction.
        """
        if not self.image_is_loaded():
            return
        
        self.deactivate_tools()
        p = filedialog.askopenfilename(filetypes=[("IndexedPNG or PNG", "*.png")])
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

        self.update_display()
        self.set_status("ready", "Ready")


    def save_mask(self, alpha=0.6):
        '''
        Save mask as a proper indexed png file and an associated png image to 
        see the identified masks.
        '''
        # Check if an image is loaded
        if not self.image_is_loaded():
            return
        if self.mask_orig is None:
            return
    
        p = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG indexed", "*.png")])
        if not p:
            return
        
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



    #%% MOUSE EVENTS ----------------------------------------------------------
    def on_canvas_left(self, e):
        '''
        Handles left-clicks on the canvas, performing the active tool's action 
        (brush, magic wand, connected component, or smoothing) and using Shift 
        to modify behaviour.
        '''
        shift_pressed = (e.state & 0x0001) != 0 or self.b3_pressed
        ctrl_pressed = (e.state & 0x0004) != 0
        self._prev_brush_pos = None
        
        
        if not (self.brush_active or self.magic_mode or self.cc_mode or self.smoothing_active):
            self._pan_start = (e.x, e.y, self.view_x, self.view_y)
            return
        
        # Check position
        x_check = int((e.x)*(self.view_w/self.canvas.winfo_width())) + self.view_x
        y_check = int((e.y)*(self.view_h/self.canvas.winfo_height())) + self.view_y
        
        
        if (x_check < 0) or (x_check > self.orig_w) or (y_check < 0) or (y_check > self.orig_h):
            check_inside_image = False
        else:
            check_inside_image = True
        
        
        if self.smoothing_active and check_inside_image:
            x = int((e.x)*(self.view_w/self.canvas.winfo_width())) + self.view_x
            y = int((e.y)*(self.view_h/self.canvas.winfo_height())) + self.view_y
            op = "erosion" if shift_pressed else "dilation"
            self.apply_smoothing(y, x, operation=op)
            return
    
        if self.cc_mode and check_inside_image:
            self.connected_component_click(e, remove_only=not shift_pressed)
            return
        
        if self.magic_mode and check_inside_image:
            self.sam_add_point(e, add=not shift_pressed, multipoint=ctrl_pressed)
            return
        
        if self.brush_active and check_inside_image:
            x = int((e.x)*(self.view_w/self.canvas.winfo_width())) + self.view_x
            y = int((e.y)*(self.view_h/self.canvas.winfo_height())) + self.view_y
            self.brush_at(x, y, add=not shift_pressed)
            self.push_undo()
            self.update_display(update_all="Mask")
            self.draw_brush_preview(e)
            return

        
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
        # '''
        # ctrl_pressed = (e.state & 0x0004) != 0
        
        # # Check position
        # x_check = int((e.x)*(self.view_w/self.canvas.winfo_width())) + self.view_x
        # y_check = int((e.y)*(self.view_h/self.canvas.winfo_height())) + self.view_y
        
        
        # if (x_check < 0) or (x_check > self.orig_w) or (y_check < 0) or (y_check > self.orig_h):
        #     check_inside_image = False
        # else:
        #     check_inside_image = True
            
        # if self.smoothing_active and check_inside_image:
        #     x = int((e.x)*(self.view_w/self.canvas.winfo_width())) + self.view_x
        #     y = int((e.y)*(self.view_h/self.canvas.winfo_height())) + self.view_y
        #     self.apply_smoothing(y, x, operation="erosion")
        #     return
    
        # if self.cc_mode and check_inside_image:
        #     self.connected_component_click(e, remove_only=False)
        #     return
        
        # if self.magic_mode and check_inside_image:
        #     self.sam_add_point(e, add=False, multipoint=ctrl_pressed)
        #     return
        
        # if self.brush_active and check_inside_image:
        #     self.brush(e, add=False)
        #     return
        # '''
    def on_canvas_right_release(self, e):
        self.b3_pressed = False
        self.on_canvas_left_release(e)

    def on_canvas_drag(self, e):
        '''
        Updates the brush continuously while dragging the mouse.
        Draws one circle per event, using add/subtract depending on Shift.
        No interpolation between points to avoid undesired smoothing.
        '''
        shift_pressed = (e.state & 0x0001) != 0 or self.b3_pressed
        
        # Move the canvas if not tools selected
        if not (self.brush_active or self.magic_mode or self.cc_mode or self.smoothing_active):
            
            if self._pan_start is not None:
                x0, y0, ox0, oy0 = self._pan_start
                self.view_x = ox0 -int((e.x - x0)*(self.view_w/self.canvas.winfo_width()))
                self.view_y = oy0- int((e.y - y0)*(self.view_h/self.canvas.winfo_height()))
                self.update_display()
            return
        
        # Check if the brush is not active (only draggable tool)
        if not self.brush_active:
            return
        
        # Define the brush drag
        x1 = int((e.x)*(self.view_w/self.canvas.winfo_width())) + self.view_x
        y1 = int((e.y)*(self.view_h/self.canvas.winfo_height())) + self.view_y
        
        if not hasattr(self, "_prev_brush_pos") or self._prev_brush_pos is None:
            self._prev_brush_pos = (x1, y1)
            self.push_undo() # TODO: Check problem for undo
            self.brush_at(x1, y1, add=not shift_pressed)
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
                self.brush_at(xi, yi, add=not shift_pressed)

            self.update_display(update_all="Mask") # update only mask
            self.draw_brush_preview(e)
            self._last_brush_update = now
            self._prev_brush_pos = (x1, y1)

    
    #%% PAN & ZOOM-------------------------------------------------------------
    
    def pan_view(self, dx, dy):
        '''
        Pan view when distance is fixed. Used e.g. to bind keyboard arrows
        '''
        if not self.image_is_loaded():
            return
        self.view_x += dx
        self.view_y += dy
        self.update_display()
    
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
        # TODO update also self.view_x, self.view_y to focus the zoom
        #self.view_h = int(min(self.canvas.winfo_height(), self.orig_h)/self.zoom)
        #self.view_w = int(min(self.canvas.winfo_width(), self.orig_w)/self.zoom)
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


    #%% MASKING TECHNIQUES ----------------------------------------------------
    # BRUSH -------------------------------------------------------------------
    def brush(self, e, add=True):
        '''
        Paints or erases a circular area on the active mask at the mouse 
        position, saving the previous state for undo and updating the display.
        '''
        if self.mask_orig is None or self.active_mask_id is None:
            return
        self.push_undo()
    
        # Mouse position in image coordinates
        x = int((e.x)*(self.view_w/self.canvas.winfo_width())) + self.view_x
        y = int((e.y)*(self.view_h/self.canvas.winfo_height())) + self.view_y
        
        r = self.brush_size // 2
    
        # Extend bounding box slightly to avoid gaps
        buffer = max(1, r // 2)
        y0, y1 = max(0, y - r - buffer), min(self.mask_orig.shape[0], y + r + buffer)
        x0, x1 = max(0, x - r - buffer), min(self.mask_orig.shape[1], x + r + buffer)
    
        # Create circular mask
        yy, xx = np.ogrid[y0:y1, x0:x1]
        circle = (yy - y)**2 + (xx - x)**2 <= r*r
    
        if add:
            if self.only_on_empty.get():
                mask_area = self.mask_orig[y0:y1, x0:x1]
                mask_area[circle & (mask_area==0)] = self.active_mask_id
            else:
                self.mask_orig[y0:y1, x0:x1][circle] = self.active_mask_id
        else:
            erase_mask = circle & (self.mask_orig[y0:y1, x0:x1] == self.active_mask_id)
            self.mask_orig[y0:y1, x0:x1][erase_mask] = 0
        
        self.set_modified(True)
        self.update_display(update_all="Mask")

    # TODO unify brush and brush_at methods?
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
        dy = ys[:, None] - y   # shape (height, 1)
        dx = xs[None, :] - x   # shape (1, width)
        circle = dx**2 + dy**2 <= r*r  # boolean array, shape (y1-y0, x1-x0)
        
        # Slice of the mask corresponding to the bounding box
        mask_area = self.mask_orig[y0:y1, x0:x1]
        
        if add:
            if self.only_on_empty.get():
                # Paint only on pixels that are currently empty (0)
                mask_area[circle & (mask_area == 0)] = self.active_mask_id
            else:
                # Paint over all pixels in the brush
                mask_area[circle] = self.active_mask_id
        else:
            # Erase only pixels that match the active mask label
            erase_mask = circle & (mask_area == self.active_mask_id)
            mask_area[erase_mask] = 0
        
        # Mark mask as modified for later saving or GUI update
        self.set_modified(True)


    def draw_brush_preview(self, e):
        '''
        Draws a semi-transparent circle on the canvas to show the brush size
        and position before painting. The circle is solid in 'add mask' mode
        and dashed in 'remove mask' mode.
        '''
        # self.mouse = {
        #     'x': e.x, 
        #     'y': e.y
        # }
        self.mouse['x'], self.mouse['y'] = e.x, e.y # store mouse position
        x, y = e.x, e.y

        shift_pressed = (e.state & 0x0001) != 0 or self.b3_pressed
        self._draw_brush_preview(x, y, shift_pressed)

    def _draw_brush_preview(self, x, y, shift_pressed=False):
        self.canvas.delete("brush")
        
        if not self.brush_active:
            return

        r = int(self.brush_size * self.zoom / 2)
        outline_color = "#" + "".join([f"{c:02x}" for c in self.mask_colors[self.active_mask_id]])
        dash = (5,10) if shift_pressed else None

        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="", outline=outline_color, dash=dash, width=2, tag="brush")
        
    # SAM ---------------------------------------------------------------------
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
        if not multipoint:
            self.sam_pt_labels.append(1)
            self.sam_apply(add=add, multipoint=multipoint)
        else:
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

    
    def sam_apply_release(self):
        """
        Event bound to the release of the "Multipoint" key
        """
        if (not self.magic_mode) or (not self.sam_points): # empty lists are false
            return
        self.sam_apply(multipoint=True)

    
    def sam_apply(self, add=True, multipoint=False):
        """
        Use the SAM to generate a mask depending on the information stored in
        self.sam_points and self.sam_pt_labels.
        
        If add is True, the mask will be added; otherwise, it will be removed.
        If multipoint is True, add is ignored and the mask will be added.
        """
        if self.image_orig is None or self.active_mask_id is None:
            return
        self.set_status("loading", "SAM computing...")
        self.push_undo()
        masks, _, _ = self.sam.predict(np.array(self.sam_points),
                                       np.array(self.sam_pt_labels),
                                       multimask_output=False)
        
        if not multipoint:
            if add:
                if self.only_on_empty.get():
                    self.mask_orig[masks[0] & (self.mask_orig==0)] = self.active_mask_id
                else:
                    self.mask_orig[masks[0]] = self.active_mask_id
            else:
                self.mask_orig[masks[0] & (self.mask_orig==self.active_mask_id)] = 0
        else:
            if self.only_on_empty.get():
                self.mask_orig[masks[0] & (self.mask_orig==0)] = self.active_mask_id
            else:
                self.mask_orig[masks[0]] = self.active_mask_id

        self.sam_points = []
        self.sam_pt_labels = []
        self.set_modified(True)
        self.canvas.delete("sam_pt")
        self.update_display(update_all="Mask")
        self.set_status("ready", "Ready")


    # CONNECTED COMPONENT -----------------------------------------------------
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
        if self.mask_orig is None or self.active_mask_id is None: return
        x = int((e.x)*(self.view_w/self.canvas.winfo_width())) + self.view_x
        y = int((e.y)*(self.view_h/self.canvas.winfo_height())) + self.view_y
        
        comp = self.get_connected_component(self.mask_orig, y, x, self.active_mask_id)
        self.push_undo()
        
        if remove_only:
            self.mask_orig[comp] = 0
        else:
            if self.only_on_empty.get():
                self.mask_orig[(self.mask_orig==0) & comp] = self.active_mask_id
            else:
                self.mask_orig[(self.mask_orig==self.active_mask_id) & (~comp)]=0
        
        self.set_modified(True)
        self.update_display(update_all="Mask")
        

    # SMOOTHING (EROSION + DILATION) ------------------------------------------
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
    
        if operation == "dilation":
            comp_smooth = binary_dilation(comp, structure=struct, iterations=1)
        elif operation == "erosion":
            comp_smooth = binary_erosion(comp, structure=struct, iterations=1)
        else:
            return
        
        self.mask_orig[comp] = 0
        if self.only_on_empty.get():
            self.mask_orig[comp_smooth & (self.mask_orig==0)] = self.active_mask_id
        else:
            self.mask_orig[comp_smooth] = self.active_mask_id
        
        self.set_modified(True)
        self.update_display(update_all="Mask")
        self.set_status("ready", "Ready")

#%% Main cycle
if __name__ == "__main__":

    SegmentationApp().mainloop()
