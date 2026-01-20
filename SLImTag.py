'''
SLImTAG gui. Version 1.0

01/20/2026

Giulio Del Corso & Oscar Papini
'''


#%% Libraries
import os                      

# Numerical arrays manipulation
import numpy as np             
from scipy.ndimage import binary_dilation, binary_erosion  

# Image manipulation and TkIntert (ImageTk) 
from PIL import Image, ImageTk  

# TkInter and CustomTkInter GUI
import tkinter as tk                 
from tkinter import filedialog, ttk, simpledialog
import customtkinter as ctk    

# Torch and SAM (Segment anything model)
import torch                    
from segment_anything import sam_model_registry, SamPredictor  



#%% Suppress specific PyTorch warnings
import warnings

warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`"
)



#%% User selected parameters
MAX_DISPLAY = 800   # Maximum display size for resizing images
UNDO_DEPTH = 10     # Maximum number of undo steps

# predefined high contrast colors for masks
MAX_MASKS = 20 
HIGH_CONTRAST_COLORS = [
    (255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),
    (0,255,255),(255,128,0),(128,0,255),(0,255,128),
    (128,255,0),(255,0,128),(0,128,255),(128,128,0),
    (128,0,0),(0,128,0),(0,0,128),(200,200,200),
    (255,200,200),(200,255,200),(200,200,255)
]

# colors for different tool states
TOOL_OFF_COLOR = "#3A3A3A"   # neutral grey when tool is off
BRUSH_ON_COLOR = "#4CAF50"   # green
MAGIC_ON_COLOR = "#FF9800"   # orange
CC_ON_COLOR = "#9C27B0"      # purple
SMOOTH_ON_COLOR = "#2196F3"  # blue

# SAM parameters
MODEL_TYPE = "vit_b"
MODEL_WEIGHTS_PATH = "models/sam_vit_b_01ec64.pth"



#%% CTK parameters
ctk.set_appearance_mode("System")   # System theme
ctk.set_default_color_theme("blue") # CTK color theme




#%% Main class
class SegmentationApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("EasyTAG")
        self.geometry("1300x900")
        
        self.menu_bar = tk.Menu(self)
        self.config(menu=self.menu_bar)
        
        # Menu File (top menu)
        file_menu = tk.Menu(self.menu_bar, tearoff=0)
        file_menu.add_command(label="Import Image", command=self.load_image)
        file_menu.add_command(label="Load Mask", command=self.load_mask)
        file_menu.add_command(label="Save Mask", command=self.save_mask)
        file_menu.add_separator()
        file_menu.add_command(label="Clear Mask", 
                                                command=self.clear_active_mask)
        self.menu_bar.add_cascade(label="File", menu=file_menu)


        # STATE ---------------------------------------------------------------
        self.image_orig = None
        self.mask_orig = None
        self.mask_disp = None
        self.zoom = 1.0
        self.offset_x = 0
        self.offset_y = 0

        self.mask_labels = {}
        self.mask_colors = {}
        self.active_mask_id = None
 
        self.brush_active = False
        self.magic_mode = False
        self.cc_mode = False 
        self.smoothing_active = False  # nuovo stato
        
        self.last_brush_pos = None

        self.brush_size = 30
        self.undo_stack = []
        

        # UI ------------------------------------------------------------------
        # Main canvas for image + masks
        self.canvas = ctk.CTkCanvas(self, bg="black")  
        self.canvas.pack(side="left", fill="both", expand=True)

        # Right panel for controls
        self.panel = ctk.CTkFrame(self, width=250)    
        self.panel.pack(side="right", fill="y")


        # Mask controls
        # self.new_label = ctk.CTkEntry(self.panel, placeholder_text="Mask name")
        # self.new_label.pack(padx=10, pady=5, fill="x")
        ctk.CTkButton(self.panel, text="Add new mask", 
                                 command=self.add_mask).pack(fill="x", padx=10)
        
        # Color preview for currently selected mask
        self.color_preview = tk.Canvas(self.panel, width=40, height=20, 
                                              bg="black", highlightthickness=1)
        self.color_preview.pack(padx=10, pady=(0,5), fill="x")
         
        # Combobox to select active mask
        self.combo = ttk.Combobox(self.panel, state="readonly")
        self.combo.pack(padx=10, pady=5, fill="x")
        self.combo.bind("<<ComboboxSelected>>", self.change_mask)
         
        # Spacer after combobox
        self.panel_spacer = ctk.CTkFrame(self.panel, height=20, 
                                                        fg_color="transparent")
        self.panel_spacer.pack()


        # Undo button
        ctk.CTkButton(self.panel, text="Undo [Z]", command=self.undo).pack(
                                                     fill="x", padx=10, pady=5)


        # Tool buttons
        self.brush_btn = ctk.CTkButton(self.panel, text="Brush [B]", 
                                                     command=self.toggle_brush)
        self.brush_btn.pack(fill="x", padx=10, pady=5)
        self.brush_slider = ctk.CTkSlider(self.panel, from_=5, to=100, 
                           command=lambda v: setattr(self,"brush_size",int(v)))
        self.brush_slider.set(self.brush_size)
        self.brush_slider.pack(fill="x", padx=10)

        self.magic_btn = ctk.CTkButton(self.panel, text="Magic Wand [M]", 
                                                     command=self.toggle_magic)
        self.magic_btn.pack(fill="x", padx=10, pady=5)

        self.smoothing_btn = ctk.CTkButton(self.panel, text="Smoothing [S]", 
                                                 command=self.toggle_smoothing)
        self.smoothing_btn.pack(fill="x", padx=10, pady=5)

        self.brush_btn.configure(fg_color=TOOL_OFF_COLOR)
        self.magic_btn.configure(fg_color=TOOL_OFF_COLOR)
        self.smoothing_btn.configure(fg_color=TOOL_OFF_COLOR)
        self.cc_btn = ctk.CTkButton(self.panel, text="Connected Component [C]", 
                                                   command=self.toggle_cc_mode)
        self.cc_btn.pack(fill="x", padx=10)
        self.cc_btn.configure(fg_color=TOOL_OFF_COLOR)


        # SAM -----------------------------------------------------------------
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[MODEL_TYPE](checkpoint=MODEL_WEIGHTS_PATH)
        sam.to(device).eval()
        self.sam = SamPredictor(sam)
        

        # BINDINGS ------------------------------------------------------------
        self.canvas.bind("<MouseWheel>", self.zoom_evt)
        self.canvas.bind("<Motion>", self.draw_brush_preview)
        self.canvas.bind("<Button-1>", self.on_canvas_left)
        self.canvas.bind("<Button-3>", self.on_canvas_right)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_left_release)
        
        # Zoom via keyboard (Ctrl + / Ctrl -)
        self.bind("<Control-plus>", self.zoom_in_keyboard)
        self.bind("<Control-equal>", self.zoom_in_keyboard)  
        self.bind("<Control-minus>", self.zoom_out_keyboard)
        
        
        # SHORTCUT KEYS -------------------------------------------------------
        self.bind("<b>", lambda e: self.toggle_brush())
        self.bind("<B>", lambda e: self.toggle_brush())
        self.bind("<m>", lambda e: self.toggle_magic())
        self.bind("<M>", lambda e: self.toggle_magic())
        self.bind("<c>", lambda e: self.toggle_cc_mode())
        self.bind("<C>", lambda e: self.toggle_cc_mode())
        self.bind("<s>", lambda e: self.toggle_smoothing())
        self.bind("<S>", lambda e: self.toggle_smoothing())
        self.bind("<z>", lambda e: self.undo()) 
        self.bind("<Z>", lambda e: self.undo()) 
        
        

    #%% AUX METHODS  ----------------------------------------------------------
    def update_display(self):
        '''
        Aux method to update display whenever a change occurs.
        '''
        if self.image_orig is None: return
        w,h = self.image_orig.size
        scale = min(MAX_DISPLAY/w, MAX_DISPLAY/h) * self.zoom
        self.display_scale = scale
        disp = self.image_orig.resize((int(w*scale),int(h*scale)))
        self.mask_disp = np.array(Image.fromarray(self.mask_orig).resize(
                                                     disp.size, Image.NEAREST))

        self.tk_img = ImageTk.PhotoImage(disp)
        self.canvas.delete("all")
        self.canvas.create_image(self.offset_x,self.offset_y,anchor="nw",
                                                             image=self.tk_img)

        # overlay
        overlay = np.zeros((*self.mask_disp.shape,4),np.uint8)
        for mid,c in self.mask_colors.items():
            overlay[self.mask_disp==mid] = [*c,150]
        self.tk_ov = ImageTk.PhotoImage(Image.fromarray(overlay))
        self.canvas.create_image(self.offset_x,self.offset_y,anchor="nw",
                                                              image=self.tk_ov)
        self.update_mask_color_preview()
        
        
    def update_mask_color_preview(self):
        '''
        Update color preview of active mask
        '''
        
        self.color_preview.delete("all")
        if self.active_mask_id is None:
            return
        c = self.mask_colors.get(self.active_mask_id, (0,0,0))
        hex_color = "#%02x%02x%02x" % c
        self.color_preview.create_rectangle(0, 0, 40, 20, fill=hex_color, 
                                                                    outline="")
        
        
    def update_button_colors(self):
        '''
        Update button colors based on active tool
        '''
        self.brush_btn.configure(fg_color=BRUSH_ON_COLOR if 
                                         self.brush_active else TOOL_OFF_COLOR)
        self.magic_btn.configure(fg_color=MAGIC_ON_COLOR if 
                                           self.magic_mode else TOOL_OFF_COLOR)
        self.cc_btn.configure(fg_color=CC_ON_COLOR if 
                                              self.cc_mode else TOOL_OFF_COLOR)
        self.smoothing_btn.configure(fg_color=SMOOTH_ON_COLOR if 
                                     self.smoothing_active else TOOL_OFF_COLOR)
        
        
        
    def deactivate_tools(self):
        '''
        Keep one tool button active at time
        '''
        self.brush_active = False
        self.magic_mode = False
        self.cc_mode = False
        self.smoothing_active = False
        self.update_button_colors()


    
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
        if self.undo_stack:
            self.mask_orig = self.undo_stack.pop()
            self.update_display()


    # MASK MANAGEMENT ---------------------------------------------------------
    def add_mask(self):
        '''
        Creates a new mask, asks the user for a name via dialog, assigns it a 
        unique ID and a color, and updates the UI accordingly.
        '''
        self.deactivate_tools()
        if len(self.mask_labels) >= MAX_MASKS: 
            return
        
        # Ask user for mask name
        name = simpledialog.askstring("New Mask", 
                                                "Enter name for the new mask:")
        if not name:  # User cancelled or empty
            return
        
        mid = max(self.mask_labels.keys(), default=0) + 1
        self.mask_labels[mid] = name
        self.mask_colors[mid] = HIGH_CONTRAST_COLORS[mid-1]
        self.active_mask_id = mid
        self.combo["values"] = [f"{i}:{l}" for i,l in self.mask_labels.items()]
        self.combo.current(len(self.mask_labels)-1)
        self.update_mask_color_preview()



    def change_mask(self,e):
        '''
        Changes the currently active mask based on the user selection in the 
        combo box.
        '''
        # Retrieves the mask ID corresponding to the current selection 
        self.active_mask_id \
                          = list(self.mask_labels.keys())[self.combo.current()]
        # Updates the UI buttons’ colors 
        self.update_button_colors()
        # Updates the small color preview rectangle
        self.update_mask_color_preview()


    # BUTTON TOGGLES ----------------------------------------------------------
    def toggle_brush(self):
        '''
        Activates or deactivates the brush tool and updates the button colors 
        and mask preview.
        '''
        if not self.brush_active:
            self.deactivate_tools()
            self.brush_active = True
        else:
            self.brush_active = False
        self.update_button_colors()
        self.update_mask_color_preview()


    def toggle_cc_mode(self):
        '''
        Activates or deactivates the connected component tool and refreshes the 
        UI to reflect its state.
        '''
        if not self.cc_mode:
            self.deactivate_tools()
            self.cc_mode = True
        else:
            self.cc_mode = False
        self.update_button_colors()
        self.update_mask_color_preview()


    def toggle_magic(self):
        '''
        Activates or deactivates the magic wand tool and updates the button 
        colors and mask preview accordingly.
        '''
        if not self.magic_mode:
            self.deactivate_tools()
            self.magic_mode = True
        else:
            self.magic_mode = False
        self.update_button_colors()
        self.update_mask_color_preview()


    def toggle_smoothing(self):
        '''
        Activates or deactivates the smoothing tool and updates the interface 
        to show its current status.
        '''
        if not self.smoothing_active:
            self.deactivate_tools()
            self.smoothing_active = True
        else:
            self.smoothing_active = False
        self.update_button_colors()
        self.update_mask_color_preview()



    #%% IMAGE AND MASK --------------------------------------------------------
    def load_image(self):
        '''
        Load a .png or .jpg image and define and empty mask on it
        '''
        self.deactivate_tools()
        p = filedialog.askopenfilename()
        if not p: return
        self.image_orig = Image.open(p).convert("RGB")
        self.mask_orig = np.zeros(self.image_orig.size[::-1], np.uint8)
        self.sam.set_image(np.array(self.image_orig))
          
        self.update_display()
        

    def load_mask(self):
        '''
        Upload an existing mask (png or numpy). It can handles arbitrary pngs
        with a maximum of MAX_MASKS non zero values. All subsequent values are 
        dropped. 
        '''
        self.deactivate_tools()
        p = filedialog.askopenfilename(filetypes=[("NumPy or PNG", "*.npy *.png")])
        if not p:
            return
    
        self.push_undo()
    
        ext = p.split(".")[-1].lower()
        self.mask_labels.clear()
        self.mask_colors.clear()
    
        if ext == "npy":
            mask = np.load(p)
            self.mask_orig = mask.copy()
            labels = np.unique(mask)
            labels = labels[labels != 0]
            for i, l in enumerate(labels[:20], 1):
                self.mask_labels[l] = f"mask_{i}"
                self.mask_colors[l] = HIGH_CONTRAST_COLORS[i-1]
            self.active_mask_id = labels[0] if len(labels) > 0 else None
    
        elif ext == "png":
            img = Image.open(p).convert("RGB")
            arr = np.array(img)
            h, w, _ = arr.shape
    
            # Identifies unique colors
            arr_flat = arr.reshape(-1, 3)
            arr_flat_nonblack = arr_flat[~np.all(arr_flat == 0, axis=1)]
            if len(arr_flat_nonblack) == 0:
                self.mask_orig = np.zeros((h, w), np.uint8)
                return
    
            # Select the first MAX_MASKS
            unique_colors = []
            seen = set()
            for color in arr_flat_nonblack:
                t = tuple(color)
                if t not in seen:
                    seen.add(t)
                    unique_colors.append(t)
                    if len(unique_colors) >= 20:
                        break
    
            mask = np.zeros((h, w), np.uint8)
            for i, color in enumerate(unique_colors, 1):
                mask[np.all(arr == color, axis=-1)] = i
                self.mask_labels[i] = f"mask_{i}"
                self.mask_colors[i] = color
    
            self.mask_orig = mask
            self.active_mask_id = 1 if unique_colors else None
    
        # Update combo box
        values = [f"{i}:{l}" for i, l in self.mask_labels.items()]
        self.combo["values"] = values
        if self.mask_labels:
            self.combo.current(0)
        else:
            self.combo.set('')
            
        # Dectivate all the buttons
        self.switch_state = None
    
        self.update_display()
        self.update_mask_color_preview()

    
    def save_mask(self, alpha=0.6):
        '''
        Save mask as a proper .npy file and an associated png image to see the
        identified masks.
        alpha=0.6 float [0,1] transparecy of the saved mask
        '''
        self.deactivate_tools()
        
        if self.mask_orig is None:
            return
    
        # .npy save
        p = filedialog.asksaveasfilename(defaultextension=".npy")
        if not p:
            return
        np.save(p, self.mask_orig)
    
        if self.image_orig is None:
            return
    
        # Transparent RGBA overlay
        h, w = self.mask_orig.shape
        overlay = np.zeros((h, w, 4), dtype=np.uint8)  # RGBA
    
        for mid, color in self.mask_colors.items():
            mask_idx = self.mask_orig == mid
            overlay[mask_idx, :3] = color
            overlay[mask_idx, 3] = int(255 * alpha)  # Alpha channeò
    
        overlay_img = Image.fromarray(overlay, mode="RGBA")
    
        bg = self.image_orig.convert("RGBA")
        combined = Image.alpha_composite(bg, overlay_img)
    
        # PNG save
        png_path = os.path.splitext(p)[0] + ".png"
        combined.save(png_path)
        


    #%% MOUSE EVENTS ----------------------------------------------------------
    def on_canvas_left(self, e):
        '''
        Handles left-clicks on the canvas, performing the active tool’s action 
        (brush, magic wand, connected component, or smoothing) and using Shift 
        to modify behavior.
        '''
        shift_pressed = (e.state & 0x0001) != 0  
        
        self._prev_brush_pos = None
        
        if self.smoothing_active:
            y = int((e.y - self.offset_y)/self.display_scale)
            x = int((e.x - self.offset_x)/self.display_scale)
            op = "erosion" if shift_pressed else "dilation"
            self.apply_smoothing(y, x, operation=op)
            return "break"
        
        if self.cc_mode:
            self.connected_component_click(e, remove_only=not shift_pressed)
            return "break"
        if self.magic_mode:
            self.sam_click(e, add=not shift_pressed)
            return "break"
        if self.brush_active:
            self.brush_at(int((e.x - self.offset_x)/self.display_scale),
                          int((e.y - self.offset_y)/self.display_scale),
                          add=not shift_pressed)
            self.push_undo()  
            self.update_display()
            return "break"
        
        
    def on_canvas_left_release(self, e):
        self.last_brush_pos = None

    
    def on_canvas_right(self, e):
        '''
        Handles right-clicks on the canvas, applying the active tool’s removal 
        or erosion action without toggling tools.
        '''
        if self.smoothing_active:
            y = int((e.y - self.offset_y)/self.display_scale)
            x = int((e.x - self.offset_x)/self.display_scale)
            self.apply_smoothing(y, x, operation="erosion")
            return "break"
    
        if self.cc_mode:
            self.connected_component_click(e, remove_only=False)
            return "break"
        if self.magic_mode:
            self.sam_click(e, add=False)
            return "break"
        if self.brush_active:
            self.brush(e, add=False)
            return "break"


    def on_canvas_drag(self, e):
        '''
        Updates the brush continuously while dragging the mouse.
        Draws one circle per event, using add/subtract depending on Shift.
        No interpolation between points to avoid undesired smoothing.
        '''
        if not self.brush_active or self.cc_mode:
            return
        
        shift_pressed = (e.state & 0x0001) != 0
        
        x1 = int((e.x - self.offset_x) / self.display_scale)
        y1 = int((e.y - self.offset_y) / self.display_scale)
        
        if not hasattr(self, "_prev_brush_pos") or self._prev_brush_pos is None:
            self._prev_brush_pos = (x1, y1)
            self.push_undo()  
            self.brush_at(x1, y1, add=not shift_pressed)
            self.update_display()
            return
        
        x0, y0 = self._prev_brush_pos
        dx = x1 - x0
        dy = y1 - y0
        dist = max(1, int(np.hypot(dx, dy)))
        
        r = max(1, self.brush_size // 2)
        step = max(1, r // 5)
        
        for i in range(0, dist + 1, step):
            xi = int(x0 + dx * i / dist)
            yi = int(y0 + dy * i / dist)
            self.brush_at(xi, yi, add=not shift_pressed)
        
        self.update_display()
        self._prev_brush_pos = (x1, y1)


    def brush_at(self, x, y, add=True):
        '''
        Aux method to define brush position without updating display or undo.
        '''
        if self.mask_orig is None or self.active_mask_id is None:
                return
        
        r = self.brush_size // 2
        buffer = max(1, r // 2)
    
        y0, y1 = max(0, y - r - buffer), min(self.mask_orig.shape[0], y + r + buffer)
        x0, x1 = max(0, x - r - buffer), min(self.mask_orig.shape[1], x + r + buffer)
    
        yy, xx = np.ogrid[y0:y1, x0:x1]
        circle = (yy - y)**2 + (xx - x)**2 <= r*r
    
        if add:
            self.mask_orig[y0:y1, x0:x1][circle] = self.active_mask_id
        else:
            erase_mask = circle & (self.mask_orig[y0:y1, x0:x1] == self.active_mask_id)
            self.mask_orig[y0:y1, x0:x1][erase_mask] = 0


    def zoom_evt(self,e):
        '''
        Adjusts the zoom level of the displayed image based on mouse wheel 
        input and refreshes the display.
        '''
        self.zoom *= 1.1 if e.delta>0 else 0.9
        self.update_display()
        
        
    def zoom_in_keyboard(self, e=None):
        '''
        Adjust zoom level using keyboard (zoom in).
        '''
        self.zoom *= 1.1
        self.update_display()


    def zoom_out_keyboard(self, e=None):
        '''
        Adjust zoom level using keyboard (zoom in).
        '''
        self.zoom *= 0.9
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
        x = int((e.x - self.offset_x) / self.display_scale)
        y = int((e.y - self.offset_y) / self.display_scale)
        r = self.brush_size // 2
    
        # Extend bounding box slightly to avoid gaps
        buffer = max(1, r // 2)
        y0, y1 = max(0, y - r - buffer), min(self.mask_orig.shape[0], y + r + buffer)
        x0, x1 = max(0, x - r - buffer), min(self.mask_orig.shape[1], x + r + buffer)
    
        # Create circular mask
        yy, xx = np.ogrid[y0:y1, x0:x1]
        circle = (yy - y)**2 + (xx - x)**2 <= r*r
    
        if add:
            self.mask_orig[y0:y1, x0:x1][circle] = self.active_mask_id
        else:
            erase_mask = circle & (self.mask_orig[y0:y1, x0:x1] == self.active_mask_id)
            self.mask_orig[y0:y1, x0:x1][erase_mask] = 0
    
        self.update_display()    


    def draw_brush_preview(self,e):
        '''
        Draws a semi-transparent yellow circle on the canvas to show the brush 
        size and position before painting.
        '''
        self.canvas.delete("brush")
        if not self.brush_active: return
        x, y = e.x, e.y
        r = self.brush_size*self.display_scale/2
        self.canvas.create_oval(x-r, y-r, x+r, y+r,
                                fill="#FFFF00", outline="#FFFF00",
                                stipple="gray25", width=1, tag="brush")
        
    
    # SAM ---------------------------------------------------------------------
    def sam_click(self,e, add=True):
        '''
        Uses the SAM model to generate a mask at the clicked point and either 
        adds it to or removes it from the active mask, saving the previous 
        state for undo and updating the display.
        '''
        if self.image_orig is None or self.active_mask_id is None: return
        self.push_undo()
        x = int((e.x-self.offset_x)/self.display_scale)
        y = int((e.y-self.offset_y)/self.display_scale)
        masks,_,_ = self.sam.predict(np.array([[x,y]]), np.array([1]), 
                                                        multimask_output=False)
        if add:
            self.mask_orig[masks[0]] = self.active_mask_id
        else:
            self.mask_orig[masks[0] & (self.mask_orig==self.active_mask_id)] = 0
        self.update_display()


    # CLEAR MASK --------------------------------------------------------------
    def clear_active_mask(self):
        '''
        Deletes all pixels of the currently active mask, removes its label and 
        color, updates the combo box to reflect remaining masks, and refreshes 
        the display and color preview.
        '''
        self.deactivate_tools()
        if self.active_mask_id is None or self.mask_orig is None: return
        self.push_undo()
        self.mask_orig[self.mask_orig == self.active_mask_id] = 0
        del self.mask_labels[self.active_mask_id]
        del self.mask_colors[self.active_mask_id]

        values = [f"{i}:{l}" for i,l in self.mask_labels.items()]
        self.combo["values"] = values
        if self.mask_labels:
            self.active_mask_id = list(self.mask_labels.keys())[0]
            self.combo.current(0)
        else:
            self.active_mask_id = None
            self.combo.set('')
        self.update_display()
        self.update_mask_color_preview()
        

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
        y = int((e.y - self.offset_y)/self.display_scale)
        x = int((e.x - self.offset_x)/self.display_scale)
        comp = self.get_connected_component(self.mask_orig, y, x, 
                                                           self.active_mask_id)
        self.push_undo()
        if remove_only:
            self.mask_orig[comp] = 0
        else:
            self.mask_orig[(self.mask_orig==self.active_mask_id) & (~comp)] = 0
        self.update_display()
        

    # SMOOTHING (EROSION + DILATION) ------------------------------------------
    def apply_smoothing(self, y, x, operation="dilation", size=3):
        '''
        Applies dilation or erosion to the connected component under the 
        clicked point, saving the previous state for undo, updating the mask 
        with the smoothed result, and refreshing the display.
        '''
        if self.mask_orig is None or self.active_mask_id is None:
            return
    
        # Identify the connected component
        comp = self.get_connected_component(self.mask_orig, y, x, 
                                                           self.active_mask_id)
        if not comp.any():
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
        self.mask_orig[comp_smooth] = self.active_mask_id
        self.update_display()



#%% Main cycle
if __name__ == "__main__":
    SegmentationApp().mainloop()