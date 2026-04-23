import customtkinter as ctk
import tkinter as tk

from PIL import Image, ImageDraw, ImageTk
import os
import math
import re

import numpy as np

from slimtag_color_utils import rgb_to_hex, hex_to_rgb, rgb_to_hsv, hsv_to_rgb

class MultiButtonDialog(ctk.CTkToplevel):
    """
    Class for dialog window with a message and multiple customizable buttons.
    
    The dialog window waits for the user to press a button, then closes.
    
    Usage:
        window = MultiButtonDialog(parent, message="Are you OK?",
                                   buttons=[("Yes", 0), ("No", 1)])
        if window.return_value == 0:
            # do something if user pressed "Yes"
        else:
            # do something if user pressed "No"
    
    buttons is a list of pairs ("Message", value) so that the button displays
    "Message", and sets self.return_value to value when pressed
    """
    def __init__(self, parent, message="", buttons=[("Cancel", None)]):
        super().__init__(parent)

        self.parent = parent
        self.return_value = None

        self.title("SLImTAG")
        self.resizable(False, False)
        self.maxsize = 500
       
        n_buttons = len(buttons)
        n_btn_rows = ((n_buttons - 1) // 3) + 1 # number of button rows

        # Message
        ctk.CTkLabel(self, text=message, wraplength=self.maxsize, justify="left").grid(row=0, column=0, sticky="nsew", padx=20, pady=(20, 10))

        # Buttons frame
        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=(5, 20))

        # Buttons rows
        button_row = {}
        for i in range(n_btn_rows):
            button_row[i] = ctk.CTkFrame(button_frame, fg_color="transparent")
            button_row[i].grid(row=i, column=0,  pady=5)
            if i < n_btn_rows-1 or n_buttons % 3 == 0:
                button_row[i].grid_columnconfigure((0,1,2), weight=1)
            else: # just last line, if it does NOT have three buttons
                button_row[i].grid_columnconfigure((0,1,2), weight=0)
        
        # Save buttons for reference
        self.buttons = {}
        for i in range(n_buttons):
            text, value = buttons[i]
            btn = ctk.CTkButton(button_row[i//3], text=text, command=lambda v=value: self._on_button(v))
            btn.grid(row=0, column=i%3, sticky= "ew" if i//3 < n_btn_rows-1 else "", padx=5)
            self.buttons[i] = btn
        
        self.grid_columnconfigure(0, weight=1)
        button_frame.grid_columnconfigure(0, weight=1)

        # Handle window close (X button)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # If there is only one button, bind <Enter> to it
        if n_buttons == 1:
            self.bind("<Return>", lambda e: self.buttons[0].invoke())

        # Determine window size
        self.update_idletasks()
        self.minsize(min(self.winfo_width(), self.maxsize), self.winfo_height())
        
        # Put dialog on top of parent
        self.transient(parent)
        # Grab events
        self.grab_set()
        
        # Center on parent
        self._center_on_parent()

        # Wait until closed
        self.wait_window(self)

    def _on_button(self, value):
        self.return_value = value
        self.destroy()

    def _on_close(self):
        self.return_value = None
        self.destroy()

    def _center_on_parent(self):
        self.update_idletasks()
        px = self.parent.winfo_x()
        py = self.parent.winfo_y()
        pw = self.parent.winfo_width()
        ph = self.parent.winfo_height()
        w = self.winfo_width()
        h = self.winfo_height()
        x = px + (pw - w) // 2
        y = py + (ph - h) // 2
        self.geometry(f"+{x}+{y}")

class EntryDialog(ctk.CTkToplevel):
    def __init__(self, parent, message="", value=""):
        super().__init__(parent)

        self.parent = parent
        self.tk_value = tk.StringVar(self, value=value)
        self.value = None # to expose it

        self.title("SLImTAG")
        self.resizable(False, False)
       
        # Message
        input_frame = ctk.CTkFrame(self, fg_color="transparent")
        input_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=(20, 10))
        
        ctk.CTkLabel(input_frame, text=message, wraplength=300, justify="left").grid(row=0, column=0, padx=(5, 10), sticky="ew")
        entry = ctk.CTkEntry(input_frame, textvariable=self.tk_value)
        entry.grid(row=0, column=1, padx=(0, 5), sticky="ew")
        entry.icursor("end")
        
        input_frame.grid_columnconfigure(1, weight=1)

        # Buttons frame
        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=(5, 20))

        btn_ok = ctk.CTkButton(button_frame, text="OK", command=self._on_ok)
        btn_ok.grid(row=0, column=0, padx=5)
        btn_cancel = ctk.CTkButton(button_frame, text="Cancel", command=self._on_cancel)
        btn_cancel.grid(row=0, column=1, padx=5)
        
        self.grid_columnconfigure(0, weight=1)

        # Handle window close (X button)
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        
        # Binding <Enter> to OK button
        self.bind("<Return>", lambda e: btn_ok.invoke())

        # Determine window size
        self.update_idletasks()
        self.minsize(self.winfo_width(), self.winfo_height())
        
        # Put dialog on top of parent
        self.transient(parent)
        # Grab events
        self.grab_set()
        # Get focus on Entry
        self.after(100, entry.focus_set)
        
        # Center on parent
        self._center_on_parent()

        # Wait until closed
        self.wait_window(self)

    def _on_ok(self):
        self.value = self.tk_value.get()
        if self.value == "": # do not accept empty values
            self.value = None
        self.destroy()

    def _on_cancel(self):
        self.value = None
        self.destroy()

    def _center_on_parent(self):
        self.update_idletasks()
        px = self.parent.winfo_x()
        py = self.parent.winfo_y()
        pw = self.parent.winfo_width()
        ph = self.parent.winfo_height()
        w = self.winfo_width()
        h = self.winfo_height()
        x = px + (pw - w) // 2
        y = py + (ph - h) // 2
        self.geometry(f"+{x}+{y}")

# custom color picker with CTk flavor
# adapted from https://github.com/Akascape/CTkColorPicker
# with improvements by Federico Volpini and Oscar Papini
class MaskEditDialog(ctk.CTkToplevel):
    """
    Change mask color and name.
    
    Opens a new TopLevel window with a color selector. The .get() method
    recovers the selected color, as a HEX string of the form '#RRGGBB'.
    """
    def __init__(self,
                 parent,
                 title: str = "Edit mask",
                 initial_color: str = None, # initial color to be displayed, in HEX '#RRGGBB'
                 mask_name: str = None # initial mask name
                 ):
        super().__init__(parent)
        
        self.parent = parent
        # path for images
        RESOURCES_PATH = "./images/ColorPicker"
        
        self.title(title)
        # define geometry
        WIDTH = 320
        HEIGHT = WIDTH + 200
        self.maxsize(WIDTH, HEIGHT)
        self.minsize(WIDTH, HEIGHT)
        self.resizable(width=False, height=False)
        
        # dimension of color wheel canvas relative to window size
        self.wheel_dim = self._apply_window_scaling(WIDTH - 50)
        # dimension of inner SL canvas relative to wheel
        # the original wheel has a dimension of 1000 px and the width of the annulus is 100 px
        self.inner_dim = self._apply_window_scaling(int(self.wheel_dim * 0.8 / math.sqrt(2)))
        # dimension of crosshair for color selection
        self.target_dim = self._apply_window_scaling(21)
        
        # load images
        self.wheel_img = Image.open(os.path.join(RESOURCES_PATH, 'hue_wheel.png')).resize((self.wheel_dim, self.wheel_dim), Image.Resampling.LANCZOS)
        self.cross_img = Image.open(os.path.join(RESOURCES_PATH, 'crosshair.png')).resize((self.target_dim, self.target_dim), Image.Resampling.LANCZOS)

        self.wheel = ImageTk.PhotoImage(self.wheel_img)
        self.cross = ImageTk.PhotoImage(self.cross_img)
        
        # vectorized version of hsv_to_rgb (for numpy computation)
        self.np_hsv_to_rgb = np.vectorize(hsv_to_rgb)
        
        # Put dialog on top of parent
        self.transient(parent)
        self.lift()
        # Grab events
        self.grab_set()

        # configure geometry
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        
        self.protocol("WM_DELETE_WINDOW", lambda: self._update_and_close(cancel=True))
        
        # PARAMETERS
        self.return_color = None
        self.current_color = (0, 0, 0) if initial_color is None else hex_to_rgb(initial_color)
        self.mask_name = tk.StringVar()
        self.mask_name.set(mask_name)
        
        # WINDOW ELEMENTS
        # frames
        self.preview_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.preview_frame.grid(row=0, column=0, padx=20, pady=(10, 5), sticky="nsew")

        self.color_frame = ctk.CTkFrame(self)
        self.color_frame.grid(row=1, column=0, padx=20, pady=5, sticky="nsew")
        
        self.button_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.button_frame.grid(row=2, column=0, padx=20, pady=(5, 10), sticky="nsew")
        
        # preview frame elements
        self.preview_crc = ctk.CTkLabel(self.preview_frame, text="", image=self._make_circle(color=self.current_color))
        self.preview_crc.grid(row=0, column=0, padx=(0, 5))
        self.name_entry = ctk.CTkEntry(self.preview_frame, textvariable=self.mask_name)
        self.name_entry.grid(row=0, column=1, sticky="ew", padx=(5, 0))
        
        self.preview_frame.grid_columnconfigure(1, weight=1)
        
        # color frame elements
        # canvas
        canvas_bg = self._apply_appearance_mode(ctk.ThemeManager.theme["CTkFrame"]["fg_color"]) # recover appropriate color to mimic transparency
        self.color_canvas = ctk.CTkCanvas(self.color_frame, height=self.wheel_dim, width=self.wheel_dim, highlightthickness=0, bg=canvas_bg)
        self.color_canvas.grid(row=0, column=0, columnspan=7, padx=5, pady=5, sticky="new")
        
        self.color_canvas.create_image(self.wheel_dim/2, self.wheel_dim/2, image=self.wheel)
        self.inner_canvas = ctk.CTkCanvas(self.color_canvas, height=self.inner_dim, width=self.inner_dim, highlightthickness=0, bg=canvas_bg)
        self.color_canvas.create_window(self.wheel_dim/2, self.wheel_dim/2, window=self.inner_canvas)
        
        # variable entries
        self.color_vars = {"r": tk.StringVar(), "g": tk.StringVar(), "b": tk.StringVar(),
                           "h": tk.StringVar(), "s": tk.StringVar(), "v": tk.StringVar(),
                           "hex": tk.StringVar()
                           }
        self.current_color_vars = {k: "" for k in self.color_vars}
        ctk.CTkLabel(self.color_frame, text="R", anchor="w").grid(row=1, column=0, padx=5, sticky="ew")
        ctk.CTkLabel(self.color_frame, text="G", anchor="w").grid(row=2, column=0, padx=5, sticky="ew")
        ctk.CTkLabel(self.color_frame, text="B", anchor="w").grid(row=3, column=0, padx=5, sticky="ew")
        ctk.CTkLabel(self.color_frame, text="[0-255]", anchor="w").grid(row=1, column=2, padx=(5, 15), sticky="ew")
        ctk.CTkLabel(self.color_frame, text="[0-255]", anchor="w").grid(row=2, column=2, padx=(5, 15), sticky="ew")
        ctk.CTkLabel(self.color_frame, text="[0-255]", anchor="w").grid(row=3, column=2, padx=(5, 15), sticky="ew")
        ctk.CTkLabel(self.color_frame, text="H", anchor="w").grid(row=1, column=3, padx=5, sticky="ew")
        ctk.CTkLabel(self.color_frame, text="S", anchor="w").grid(row=2, column=3, padx=5, sticky="ew")
        ctk.CTkLabel(self.color_frame, text="V", anchor="w").grid(row=3, column=3, padx=5, sticky="ew")
        ctk.CTkLabel(self.color_frame, text="°", anchor="w").grid(row=1, column=5, padx=(0, 5), sticky="ew")
        ctk.CTkLabel(self.color_frame, text="%", anchor="w").grid(row=2, column=5, padx=(0, 5), sticky="ew")
        ctk.CTkLabel(self.color_frame, text="%", anchor="w").grid(row=3, column=5, padx=(0, 5), sticky="ew")
        ctk.CTkLabel(self.color_frame, text="[0-360]", anchor="e").grid(row=1, column=6, padx=5, sticky="ew")
        ctk.CTkLabel(self.color_frame, text="[0-100]", anchor="e").grid(row=2, column=6, padx=5, sticky="ew")
        ctk.CTkLabel(self.color_frame, text="[0-100]", anchor="e").grid(row=3, column=6, padx=5, sticky="ew")
        hex_frame = ctk.CTkFrame(self.color_frame, fg_color="transparent")
        hex_frame.grid(row=4, column=0, columnspan=7, padx=5, pady=(10, 5), sticky="nsew")
        ctk.CTkLabel(hex_frame, text="Hex #", anchor="e").grid(row=0, column=0, sticky="ew")
        self.color_entry = {}
        self.color_entry["r"] = ctk.CTkEntry(self.color_frame, textvariable=self.color_vars["r"])
        self.color_entry["r"].grid(row=1, column=1, sticky="ew")
        self.color_entry["g"] = ctk.CTkEntry(self.color_frame, textvariable=self.color_vars["g"])
        self.color_entry["g"].grid(row=2, column=1, sticky="ew")
        self.color_entry["b"] = ctk.CTkEntry(self.color_frame, textvariable=self.color_vars["b"])
        self.color_entry["b"].grid(row=3, column=1, sticky="ew")
        self.color_entry["h"] = ctk.CTkEntry(self.color_frame, textvariable=self.color_vars["h"])
        self.color_entry["h"].grid(row=1, column=4, sticky="ew")
        self.color_entry["s"] = ctk.CTkEntry(self.color_frame, textvariable=self.color_vars["s"])
        self.color_entry["s"].grid(row=2, column=4, sticky="ew")
        self.color_entry["v"] = ctk.CTkEntry(self.color_frame, textvariable=self.color_vars["v"])
        self.color_entry["v"].grid(row=3, column=4, sticky="ew")
        self.color_entry["hex"] = ctk.CTkEntry(hex_frame, textvariable=self.color_vars["hex"])
        self.color_entry["hex"].grid(row=0, column=1, sticky="ew")
        for k in self.color_entry:
            self.color_entry[k].bind("<KeyRelease>", lambda e, k=k: self.update_color_vars(e, keep=k))
        
        self.color_frame.grid_rowconfigure(0, weight=1)
        self.color_frame.grid_columnconfigure([1, 4], weight=1)
        hex_frame.grid_columnconfigure(1, weight=1)

        # buttons frame elements
        self.btn_ok = ctk.CTkButton(self.button_frame, text="OK", command=lambda: self._update_and_close(cancel=False))
        self.btn_ok.grid(row=0, column=0, padx=(0, 5))
        self.btn_cancel = ctk.CTkButton(self.button_frame, text="Cancel", command=lambda: self._update_and_close(cancel=True))
        self.btn_cancel.grid(row=0, column=1, padx=(5, 0))
        
        self.button_frame.grid_columnconfigure([0, 1], weight=1)
        
        # bind <Enter> to OK from everywhere
        self.bind("<Return>", lambda e: self.btn_ok.invoke())
        # bind mouse events
        self.color_canvas.bind("<Button-1>", self.on_mouse_wheel)
        self.color_canvas.bind("<B1-Motion>", self.on_mouse_wheel)
        self.inner_canvas.bind("<Button-1>", self.on_mouse_inner)
        self.inner_canvas.bind("<B1-Motion>", self.on_mouse_inner)
        
        # AFTER DEFINITIONS OF GRAPHICAL ELEMENTS
        # finish init cycle
        self.update_color_vars()
        self.update_inner_canvas()
        self.name_entry.icursor("end")
        self.after(150, lambda: self.name_entry.focus())
        self.grab_set()
    
    def _make_circle(self, color=(0, 0, 0), circle_size=21):
        # aux function that draws a circle with specified color
        color_circle = Image.new("RGBA", (circle_size+1, circle_size+1), (0, 0, 0, 0))
        color_circle_draw = ImageDraw.Draw(color_circle)
        color_circle_draw.ellipse((0, 0, circle_size, circle_size), fill=color)
        return ctk.CTkImage(color_circle, size=(circle_size+1, circle_size+1))
    
    def _update_and_close(self, cancel=False):
        """
        Update the return color and close the window.
        
        If cancel is True, the returned color is None; otherwise, the returned
        color is self.current_color
        """
        if cancel:
            self.return_color = None
            self.mask_name.set("")
        else:
            self.return_color = rgb_to_hex(self.current_color)
        self.grab_release()
        self.destroy()
        # free resources
        del self.wheel_img
        del self.cross_img
        del self.wheel
        del self.cross
    
    def get(self):
        """
        Return tuple (new mask name, new mask color)
        """
        self.parent.wait_window(self)
        return (self.mask_name.get(), self.return_color)
    
    def update_color_vars(self, event=None, keep=None):
        # keep may be one of "r", "g", "b", "h", "s", "v", "hex" or None
        # if it is None, update all the variables based on self.current_color
        # otherwise use the value of the variable that "keep" refers to (and the others of the same color space)
        # to update the ones of the other color space
        if keep is None:
            self.color_vars["r"].set(str(self.current_color[0]))
            self.color_vars["g"].set(str(self.current_color[1]))
            self.color_vars["b"].set(str(self.current_color[2]))
            hexstring = rgb_to_hex(self.current_color)
            self.color_vars["hex"].set(hexstring[1:]) # remove '#'
            hsv = rgb_to_hsv(*self.current_color)
            self.color_vars["h"].set(str(hsv[0]))
            self.color_vars["s"].set(str(hsv[1]))
            self.color_vars["v"].set(str(hsv[2]))
            for k in self.color_vars:
                self.current_color_vars[k] = self.color_vars[k].get()
        else:
            if self.current_color_vars[keep] == self.color_vars[keep].get():
                return
            if keep == "hex":
                try:
                    hexstring = self.color_vars["hex"].get()
                    assert re.match(r"[0-9A-Fa-f]{6}", hexstring)
                    r, g, b = hex_to_rgb("#"+hexstring)
                    self.current_color = (r, g, b)
                    self.color_vars["r"].set(str(r))
                    self.color_vars["g"].set(str(g))
                    self.color_vars["b"].set(str(b))
                    hsv = rgb_to_hsv(r, g, b)
                    self.color_vars["h"].set(str(hsv[0]))
                    self.color_vars["s"].set(str(hsv[1]))
                    self.color_vars["v"].set(str(hsv[2]))
                except AssertionError:
                    return
            elif keep in ["r", "g", "b"]:
                try:
                    r = int(self.color_vars["r"].get())
                    g = int(self.color_vars["g"].get())
                    b = int(self.color_vars["b"].get())
                    assert 0 <= r <= 255
                    assert 0 <= g <= 255
                    assert 0 <= b <= 255
                    self.current_color = (r, g, b)
                    hexstring = rgb_to_hex((r, g, b))
                    self.color_vars["hex"].set(hexstring[1:]) # remove '#'
                    hsv = rgb_to_hsv(r, g, b)
                    self.color_vars["h"].set(str(hsv[0]))
                    self.color_vars["s"].set(str(hsv[1]))
                    self.color_vars["v"].set(str(hsv[2]))
                except (ValueError, AssertionError):
                    return
            elif keep in ["h", "s", "v"]:
                try:
                    h = int(self.color_vars["h"].get()) % 360
                    s = float(self.color_vars["s"].get())
                    v = float(self.color_vars["v"].get())
                    assert 0 <= s <= 100
                    assert 0 <= v <= 100
                    r, g, b = hsv_to_rgb(h, s, v)
                    self.current_color = (r, g, b)
                    hexstring = rgb_to_hex((r, g, b))
                    self.color_vars["hex"].set(hexstring[1:]) # remove '#'
                    self.color_vars["r"].set(str(r))
                    self.color_vars["g"].set(str(g))
                    self.color_vars["b"].set(str(b))
                except (ValueError, AssertionError):
                    return
            else:
                return
        # finally update preview & update current
        self.preview_crc.configure(image=self._make_circle(color=self.current_color))
        for k in self.color_vars:
            self.current_color_vars[k] = self.color_vars[k].get()
        self.update_inner_canvas()
        self.update_taget_h()
        self.update_target_sv()
    
    def update_inner_canvas(self):
        # create numpy array with S and V values depending on current H, and change inner canvas accordingly
        self.inner_canvas.delete("inner")
        h = int(self.color_vars["h"].get())
        new_image = np.stack([h*np.ones((101, 101), dtype=int), *np.meshgrid(range(101), range(100, -1, -1))], axis=-1, dtype=int)
        rgb_image = np.stack(self.np_hsv_to_rgb(new_image[..., 0], new_image[..., 1], new_image[..., 2]), axis=-1)
        self.inner_img = Image.fromarray(rgb_image.astype(np.uint8)).resize((self.inner_dim, self.inner_dim), Image.Resampling.LANCZOS)
        self.inner = ImageTk.PhotoImage(self.inner_img)
        self.inner_canvas.create_image(self.inner_dim/2, self.inner_dim/2, image=self.inner, tag="inner")
        self.inner_canvas.tag_raise("target_sv")
    
    def update_taget_h(self):
        # uses self.current_color H value to compute line on hue wheel
        h, _, _ = rgb_to_hsv(*self.current_color)
        self.color_canvas.delete("target_h")
        sn = math.sin(math.radians(h))
        cs = math.cos(math.radians(h))
        x0 = self.wheel_dim * (0.5 + 0.4 * cs)
        y0 = self.wheel_dim * (0.5 - 0.4 * sn)
        x1 = self.wheel_dim * (0.5 + 0.5 * cs)
        y1 = self.wheel_dim * (0.5 - 0.5 * sn)
        self.color_canvas.create_line(x0, y0, x1, y1, width=3, tag="target_h")
    
    def update_target_sv(self):
        # uses self.current_color S and V values to compute target position on inner canvas
        _, s, v = rgb_to_hsv(*self.current_color)
        self.inner_canvas.delete("target_sv")
        x = self.inner_dim * s / 100
        y = self.inner_dim * (100 - v) / 100
        self.inner_canvas.create_image(x, y, image=self.cross, tag="target_sv")
    
    def on_mouse_wheel(self, event):
        # track mouse position, and update H of self.current_color
        x, y = event.x, event.y
        _, s, v = rgb_to_hsv(*self.current_color)
        hrad = math.atan2(-(y-self.wheel_dim/2), x-self.wheel_dim/2)
        h = int(math.degrees(hrad))
        self.current_color = hsv_to_rgb(h, s, v)
        self.update_color_vars()
    
    def on_mouse_inner(self, event):
        # track mouse position, and update S and V of self.current_color
        x, y = event.x, event.y
        h, _, _ = rgb_to_hsv(*self.current_color)
        srel = min(max(x / self.inner_dim, 0), 1)
        vrel = min(max((self.inner_dim - y) / self.inner_dim, 0), 1)
        s = round(100*srel, 1)
        v = round(100*vrel, 1)
        self.current_color = hsv_to_rgb(h, s, v)
        self.update_color_vars()

class PreprocessingAdjustments(ctk.CTkToplevel):
    """
    Allows adjustments of preprocessing parameters.
    """
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        
        self.title("SLImTAG")
        self.resizable(False, False)
        
        self.canvas_size = 500
        
        # return values
        # if not None, this is a triple (brightness, contrast, shadows) with integers in -100 .. 100
        self.values = None
        
        # parameters
        self.brightness = parent.wand_brightness
        self.contrast = parent.wand_contrast
        self.shadows = parent.wand_gamma
        
        # UI
        # canvas
        self.preview_canvas = ctk.CTkCanvas(self, width=self.canvas_size, height=self.canvas_size)
        self.preview_canvas.grid(row=0, column=0, sticky="nsew")
        
        scale = max(parent.orig_w, parent.orig_h) / 400
        self.image = parent.image_orig.resize((int(parent.orig_w / scale), int(parent.orig_h / scale)), Image.Resampling.LANCZOS)
        self.update_display()
        
        # control panel
        self.panel = ctk.CTkFrame(self, fg_color="transparent")
        self.panel.grid(row=0, column=1, sticky="nsew")
        
        # sliders
        self.slider_lbl = {}
        
        ctk.CTkLabel(self.panel, text="Brightness", fg_color="transparent", anchor="w").grid(row=0, column=0, sticky="ew", padx=(10, 5), pady=(10, 2))
        self.slider_lbl["brightness"] = ctk.CTkLabel(self.panel, text=str(self.brightness), fg_color="transparent", anchor="e")
        self.slider_lbl["brightness"].grid(row=0, column=1, sticky="ew", padx=(5, 10), pady=(10, 2))
        self.brightness_slider = ctk.CTkSlider(self.panel, from_=-100, to=100, command=lambda v: self.slider_command(v, "brightness"))
        self.brightness_slider.set(self.brightness)
        self.brightness_slider.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=0)
        
        ctk.CTkLabel(self.panel, text="Contrast", fg_color="transparent", anchor="w").grid(row=2, column=0, sticky="ew", padx=(10, 5), pady=(10, 2))
        self.slider_lbl["contrast"] = ctk.CTkLabel(self.panel, text=str(self.contrast), fg_color="transparent", anchor="e")
        self.slider_lbl["contrast"].grid(row=2, column=1, sticky="ew", padx=(5, 10), pady=(10, 2))
        self.contrast_slider = ctk.CTkSlider(self.panel, from_=-100, to=100, command=lambda v: self.slider_command(v, "contrast"))
        self.contrast_slider.set(self.contrast)
        self.contrast_slider.grid(row=3, column=0, columnspan=2, sticky="ew", padx=10, pady=0)
        
        ctk.CTkLabel(self.panel, text="Shadows", fg_color="transparent", anchor="w").grid(row=4, column=0, sticky="ew", padx=(10, 5), pady=(10, 2))
        self.slider_lbl["shadows"] = ctk.CTkLabel(self.panel, text=str(self.shadows), fg_color="transparent", anchor="e")
        self.slider_lbl["shadows"].grid(row=4, column=1, sticky="ew", padx=(5, 10), pady=(10, 2))
        self.shadows_slider = ctk.CTkSlider(self.panel, from_=-100, to=100, command=lambda v: self.slider_command(v, "shadows"))
        self.shadows_slider.set(self.shadows)
        self.shadows_slider.grid(row=5, column=0, columnspan=2, sticky="ew", padx=10, pady=0)
        
        self.button_frame = ctk.CTkFrame(self.panel, fg_color="transparent")
        self.button_frame.grid(row=6, column=0, columnspan=2, sticky="sew", padx=10, pady=(0, 10))
        
        self.btn_ok = ctk.CTkButton(self.button_frame, text="Apply", command=self._on_ok)
        self.btn_ok.grid(row=0, column=0, padx=5)
        self.btn_cancel = ctk.CTkButton(self.button_frame, text="Cancel", command=self._on_cancel)
        self.btn_cancel.grid(row=0, column=1, padx=5)
        
        self.panel.grid_rowconfigure(6, weight=1)
        self.panel.grid_columnconfigure(0, weight=1)
        
        # Handle window close (X button)
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        
        # Binding <Enter> to OK button
        self.bind("<Return>", lambda e: self.btn_ok.invoke())

        # Determine window size
        self.update_idletasks()
        self.minsize(self.winfo_width(), self.winfo_height())
        
        # Put dialog on top of parent
        self.transient(parent)
        # Grab events
        self.grab_set()
        
        # Center on parent
        self._center_on_parent()

        # Wait until closed
        self.wait_window(self)
    
    def slider_command(self, value, slider):
        setattr(self, slider, int(value))
        self.slider_lbl[slider].configure(text=f"{getattr(self, slider)}")
        self.update_display()
        
    def update_display(self):
        self.preview_canvas.delete("all")
        image = Image.fromarray(adjust_image(np.array(self.image), self.brightness, self.contrast, self.shadows))
        self.tk_img = ImageTk.PhotoImage(image)
        self.preview_canvas.create_image(self.canvas_size/2, self.canvas_size/2, anchor="center", image=self.tk_img, tag="background_image")
    
    def _on_ok(self):
        if self.brightness != self.parent.wand_brightness or self.contrast != self.parent.wand_contrast or self.shadows != self.parent.wand_gamma:
            self.values = (self.brightness, self.contrast, self.shadows)
        else:
            self.values = None
        self.destroy()

    def _on_cancel(self):
        self.values = None
        self.destroy()

    def _center_on_parent(self):
        self.update_idletasks()
        px = self.parent.winfo_x()
        py = self.parent.winfo_y()
        pw = self.parent.winfo_width()
        ph = self.parent.winfo_height()
        w = self.winfo_width()
        h = self.winfo_height()
        x = px + (pw - w) // 2
        y = py + (ph - h) // 2
        self.geometry(f"+{x}+{y}")

def adjust_image(image, brightness=0, contrast=0, shadows=0):
    """
    Apply brightness, contrast and gamma to an image.

    Parameters
    ----------
    image : numpy.array with dtype=uint8
        Array representing the image to be adjusted.
    brightness : int or float in range [-100,100]
        The amount of brightness to be applied. A value of 0 leaves the image
        untouched.
    contrast : int or float in range [-100,100]
        The amount of contrast to be applied. A value of 0 leaves the image
        untouched.
    shadows : int or float in range [-100,100]
        The gamma correction to be applied. A value of 0 leaves the image
        untouched. Internally, -100 is mapped to gamma = 0.5 and 100 is mapped
        to gamma = 2.

    Returns
    -------
    A numpy.array with dtype=uint8 representing the adjusted image.

    """
    img = image.astype(np.float32) / 255.0
    alpha = 1 + contrast / 100.0
    beta = brightness / 100.0
    gamma = math.exp2(shadows / 100.0) # math.exp2 requires python >= 3.11
    img = alpha * (img - 0.5) + 0.5 + beta
    img = np.clip(img, 0.0, 1.0)
    img = img ** gamma
    return (255 * img).astype(np.uint8)


class Tooltip():
    '''
    Create a tooltip for a given widget as the mouse hovers over it.

    Source - https://stackoverflow.com/a/41079350
    by Alberto Vassena, adapted by Oscar Papini
    License - CC BY-SA 3.0
    '''

    def __init__(self, widget,
                 *,
                 bg=ctk.ThemeManager.theme["CTk"]["fg_color"],
                 pad=(10, 2, 10, 2),
                 text='',
                 waittime=400):

        self.waittime = waittime  # in miliseconds
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.onEnter)
        self.widget.bind("<Leave>", self.onLeave)
        self.widget.bind("<ButtonPress>", self.onLeave)
        self.bg = bg
        self.pad = pad
        self.id = None
        self.tw = None

    def onEnter(self, event=None):
        self.schedule()

    def onLeave(self, event=None):
        self.unschedule()
        self.hide()

    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(self.waittime, self.show)

    def unschedule(self):
        id_ = self.id
        self.id = None
        if id_:
            self.widget.after_cancel(id_)

    def show(self):
        def tip_pos_calculator(widget, label, *, tip_delta=(10, 5), pad=self.pad):
            # function that computes the position of tne tooltip depending on
            # the position of the widget
            w = widget

            s_width, s_height = w.winfo_screenwidth(), w.winfo_screenheight()

            width, height = (pad[0] + label.winfo_reqwidth() + pad[2],
                             pad[1] + label.winfo_reqheight() + pad[3])

            mouse_x, mouse_y = w.winfo_pointerxy()

            x1, y1 = mouse_x + tip_delta[0], mouse_y + tip_delta[1]
            x2, y2 = x1 + width, y1 + height

            x_delta = x2 - s_width
            if x_delta < 0:
                x_delta = 0
            y_delta = y2 - s_height
            if y_delta < 0:
                y_delta = 0

            offscreen = (x_delta, y_delta) != (0, 0)

            if offscreen:
                if x_delta:
                    x1 = mouse_x - tip_delta[0] - width
                if y_delta:
                    y1 = mouse_y - tip_delta[1] - height

            offscreen_again = y1 < 0  # out on the top

            if offscreen_again:
                # No further checks will be done.
                # TIP:
                # A further mod might automagically augment the
                # wraplength when the tooltip is too high to be
                # kept inside the screen.
                y1 = 0

            return x1, y1

        bg = self.bg
        pad = self.pad
        widget = self.widget

        # creates a toplevel window
        self.tw = tk.Toplevel(widget)
        self.tw.withdraw() # hide until composed

        # Leaves only the label and removes the app window
        self.tw.wm_overrideredirect(True)

        win = ctk.CTkFrame(self.tw, fg_color=bg, border_width=1, corner_radius=0)
        label = ctk.CTkLabel(win,
                             text=self.text,
                             justify="left",
                             fg_color=bg)
        label.grid(padx=(pad[0], pad[2]),
                   pady=(pad[1], pad[3]),
                   sticky="nsew")
        win.grid()

        x, y = tip_pos_calculator(widget, label)
        self.tw.update_idletasks()
        self.tw.geometry(f"{win.winfo_width()-(win.winfo_width()%2)}x{win.winfo_height()}+{x}+{y}")
        self.tw.deiconify()

    def hide(self):
        tw = self.tw
        if tw:
            tw.destroy()
        self.tw = None

class SplashScreen(tk.Toplevel):
    def __init__(self):
        super().__init__()
        self.overrideredirect(True)
        self.title("Loading...")
        self.configure(bg="black")
        
        logo_size = 394
        splash_height = logo_size + 32
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width // 2) - (logo_size // 2)
        y = (screen_height // 2) - (splash_height // 2)
        self.geometry(f"{logo_size}x{splash_height}+{x}+{y}")
        my_image = ctk.CTkImage(dark_image=Image.open(os.path.join("images", "logo.png")), size=(logo_size,logo_size))
        ctk.CTkLabel(self, text="Loading...", image=my_image).pack()
        self.progress = ctk.CTkProgressBar(self, width=logo_size-32, height=16, progress_color="red", fg_color="#101010")
        self.progress.pack(pady=8)
        self._set(0)
        

    def step(self, value):
        value = value / 100
        self._set(self.progress.get()+value) 

    def _set(self, value):
        self.progress.set(value) 
        self.update()