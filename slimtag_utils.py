import customtkinter as ctk
import tkinter as tk

from PIL import Image, ImageTk
import sys
import os
import math
import re

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
            
        for i in range(n_buttons):
            text, value = buttons[i]
            btn = ctk.CTkButton(button_row[i//3], text=text, command=lambda v=value: self._on_button(v))
            btn.grid(row=0, column=i%3, sticky= "ew" if i//3 < n_btn_rows-1 else "", padx=5)
        
        self.grid_columnconfigure(0, weight=1)
        button_frame.grid_columnconfigure(0, weight=1)

        # Handle window close (X button)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

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

class ColorHelper():
    '''
    calculates text color based on background one   
    '''
    def getTextColor(color, lightColor='#ffffff', darkColor='#000000'):
        r = int(color[1:3], 16); # hexToR
        g = int(color[3:5], 16); # hexToG
        b = int(color[5:7], 16); # hexToB
        uicolors = [r / 255, g / 255, b / 255];
        c = list(map(ColorHelper._col, uicolors))
        L = (0.2126 * c[0]) + (0.7152 * c[1]) + (0.0722 * c[2]);
        if L > 0.179:
            return darkColor
        return lightColor

    def _col(col):
        if col <= 0.03928:
            return col / 12.92
        return pow((col + 0.055) / 1.055, 2.4)

    def hexToRGB(h):
        if len(h) != 7:
            return h if h == '*' else ''
        h = h.lstrip("#")
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    def rgbToHEX(rgb):
        if type(rgb) == str:
            rgb = rgb[1:-1]
            rgb = list(map(int,rgb.split(',')))

        red, green, blue = rgb
        return '#%02x%02x%02x' % (red, green, blue)

class MyColorPicker(ctk.CTkToplevel):

    def __init__(self,
                 width: int = 300,
                 title: str = "Choose Color",
                 initial_color: str = None,
                 bg_color: str = None,
                 fg_color: str = None,
                 button_color: str = None,
                 button_hover_color: str = None,
                 text: str = "OK",
                 corner_radius: int = 24,
                 slider_border: int = 1,
                 **button_kwargs):
    
        super().__init__()
        
        PATH = "MyColorPickerResources"
        self.title(title)
        WIDTH = width if width>=200 else 200
        HEIGHT = WIDTH + 150
        self.image_dimension = self._apply_window_scaling(WIDTH - 100)
        self.target_dimension = self._apply_window_scaling(20)
        
        self.maxsize(WIDTH, HEIGHT)
        self.minsize(WIDTH, HEIGHT)
        self.resizable(width=False, height=False)
        self.transient(self.master)
        self.lift()
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.after(10)
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        self.default_hex_color = "#ffffff"  
        self.default_rgb = [255, 255, 255]
        self.rgb_color = self.default_rgb[:]
        
        self.bg_color = self._apply_appearance_mode(ctk.ThemeManager.theme["CTkFrame"]["fg_color"]) if bg_color is None else bg_color
        self.fg_color = self.fg_color = self._apply_appearance_mode(ctk.ThemeManager.theme["CTkFrame"]["top_fg_color"]) if fg_color is None else fg_color
        self.button_color = self._apply_appearance_mode(ctk.ThemeManager.theme["CTkButton"]["fg_color"]) if button_color is None else button_color
        self.button_hover_color = self._apply_appearance_mode(ctk.ThemeManager.theme["CTkButton"]["hover_color"]) if button_hover_color is None else button_hover_color
        self.button_text = text
        self.corner_radius = corner_radius
        self.slider_border = 10 if slider_border>=10 else slider_border
        
        self.config(bg=self.bg_color)
        
        self.frame = ctk.CTkFrame(master=self, fg_color=self.fg_color, bg_color=self.bg_color)
        self.frame.grid(padx=20, pady=20, sticky="nswe")
          
        self.canvas = tk.Canvas(self.frame, height=self.image_dimension, width=self.image_dimension, highlightthickness=0, bg=self.fg_color)
        self.canvas.pack(pady=20)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)

        self.img1 = Image.open(os.path.join(PATH, 'color_wheel.png')).resize((self.image_dimension, self.image_dimension), Image.Resampling.LANCZOS)
        self.img2 = Image.open(os.path.join(PATH, 'target.png')).resize((self.target_dimension, self.target_dimension), Image.Resampling.LANCZOS)

        self.wheel = ImageTk.PhotoImage(self.img1)
        self.target = ImageTk.PhotoImage(self.img2)
        
        self.canvas.create_image(self.image_dimension/2, self.image_dimension/2, image=self.wheel)
        self.set_initial_color(initial_color)
        
        self.brightness_slider_value = ctk.IntVar()
        self.brightness_slider_value.set(255)
        
        self.slider = ctk.CTkSlider(master=self.frame, height=20, border_width=self.slider_border,
                                              button_length=15, progress_color=self.default_hex_color, from_=0, to=255,
                                              variable=self.brightness_slider_value, number_of_steps=256,
                                              button_corner_radius=self.corner_radius, corner_radius=self.corner_radius,
                                              button_color=self.button_color, button_hover_color=self.button_hover_color,
                                              command=lambda x:self.update_colors())
        self.slider.pack(fill="both", pady=(0,15), padx=20-self.slider_border)

        self.label = ctk.CTkEntry(master=self.frame, text_color="#000000", border_color="#ffffff", height=50, fg_color=self.default_hex_color, corner_radius=self.corner_radius, justify="center")
        
        self.update_value(self.label, self.default_hex_color)
        self.label.bind("<KeyRelease>", self.key_release)
        
        self.label.pack(fill="both", padx=10)
        
        self.button = ctk.CTkButton(master=self.frame, text=self.button_text, height=50, corner_radius=self.corner_radius, fg_color=self.button_color,
                                              hover_color=self.button_hover_color, command=self._ok_event, **button_kwargs)
        self.button.pack(fill="both", padx=10, pady=20)
                
        self.after(150, lambda: self.label.focus())
                
        self.grab_set()
       
    def key_release(self, evt):
        if re.match(r"#[0-9a-fA-F]{6}", self.label.get()):
            self.label.configure(fg_color=self.label.get(), border_color=self.label.get(), text_color=ColorHelper.getTextColor(self.label.get()))
        


    def get(self):
        self._color = self.label._fg_color
        self.master.wait_window(self)
        return self._color
    
    def _ok_event(self, event=None):
        self._color = self.label._fg_color
        self.grab_release()
        self.destroy()
        del self.img1
        del self.img2
        del self.wheel
        del self.target
        
    def _on_closing(self):
        self._color = None
        self.grab_release()
        self.destroy()
        del self.img1
        del self.img2
        del self.wheel
        del self.target
        
    def on_mouse_drag(self, event):
        x = event.x
        y = event.y
        self.canvas.delete("all")
        self.canvas.create_image(self.image_dimension/2, self.image_dimension/2, image=self.wheel)
        
        d_from_center = math.sqrt(((self.image_dimension/2)-x)**2 + ((self.image_dimension/2)-y)**2)
        
        if d_from_center < self.image_dimension/2:
            self.target_x, self.target_y = x, y
        else:
            self.target_x, self.target_y = self.projection_on_circle(x, y, self.image_dimension/2, self.image_dimension/2, self.image_dimension/2 -1)

        self.canvas.create_image(self.target_x, self.target_y, image=self.target)
        
        self.get_target_color()
        self.update_colors()
  
    def get_target_color(self):
        try:
            self.rgb_color = self.img1.getpixel((self.target_x, self.target_y))
            
            r = self.rgb_color[0]
            g = self.rgb_color[1]
            b = self.rgb_color[2]    
            self.rgb_color = [r, g, b]
            
        except AttributeError:
            self.rgb_color = self.default_rgb
    
    def update_colors(self):
        brightness = self.brightness_slider_value.get()

        self.get_target_color()

        r = int(self.rgb_color[0] * (brightness/255))
        g = int(self.rgb_color[1] * (brightness/255))
        b = int(self.rgb_color[2] * (brightness/255))
        
        self.rgb_color = [r, g, b]

        self.default_hex_color = "#{:02x}{:02x}{:02x}".format(*self.rgb_color)
        
        self.slider.configure(progress_color=self.default_hex_color)
        self.label.configure(fg_color=self.default_hex_color, border_color=self.default_hex_color)
        
        self.update_value(self.label,str(self.default_hex_color))
        
        self.label.configure(text_color=ColorHelper.getTextColor(self.label.get()), border_color=self.label.get())
        
            
    def projection_on_circle(self, point_x, point_y, circle_x, circle_y, radius):
        angle = math.atan2(point_y - circle_y, point_x - circle_x)
        projection_x = circle_x + radius * math.cos(angle)
        projection_y = circle_y + radius * math.sin(angle)

        return projection_x, projection_y
    
    def set_initial_color(self, initial_color):
        # set_initial_color is in beta stage, cannot seek all colors accurately
        
        if initial_color and initial_color.startswith("#"):
            try:
                r,g,b = tuple(int(initial_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            except ValueError:
                return
            
            self.default_hex_color = initial_color
            for i in range(0, self.image_dimension):
                for j in range(0, self.image_dimension):
                    self.rgb_color = self.img1.getpixel((i, j))
                    if (self.rgb_color[0], self.rgb_color[1], self.rgb_color[2])==(r,g,b):
                        self.canvas.create_image(i, j, image=self.target)
                        self.target_x = i
                        self.target_y = j
                        return
                    
        self.canvas.create_image(self.image_dimension/2, self.image_dimension/2, image=self.target)
    
    def update_value(self, e, txt):
        '''
        update a probability value in Textbox entry
        '''
        e.delete(0,ctk.END)
        e.insert(0,txt)

if __name__ == "__main__":
    app = MyColorPicker()
    app.mainloop()
