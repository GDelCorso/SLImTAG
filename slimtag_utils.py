import customtkinter as ctk
import tkinter as tk

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