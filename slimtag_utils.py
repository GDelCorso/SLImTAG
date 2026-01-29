import customtkinter as ctk

class MultiButtonDialog(ctk.CTkToplevel):
    def __init__(self, parent, message="", buttons=[("Cancel", None)]
                 ):
        super().__init__(parent)

        self.parent = parent
        self.return_value = None

        self.title("SLImTAG")
        self.resizable(False, False)
        self.maxsize = 500
       
        n_buttons = len(buttons)
        n_btn_rows = ((n_buttons - 1) // 3) + 1 # number of button rows

        print(n_btn_rows)

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
        self.minsize(max(self.winfo_width(), self.maxsize), self.winfo_height())
        
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