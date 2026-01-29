import customtkinter as ctk

class MultiButtonDialog(ctk.CTkToplevel):
    def __init__(self, parent,
                 message="Are you sure?",
                 buttons=(("Yes", "yes-option"), ("No", "no-option"), ("Cancel", None))
                 ):
        super().__init__(parent)

        self.parent = parent
        self.result = None

        self.title("SLImTAG")
        self.resizable(False, False)
        self.maxsize = 500

        # Make modal
        self.transient(parent)
        self.grab_set()
        
        n_buttons = len(buttons)

        # Message
        ctk.CTkLabel(self, text=message, wraplength=self.maxsize, justify="left").grid(row=0, column=0, sticky="nsew", padx=20, pady=(20, 10))

        # Buttons frame
        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=(10, 20))

        for i in range(n_buttons):
            text, value = buttons[i]
            btn = ctk.CTkButton(button_frame, text=text, command=lambda v=value: self._on_button(v))
            btn.grid(row=0, column=i, sticky="ew", padx=5)
        
        button_frame.grid_columnconfigure(list(range(n_buttons)), weight=1)

        # Handle window close (X button)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Determine window size
        self.update_idletasks()
        self.minsize(max(self.winfo_width(), self.maxsize), self.winfo_height())
        
        # Center on parent
        self._center_on_parent()

        # Wait until closed
        self.wait_window(self)

    def _on_button(self, value):
        self.result = value
        self.destroy()

    def _on_close(self):
        self.result = None
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