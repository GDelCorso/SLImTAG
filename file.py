import tkinter as tk
from PIL import Image, ImageTk

class ImageZoomApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Zoom")

        # Create a Canvas widget
        self.canvas = tk.Canvas(self.root, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Initialize image and image reference
        self.image = None
        self.tk_image = None

        # Load an initial image (you can use your own image file)
        self.load_image("images/gui.png")

        # Create Zoom In and Zoom Out buttons
        zoom_in_button = tk.Button(self.root, text="Zoom In", command=self.zoom_in)
        zoom_out_button = tk.Button(self.root, text="Zoom Out", command=self.zoom_out)
        zoom_in_button.pack(side=tk.LEFT)
        zoom_out_button.pack(side=tk.LEFT)

        # Bind mouse wheel events for zooming
        self.canvas.bind("<Button-4>", self.zoom_in)
        self.canvas.bind("<Button-5>", self.zoom_out)

    def load_image(self, filename):
        # Load the image using PIL (Python Imaging Library)
        self.image = Image.open(filename)
        self.tk_image = ImageTk.PhotoImage(self.image)

        # Display the image on the Canvas
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def zoom_in(self, event=None):
        # Increase the image size by a factor (e.g., 1.2)
        self.image = self.image.resize((int(self.image.width * 1.2), int(self.image.height * 1.2)))
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def zoom_out(self, event=None):
        # Decrease the image size by a factor (e.g., 0.8)
        self.image = self.image.resize((int(self.image.width * 0.8), int(self.image.height * 0.8)))
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageZoomApp(root)
    root.mainloop()