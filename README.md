# EasyTAG – Interactive Image Segmentation Tool

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**EasyTAG** is a simple and intuitive GUI tool for interactive image segmentation using brushes, connected components, magic wand, and SAM (Segment Anything Model). It supports multiple masks with color previews, undo, and easy saving/loading of masks.

---

## Features

- Add, select, and delete up to **20 masks**.
- Brush tool for manual painting.
- Magic Wand (SAM) for AI-assisted segmentation.
- Connected Component tool for precise selection/removal.
- Smoothing tool (dilation/erosion) for mask refinement.
- Undo support with **Z shortcut**.
- Save masks as `.npy` and overlay PNG images.
- Load masks from `.npy` or PNG files.
- Shortcut keys for all tools.

---

## Requirements

- Python ≥ 3.9  
- PyTorch  
- NumPy  
- SciPy  
- Pillow  
- Tkinter  
- CustomTkinter (`pip install customtkinter`)  
- Segment Anything Model (SAM) weights  

> Download SAM model weights and place in `models/` directory: `sam_vit_b_01ec64.pth`  

Install dependencies:

```bash
pip install -r requirements.txt
