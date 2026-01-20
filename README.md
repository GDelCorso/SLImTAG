# NOME – Interactive Image Segmentation Tool

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**EasyTAG** is a simple and intuitive GUI tool for interactive image segmentation using brushes, connected components, magic wand, and SAM (Segment Anything Model). It supports multiple masks with color previews, undo, and easy saving/loading of masks.

---

## Features

- Add, select, and delete multiple tasks
- Brush tool for manual painting.
- Magic Wand (SAM) for AI-assisted segmentation.
- Connected Component tool for precise selection/removal.
- Smoothing tool (dilation/erosion) for mask refinement.
- Undo support with **Z shortcut**.
- Save masks and load masks
- Minimal libraries requirements

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
```

## Installation
```bash
git clone <repo-url>
cd EasyTAG
```

## Usage
# Buttons
| Button / Tool           | Description                        | Shortcut |
| ----------------------- | ---------------------------------- | -------- |
| **Brush**               | Paint or erase manually            | B        |
| **Magic Wand**          | AI-assisted segmentation (SAM)     | M        |
| **Connected Component** | Select/remove connected areas      | C        |
| **Smoothing**           | Dilate/erode selected component    | S        |
| **Undo**                | Undo last change                   | Z        |
| **Add New Mask**        | Create a new mask with custom name | –        |


# Mouse Actions

Left Click – Apply active tool (Shift modifies behavior for some tools).

Right Click – Remove or erode depending on active tool.

Drag – Brush painting follows mouse movement.

Mouse Wheel – Zoom in/out of the image.

# Saving & Loading Masks
Save Mask: Saves .npy and a semi-transparent PNG overlay of the masks.

Load Mask: Loads .npy or .png. If PNG, extracts up to 20 unique colors as separate masks. Extra colors are ignored.
