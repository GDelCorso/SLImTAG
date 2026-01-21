# SLImTAG – *S*imple *L*ight-weight *Im*age *Tag*ging tool


![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**SLImTAG** is a simple and intuitive GUI tool for interactive image segmentation using brushes, connected components, magic wand, and SAM (Segment Anything Model). It supports multiple masks with color previews, undo, and easy saving/loading of masks.

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

- Python == 3.12

- PyTorch: torch == 2.5.1 (CUDA 12.1 recommended if available) and torchvision == 0.20.1

- NumPy == 2.3.1
- SciPy == 1.17.0
- Pillow == 12.0.0
- CustomTkinter == 5.2.2
- Segment Anything Model (SAM) weights == 1.0

### SAM model weights

Download the file `sam_vit_b_01ec64.pth` from [https://github.com/facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything) (in the "Model Checkpoints" section) and place it in a subfolder named `models` in SLImTAG's root folder.

---

## Installation

Clone the repository
```bash
git clone https://github.com/GDelCorso/SLImTAG
cd SLImTAG
```

Create a virtual environment

```bash
python3 -m venv slimtag-venv
```

Activate the environment: on Mac/Linux

```bash
source slimtag-venv/bin/activate
```

on Windows
```bash
myfirstproject\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

That's it! To run the program just do

```bash
python3 SLImTag.py
```

from the project folder, with the virtual environment activated.

---

## Usage
### Buttons
| Button / Tool           | Description                        | Shortcut     |
| ----------------------- | ---------------------------------- | ------------ |
| **Brush**               | Paint or erase manually            | <kbd>B</kbd> |
| **Magic Wand**          | AI-assisted segmentation (SAM)     | <kbd>M</kbd> |
| **Connected Component** | Select/remove connected areas      | <kbd>C</kbd> |
| **Smoothing**           | Dilate/erode selected component    | <kbd>S</kbd> |
| **Undo**                | Undo last change                   | <kbd>Z</kbd> |
| **Add New Mask**        | Create a new mask with custom name | –            |

---

### Mouse Actions

Left Click – Apply active tool (Shift modifies behavior for some tools).

Right Click – Remove or erode depending on active tool.

Drag – Brush painting follows mouse movement.

Mouse Wheel – Zoom in/out of the image. Also <kbd>Ctrl</kbd>+<kbd>+</kbd> and <kbd>Ctrl</kbd>+<kbd>-</kbd> work for zoom in/out, and either <kbd>Ctrl</kbd>+<kbd>=</kbd> or <kbd>Ctrl</kbd>+<kbd>space</kbd> reset the zoom.

---

### Saving & Loading Masks
Save Mask: Saves the mask as an indexed PNG file, and a semi-transparent PNG overlay of the masks.

Load Mask: Loads a PNG representing a mask, extracting up to 20 unique colors as separate masks. Extra colors are ignored.


---
# TO-DO LIST and BUGFIX:
1) Known issue that zoomed images lead to unefficient brush due to rescaling on the full img needed.
2) Integrate SAM with multiple positive and negative points as an extra buttn "advanced SAM"
