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

Please note that the version numbers listed here refer to the environment in which SLImTAG has been tested ― it is possible that lower version will work too.

- Python 3.12
- PyTorch (for SAM): torch == 2.5.1 (CUDA 12.1 recommended if available) and torchvision == 0.20.1
- numpy == 2.3.1 (mask manipulation)
- scipy == 1.17.0 (erosion/dilation tool)
- pillow == 12.0.0 (images management)
- customtkinter == 5.2.2 (GUI)

Also tkinter is required, but it cannot be installed via pip. On Windows it should be provided with Python; on Mac and Linux, you may need to install it through your system's package manager (e.g. `brew install python-tk` with homebrew for Mac, or `sudo apt install python3-tk` for Ubuntu-based Linux distros).

---

## Installation

1. Clone the repository

    ```bash
    git clone https://github.com/GDelCorso/SLImTAG
    cd SLImTAG
    ```

2. Create a virtual environment

    ```bash
    python3 -m venv slimtag-venv
    ```

3. Activate the environment: on Mac/Linux

    ```bash
    source slimtag-venv/bin/activate
    ```

    on Windows

    ```bash
    slimtag-venv\Scripts\activate
    ```

4. Install dependencies:

    ```bash
    python3 -m pip install -r requirements.txt
    ```

5. Download SAM's weights: download the file `sam_vit_b_01ec64.pth` from [https://github.com/facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything) (in the "Model Checkpoints" section) and place it in the `models` folder. If you have wget, you can do it via terminal:

    ```bash
    wget -O models/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
    ```

6. That's it! To run the program just do

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
## TO-DO LIST and BUGFIX:

### Major
- [ ] Rewrite GUI
- [ ] Redefine zoom function (by avoiding update loop on subselection)
- [ ] Ctrl+magic wand: multiple conditional points (left: positive, right: negative), activate SAM at release Ctrl, and mask preview

### Minor
- [ ] Zoom on mouse location instead of top left corner
- [ ] Add statusbar
- [ ] Brush subraction and connected component currently don't work when "Only add on empty" is active
- [ ] Implement argparse with "--no-sam" option that disables SAM (disable button; avoid libraries import; deactivate SAM loading in `load_image`; add requirements-no-sam.txt)
- [ ] Integrate PyInstaller and generate Windows binary (both with and without SAM)
- [ ] Convert hardcoded parameters to argparse arguments
- [ ] SAM click-and-drag: apply to selected bounding box (at release) (both positive and negative)
- [ ] To consider: in smoothing tool, add toggle for locking mask at image border, so that it is not erased (useful if mask is background)
- [x] Add <kbd>Q</kbd> (or <kbd>Ctrl</kbd>+<kbd>Q</kbd>) as quit shortcut
- [ ] Define an additional .csv containing name/mask value bindings for semantic segmentation when needed
- [ ] When a mask is removed, a new mask additions should be placed on the first empty value of the list
- [ ] Track if a mask is unsaved, and prompt a "There are unsaved changes to the mask. Quit anyway?" message accordingly (check to be added in `quit_program` method)

### Potential additional features
- [ ] Different brush shapes (square)
- [ ] Rectangle/polygonal "add to mask" tool
- [ ] Define plugins for different features to mantain the system lightweight and make it adaptable to user experience
- [ ] Add button "invert mask"
- [ ] Add "save png with mask as alpha channel"

### Extension to bio-medical fields
- [ ] Import MedSAM and modify the MagicWand wrapper
- [ ] Import also .nrrd and .nifti files for MRI/scan segmentation
- [ ] Allows to import 3D (i.e., ordered list of 2D elements) views for 3D MRI scans and video segmentation
