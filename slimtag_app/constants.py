"""Application-wide paths, model registry, and theme constants."""

import os

import customtkinter as ctk


CONFIG_FILE_PATH = "config.toml"
MODELS_BASE_PATH = "models"

STATUS_SYMBOL = "●"
STATUS_COLOR = {
    "ready": ("#2ECC71", "#2ECC71"),
    "loading": ("#F1C40F", "#F1C40F"),
    "error": ("#E74C3C", "#E74C3C"),
    "idle": ("#95A5A6", "#95A5A6"),
}

SAM_MODELS = {
    "SAM (ViT-B)": {
        "type": "vit_b",
        "path": os.path.join(MODELS_BASE_PATH, "sam_vit_b_01ec64.pth"),
    },
    "SAM (ViT-L)": {
        "type": "vit_l",
        "path": os.path.join(MODELS_BASE_PATH, "sam_vit_l_0b3195.pth"),
    },
    "SAM (ViT-H)": {
        "type": "vit_h",
        "path": os.path.join(MODELS_BASE_PATH, "sam_vit_h_4b8939.pth"),
    },
}

ctk.set_default_color_theme("color_palette.json")
HIGHLIGHT_COLOR = ctk.ThemeManager.theme["CTkButton"]["border_color"]
