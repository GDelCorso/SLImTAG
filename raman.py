"""
Functions for save/load csv reports

@author: Oscar Papini
"""

import re
from pathlib import Path
import csv
from scipy import ndimage
from PIL import ImageDraw, ImageFont

def check_report(path_original_image):
    """
    Check if a report csv already exists.
    
    If in {input.parent} there isn't already a file called {input.stem}.csv,
    return None; otherwise, return an integer n, which is the maximum of
    the set {0} U {n in N | {input.stem}_backup_<n>.csv exists}
    
    So for example this returns 0 if only {input.stem}.csv exists, but no
    backup csv.
    """
    path_og = Path(path_original_image)
    filename = path_og.stem
    pattern = re.compile(rf"{re.escape(filename)}(?:_backup_(\d+))?\.csv")
    backups = []
    for f in path_og.parent.iterdir():
        match = pattern.fullmatch(f.name)
        if match and f.is_file():
            backups.append(int(match.group(1) or 0))
    if backups and 0 in backups:
        return max(backups)
    else:
        return None

def save_report(mask, label_names, csv_path):
    """
    Create a report csv.
    
    The report contains a row for each connected component for each mask,
    with the columns:
        ID (progressive along all csv)
        label
        note
        x top left (%)
        y top left (%)
        x middle (%)
        y middle (%)
        x bottom right (%)
        y bottom right (%)
        x top left (px)
        y top left (px)
        x middle (px)
        y middle (px)
        x bottom right (px)
        y bottom right (px)
    """
    fieldnames = ["ID", "label", "note",
                  "x top left (%)", "y top left (%)",
                  "x middle (%)", "y middle (%)",
                  "x bottom right (%)", "y bottom right (%)",
                  "x top left (px)", "y top left (px)",
                  "x middle (px)", "y middle (px)",
                  "x bottom right (px)", "y bottom right (px)"]
    global_id = 0
    mask_h, mask_w = mask.shape
    with open(csv_path, "w", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in label_names.keys():
            selected = mask == i
            if not selected.any():
                continue
            labels, num_features = ndimage.label(selected)
            if num_features == 0:
                continue
            slices = ndimage.find_objects(labels)
            for s in slices:
                if s is not None:
                    row = {}
                    row["ID"] = global_id
                    row["label"] = label_names[i]
                    row["note"] = ""
                    xmin = s[1].start
                    xmax = s[1].stop - 1
                    ymin = s[0].start
                    ymax = s[0].stop - 1
                    row["x top left (%)"] = round(xmin / mask_w, 5)
                    row["y top left (%)"] = round(ymin / mask_h, 5)
                    row["x middle (%)"] = round((xmin + xmax) / (2 * mask_w), 5)
                    row["y middle (%)"] = round((ymin + ymax) / (2 * mask_h), 5)
                    row["x bottom right (%)"] = round(xmax / mask_w, 5)
                    row["y bottom right (%)"] = round(ymax / mask_h, 5)
                    row["x top left (px)"] = int(xmin)
                    row["y top left (px)"] = int(ymin)
                    row["x middle (px)"] = int((xmin + xmax) / 2)
                    row["y middle (px)"] = int((ymin + ymax) / 2)
                    row["x bottom right (px)"] = int(xmax)
                    row["y bottom right (px)"] = int(ymax)
                    writer.writerow(row)
                    global_id += 1
            del selected, labels
    return None

def save_preview(pil_image, csv_path, save_path):
    """
    Create mask preview with bounding boxes.
    """
    img = pil_image.copy() # just for safety
    imgdraw = ImageDraw.Draw(img)
    drawcolor = (254, 223, 0)
    font = ImageFont.load_default(size=24)
    with open(csv_path, "r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            xmin = int(row["x top left (px)"])
            ymin = int(row["y top left (px)"])
            xmax = int(row["x bottom right (px)"])
            ymax = int(row["y bottom right (px)"])
            imgdraw.rectangle([(xmin, ymin), (xmax, ymax)], outline='black', width=1)
            try:
                imgdraw.rectangle([(xmin+1, ymin+1), (xmax-1, ymax-1)], outline=drawcolor, width=2)
            except ValueError: # for small areas, overwrite outer border with draw color
                imgdraw.rectangle([(xmin, ymin), (xmax, ymax)], outline=drawcolor, width=1)
            try:
                imgdraw.rectangle([(xmin+3, ymin+3), (xmax-3, ymax-3)], outline='black', width=1)
            except ValueError: # for small areas do not add inner border
                pass
            # determine label position depending on the box position
            if float(row["x middle (%)"]) <= 0.5:
                xlabel = xmin + 4
                anchor = "l"
            else:
                xlabel = xmax - 4
                anchor = "r"
            if float(row["y middle (%)"]) <= 0.5:
                ylabel = ymin + 4
                anchor += "t"
            else:
                ylabel = ymax - 4
                anchor += "b"
            imgdraw.text((xlabel, ylabel), f"[{row['ID']}] {row['label']}",
                         font=font, fill=drawcolor, anchor=anchor,
                         stroke_fill='black', stroke_width=2)
    img.save(save_path)
    return None