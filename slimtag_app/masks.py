"""Methods for the masks responsibilities of SegmentationApp."""

import customtkinter as ctk
import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw

from slimtag_app.constants import HIGHLIGHT_COLOR
from slimtag_color_utils import hex_to_rgb, rgb_to_hex
from slimtag_utils import MaskEditDialog, MultiButtonDialog


class MasksMixin:
    """Composable masks behaviour for ``SegmentationApp``."""

    def push_undo(self):
        '''
        Saves a copy of the current mask (mask_orig) into the undo_stack.
        '''
        if self.mask_orig is not None:
            self.undo_stack.append(self.mask_orig.copy())
            if len(self.undo_stack) > self.slimtag_config["main"]["undo_depth"]:
                self.undo_stack.pop(0)

    def undo(self):
        '''
        Pops the last saved state from undo_stack and restores it as the 
        current mask_orig. Calls update_display() to refresh the canvas so the 
        user sees the previous mask.
        '''
        # Check if an image is loaded
        if not self.image_is_loaded():
            return
        
        # Deactivate undo when using multipoint wand
        if len(self.sam_points) > 0:
            return
        
        if self.undo_stack:
            self.mask_orig = self.undo_stack.pop()
            self.update_lock()
            self.update_display(update_image=False)
    
    #%% MASK MANAGEMENT
    def add_mask(self, name=None):
        '''
        Creates a new mask, asks the user for a name via dialog, assigns it a 
        unique ID and a color, and updates the UI accordingly.
        '''
        # Check if an image is loaded
        if not self.image_is_loaded():
            return
        
        self.deactivate_tools()
        if len(self.mask_labels) >= self.slimtag_config["mask"]["max_masks"]:
            return
        
        default_color = [tuple(c) for c in self.slimtag_config["mask"]["default_mask_colors"] if tuple(c) not in self.mask_colors.values()][0] # get first non-used color
        color = None
        
        # candidate mask id
        mid = min([i+1 for i in range(self.slimtag_config["mask"]["max_masks"]) if i+1 not in self.mask_labels.keys()])
        
        if name is None:
            # Ask user for mask name
            color_in_use = True
            while color_in_use:
                name, color = MaskEditDialog(self, title="New mask", initial_color=rgb_to_hex(default_color), mask_name=f"mask_{mid}").get()
                if color is not None and hex_to_rgb(color) in self.mask_colors.values():
                    MultiButtonDialog(self, message=f"Color {color} already in use. Please choose another one.", buttons=(("OK", None),))
                else:
                    color_in_use = False

        if not name:  # User cancelled or empty
            return
        color = default_color if not color else hex_to_rgb(color)
        
        self.mask_labels[mid] = name
        self.mask_colors[mid] = color
        
        self.mask_widgets[mid] = self.create_mask_widget(mid)
        self.mask_widgets[mid].pack(fill="x", expand=True, padx=(6, 2), pady=(3, 0))
        self.change_mask(target_id=mid) # this also sets self.active_mask_id
        
        # since new masks are created non-locked and non-hidden,
        # revert state of "lock all" and "hide all" buttons but do not change
        # status of other masks
        self.toggle_all_masks_hide(set_hide=False, enabled=True, propagate=False)
        self.toggle_all_masks_lock(set_lock=False, enabled=True, propagate=False)
        self.set_controls_state(True) # activate buttons if there is at least one mask
    
    def _crc(self, mid, circle_size = 21):
        # aux function that draws a circle for the mask with ID mid
        color_circle = Image.new("RGBA", (circle_size+1, circle_size+1), (0, 0, 0, 0))
        color_circle_draw = ImageDraw.Draw(color_circle)
        color_circle_draw.ellipse((0, 0, circle_size, circle_size), fill=self.mask_colors[mid])
        return ctk.CTkImage(color_circle, size=(circle_size+1, circle_size+1))

    def create_mask_widget(self, mid):
        mask_frame = ctk.CTkFrame(self.mask_list_frame)
        mask_frame._default_fg_color = mask_frame.cget("fg_color")
        mask_frame.crc = ctk.CTkLabel(mask_frame, text="", image=self._crc(mid))
        mask_frame.crc.grid(row=0, column=0, padx=(10, 5), pady=5)
        mask_frame.lbl = ctk.CTkLabel(mask_frame, text=f"{mid}: {self.mask_labels[mid]}", anchor="w")
        mask_frame.lbl.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        mask_frame._default_text_color = mask_frame.lbl.cget("text_color")
        mask_frame.hide = ctk.CTkButton(mask_frame, text="",
                                        image=self.icons_dict["EyeOpen"]["normal"],
                                        width=34, height=34,
                                        fg_color="transparent",
                                        command=lambda: self.toggle_mask_hide(mid, not mask_frame.hidden))
        mask_frame.hide.grid(row=0, column=2, padx=(5,2), pady=5)
        mask_frame.hidden = False
        mask_frame.lock = ctk.CTkButton(mask_frame, text="",
                                        image=self.icons_dict["LockOpen"]["normal"],
                                        width=34, height=34,
                                        fg_color="transparent",
                                        command=lambda: self.toggle_mask_lock(mid, not mask_frame.locked))
        mask_frame.lock.grid(row=0, column=3, padx=2, pady=5)
        mask_frame.locked = False
        clear_btn = ctk.CTkButton(mask_frame, text="×",
                                  font=ctk.CTkFont(size=24, weight="bold"),
                                  width=34, height=34,
                                  fg_color="transparent",
                                  text_color="#AB2B22",
                                  command=lambda: self.clear_mask(mid))
        clear_btn.grid(row=0, column=4, padx=(2,5), pady=5)
        clear_btn.bind("<Enter>", lambda e: clear_btn.configure(fg_color="#AB2B22", text_color="white"))
        clear_btn.bind("<Leave>", lambda e: clear_btn.configure(fg_color="transparent", text_color="#AB2B22"))
        mask_frame.grid_columnconfigure(1, weight=1)
        mask_frame.bind("<Button-1>", lambda e: self.change_mask(mid))
        mask_frame.bind("<Button-3>", lambda e: self.update_mask(e, mid))
        mask_frame.crc.bind("<Button-1>", lambda e: self.change_mask(mid))
        mask_frame.crc.bind("<Button-3>", lambda e: self.update_mask(e, mid))
        mask_frame.lbl.bind("<Button-1>", lambda e: self.change_mask(mid))
        mask_frame.lbl.bind("<Button-3>", lambda e: self.update_mask(e, mid))
        if 1 <= mid <= 9:
            self.bind(f"<Key-{mid}>", lambda e: self.change_mask(mid))
        return mask_frame

    def change_mask(self, target_id=None):
        '''
        Changes the currently active mask based on the user selection in the 
        combo box.
        '''
        # Retrieves the mask ID corresponding to the current selection
        if self.active_mask_id:
            self.mask_widgets[self.active_mask_id].configure(border_width=0, fg_color=self.mask_widgets[self.active_mask_id]._default_fg_color)
            self.mask_widgets[self.active_mask_id].hide.configure(hover_color=ctk.ThemeManager.theme["CTkButton"]["hover_color"])
            self.mask_widgets[self.active_mask_id].lock.configure(hover_color=ctk.ThemeManager.theme["CTkButton"]["hover_color"])
        self.active_mask_id = target_id
        # Change appearance of mask row in mask list
        self.mask_widgets[target_id].configure(border_width=3, border_color=HIGHLIGHT_COLOR, fg_color=ctk.ThemeManager.theme["CTkButton"]["hover_color"])
        self.mask_widgets[self.active_mask_id].hide.configure(hover_color=self.mask_widgets[self.active_mask_id]._default_fg_color)
        self.mask_widgets[self.active_mask_id].lock.configure(hover_color=self.mask_widgets[self.active_mask_id]._default_fg_color)
        self.set_controls_state(True)
        self._draw_brush_preview(self.mouse['x'], self.mouse['y'])

    def update_mask(self, e, target_id):
        if hasattr(self, 'active_context_menu'):
            self.active_context_menu.destroy()

        context_menu = tk.Menu(self, tearoff=0)
        context_menu.add_command(label="Edit mask", command=lambda: self.edit_mask(target_id))
        context_menu.add_command(label="Set as active mask", command=lambda: self.change_mask(target_id))
        context_menu.add_command(label="Delete mask", command=lambda: self.clear_mask(target_id))
        context_menu.add_separator()
        context_menu.add_command(label="Close this menu", command=lambda: context_menu.destroy())
        context_menu.post(e.x_root, e.y_root)

        self.active_context_menu = context_menu
        self.set_menu_theme(self.active_context_menu, self.slimtag_config["main"]["appearance"])
    
    def edit_mask(self, target_id): 
        self.deactivate_tools()
        name, color = MaskEditDialog(self,
                                     initial_color=rgb_to_hex(self.mask_colors[target_id]),
                                     mask_name=self.mask_labels[target_id]
                                     ).get()
        
        # Update Widget
        if color is not None:
            self.mask_colors[target_id] = hex_to_rgb(color)
            self.mask_widgets[target_id].crc.configure(image=self._crc(target_id))
        if name != "":
            self.mask_labels[target_id] = name
            self.mask_widgets[target_id].lbl.configure(text=f"{target_id}: {self.mask_labels[target_id]}")
        self.update_display(update_image=False)
    
    def clear_mask(self, mid):
        '''
        Deletes all pixels of the mask with mask id mid, removes its label and 
        color, updates the combo box to reflect remaining masks, and refreshes 
        the display and color preview.
        '''
        # Check if an image is loaded
        if not self.image_is_loaded():
            return
        
        self.deactivate_tools()
        
        if self.mask_orig is None:
            return
        
        self.push_undo()
        self.mask_locked[self.mask_orig == mid] = False # free locks
        self.mask_orig[self.mask_orig == mid] = 0
        del self.mask_labels[mid]
        del self.mask_colors[mid]
        self.mask_widgets[mid].destroy()
        del self.mask_widgets[mid]
        
        if 1 <= mid <= 9:
            self.unbind(f"<Key-{mid}>")

        if self.active_mask_id == mid:
            self.active_mask_id = None
        
        if len(self.mask_labels) == 0 or self.active_mask_id is None: # disable all buttons if there are no masks
            self.set_controls_state(False)
        self.update_display(update_image=False)
    
    def clear_active_mask(self):
        if self.active_mask_id is None:
            return
        self.clear_mask(self.active_mask_id)
    
    def clear_all_masks(self):
        mask_ids = list(self.mask_labels.keys())
        for mid in mask_ids:
            self.clear_mask(mid)
        # TODO add warning
    
    def toggle_mask_hide(self, mid, set_hide: bool, update_display=True):
        # change mid mask hidden status to set_hide
        self.mask_widgets[mid].hidden = set_hide
        self.mask_widgets[mid].hide.configure(image=self.icons_dict["EyeClosed" if set_hide else "EyeOpen"]["normal"])
        # if all the statuses of the single masks are the same, change the "all" button as well
        all_statuses = set([self.mask_widgets[m].hidden for m in list(self.mask_labels.keys())])
        if len(all_statuses) == 1:
            self.toggle_all_masks_hide(list(all_statuses)[0], propagate=False)
        else:
            self.toggle_all_masks_hide(False, propagate=False)
        if update_display:
            self.update_display(update_image=False)
    
    def toggle_mask_lock(self, mid, set_lock: bool):
        # change mid mask locked status to set_lock
        self.mask_widgets[mid].locked = set_lock
        self.mask_widgets[mid].lock.configure(image=self.icons_dict["LockClosed" if set_lock else "LockOpen"]["normal"])
        self.mask_locked[self.mask_orig==mid] = set_lock
        # if all the statuses of the single masks are the same, change the "all" button as well
        all_statuses = set([self.mask_widgets[m].locked for m in list(self.mask_labels.keys())])
        if len(all_statuses) == 1:
            self.toggle_all_masks_lock(list(all_statuses)[0], propagate=False)
        else:
            self.toggle_all_masks_lock(False, propagate=False)
    
    def toggle_all_masks_hide(self, set_hide: bool, enabled=True, propagate=True):
        state = "normal" if enabled else "disabled"
        icon = "EyeClosed" if set_hide else "EyeOpen"
        self.hide_all_mask_btn.hidden = set_hide
        self.hide_all_mask_btn.configure(state=state, image=self.icons_dict[icon][state])
        if propagate:
            mask_ids = list(self.mask_labels.keys())
            for mid in mask_ids:
                self.toggle_mask_hide(mid, set_hide, update_display=False)
            self.update_display(update_image=False)

    def toggle_all_masks_lock(self, set_lock: bool, enabled=True, propagate=True):
        state = "normal" if enabled else "disabled"
        icon = "LockClosed" if set_lock else "LockOpen"
        self.lock_all_mask_btn.locked = set_lock
        self.lock_all_mask_btn.configure(state=state, image=self.icons_dict[icon][state])
        if propagate:
            mask_ids = list(self.mask_labels.keys())
            for mid in mask_ids:
                self.toggle_mask_lock(mid, set_lock)
    
    def update_lock(self):
        # update self.mask_locked with current locked masks
        if self.mask_orig is None:
            return
        self.mask_locked = np.full(self.mask_orig.shape, False)
        mask_ids = list(self.mask_labels.keys())
        for mid in mask_ids:
            if self.mask_widgets[mid].locked:
                self.mask_locked[self.mask_orig==mid] = True
