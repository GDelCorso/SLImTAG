"""Methods for the tools responsibilities of SegmentationApp."""

import math
import threading

import numpy as np
from scipy import ndimage

import slimtag_wand as wand
from slimtag_bayesian import OptimizerDialog
from slimtag_utils import PreprocessingAdjustments


class ToolsMixin:
    """Composable tools behaviour for ``SegmentationApp``."""

    def brush_at(self, x, y, add=True):
        '''
        Aux method to define brush position without updating display or undo.
        '''
        # Return immediately if no mask or no active label
        if self.mask_orig is None or self.active_mask_id is None:
            return
        
        # Brush radius
        r = self.brush_size // 2
        buffer = max(1, r // 2)
        
        # Define bounding box of the brush, clamped to image edges
        y0 = max(0, y - r - buffer)
        y1 = min(self.mask_orig.shape[0], y + r + buffer)
        x0 = max(0, x - r - buffer)
        x1 = min(self.mask_orig.shape[1], x + r + buffer)
        
        # Create small local coordinate arrays (only the bounding box, not the whole mask)
        ys = np.arange(y0, y1)
        xs = np.arange(x0, x1)
        
        # Efficient broadcasting to create mask
        dy = ys[:, None] - y  # shape (height, 1)
        dx = xs[None, :] - x  # shape (1, width)
        
        match self.brush_shape:
            case 'Circle':
                effective_mask_area = dx**2 + dy**2 <= r*r + 4 # boolean array, shape (y1-y0, x1-x0)
            case 'Square':
                theta = np.radians(self.brush_rot)  
                dx_rot = dx * np.cos(theta) + dy * np.sin(theta)
                dy_rot = -dx * np.sin(theta) + dy * np.cos(theta)
                effective_mask_area = (np.abs(dx_rot) <= r) & (np.abs(dy_rot) <= r)
            case 'Line':
                theta = np.radians(self.brush_rot)  
                dx_rot = dx * np.cos(theta) + dy * np.sin(theta)
                dy_rot = -dx * np.sin(theta) + dy * np.cos(theta)
                effective_mask_area = (np.abs(dx_rot) <= r // self.brush_line_ratio) & (np.abs(dy_rot) <= r ) # TODO check rotation, something wrong with line

        # Slice of the mask corresponding to the bounding box
        mask_area = self.mask_orig[y0:y1, x0:x1]
        lock_area = self.mask_locked[y0:y1, x0:x1]
        
        if add:
            # Paint only on non-locked pixels
            mask_area[effective_mask_area & (~lock_area)] = self.active_mask_id
        else:
            # Erase only pixels that match the active mask label
            # (independently from their locked status)
            erase_mask = effective_mask_area & (mask_area == self.active_mask_id)
            mask_area[erase_mask] = 0
        
        # Update locked status
        # only on slices for performance
        lock_area[mask_area==self.active_mask_id] = self.mask_widgets[self.active_mask_id].locked
        lock_area[mask_area==0] = False
        # Mark mask as modified for later saving or GUI update
        self.set_modified(True)

    def draw_bbox_preview(self, e):
        if self.mask_orig is None or self.active_mask_id is None or self.bbox[0] is None:
            return
        self.canvas.delete("bbox")
        outline_color = "#" + "".join([f"{c:02x}" for c in self.mask_colors[self.active_mask_id]])
        self.canvas.create_rectangle(self.bbox[0][0], self.bbox[0][1], e.x, e.y, fill=outline_color, outline='', width=0, tag="bbox")

    def draw_brush_preview(self, e):
        '''
        Draws a semi-transparent mask contour on the canvas to show the brush size
        and position before painting. The contour is solid in 'add mask' mode
        and dashed in 'remove mask' mode.
        '''
        
        if hasattr(e, 'x'): self.mouse['x'], self.mouse['y'] = e.x, e.y # store mouse position
        x, y = self.mouse['x'], self.mouse['y']

        shift_pressed = (e.state & 0x0001) != 0 or self.b3_pressed if hasattr(e , 'state') else False
        self._draw_brush_preview(x, y, shift_pressed)

    def _draw_brush_preview(self, x, y, shift_pressed=False):
        self.canvas.delete("brush")
        
        if not (self.tool_active["brush"] or self.tool_active["eraser"]):
            return

        r = int(self.brush_size * self.zoom / 2)
        outline_color = "#" + "".join([f"{c:02x}" for c in self.mask_colors[self.active_mask_id]])
        dash = (5,10) if (shift_pressed or self.tool_active["eraser"]) else None

        match self.brush_shape:
            case 'Circle':
                self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="", outline=outline_color, dash=dash, width=2, tag="brush")
            case 'Square':
                self._draw_not_oval_brush([x-r, y-r, x+r, y-r, x+r, y+r, x-r, y+r], (x,y), outline_color, dash)
            case 'Line':
                self._draw_not_oval_brush([x-r // self.brush_line_ratio, y-r, x+r // self.brush_line_ratio, y-r, x+r // self.brush_line_ratio, y+r, x-r // self.brush_line_ratio, y+r], (x,y), outline_color, dash)
                
    def _draw_not_oval_brush(self, points, pivot, outline_color, dash):
        points = self._rotate_points(points, self.brush_rot, pivot)
        for i in range(0, len(points), 2):
            x0, y0 = points[i], points[i+1]
            x1, y1 = points[i+2 if i+2 <len(points) else 0], points[i+3 if i+3 <len(points) else 1]
            self.canvas.create_line(x0, y0, x1, y1, fill=outline_color, dash=dash, width=2, tag="brush")

    def brush_rotate(self, e): # bound to wheel event
        if self.brush_shape == 'Circle':
            return
        self.brush_rot = (self.brush_rot + e.delta) % 180
        self.update_brush_rotation_slider()
        self.draw_brush_preview(e)

    def _rotate_points(self, points, angle_deg, pivot):
        cx, cy = pivot
        a = math.radians(angle_deg)
        c, s = math.cos(a), math.sin(a)
        out = []
        for i in range(0, len(points), 2):
            x, y = points[i] - cx, points[i+1] - cy
            xr, yr = x*c - y*s + cx, x*s + y*c + cy
            out.extend([xr, yr])
        return out
    
    # SAM
    def sam_add_point(self, e, add=True, multipoint=False):
        """
        Add the clicked point to the list of points to be fed to SAM.
        
        If multipoint=False, add=True means "use only this point to compute the
        mask, and add it", while add=False means "use only this point to
        compute the mask, and remove it".
        If multipoint=True, add=True means "mark this point as foreground",
        while add=False means "mark this point as background"
        """
        if self.image_orig is None or self.active_mask_id is None:
            return
        x = int((e.x)*(self.view_w/self.canvas.winfo_width())) + self.view_x
        y = int((e.y)*(self.view_h/self.canvas.winfo_height())) + self.view_y
        self.sam_points.append([x, y])
        if multipoint:
            self.sam_pt_labels.append(1 if add else 0)
            self.sam_compute(multipoint=True)
        else:
            self.sam_pt_labels.append(1)
            self.sam_compute(multipoint=False)
            self.sam_apply(add=add)

    def sam_compute(self, multipoint=False):
        """
        Use SAM points and labels lists to compute mask, and store it in the
        preview matrix.
        
        multipoint determines if one or three masks are computed
        """
        if (self.image_orig is None) or (self.active_mask_id is None) or (not self.sam_points):
            return
        self.set_status("loading", "SAM computing...")
        
        # image = None, since preprocessing embedded image in model
        mask = wand.sam_inference(None,
                                  point=np.array(self.sam_points),
                                  pt_labels=np.array(self.sam_pt_labels),
                                  parameters={"threshold": self.wand_threshold},
                                  model=self.sam,
                                  multipoint=multipoint)
        self.sam_preview[mask & (~self.mask_locked)] = True
        
        if multipoint: # to show preview
            self.update_display(update_image=False)
        self.set_status("ready", "Ready")

    def sam_apply(self, add=True, cancel=False):
        """
        Apply the mask in SAM_preview to definitive mask, and empty SAM points
        and labels lists.
        
        If cancel=True, don't apply the computed mask and only empty SAM infos.
        """
        if self.image_orig is None or self.active_mask_id is None:
            return
        if not cancel:
            self.push_undo()
            if add:
                self.mask_orig[self.sam_preview] = self.active_mask_id
            else:
                self.mask_orig[self.sam_preview & (self.mask_orig==self.active_mask_id)] = 0
            self.set_modified(True)
        self.sam_preview = np.full(self.mask_orig.shape, False) # reset preview
        self.sam_points = []
        self.sam_pt_labels = []
        self.canvas.delete("sam_pt")
        # Update locked status (we don't use self.update_lock() for performances)
        self.mask_locked[self.mask_orig==self.active_mask_id] = self.mask_widgets[self.active_mask_id].locked
        self.mask_locked[self.mask_orig==0] = False
        self.update_display(update_image=False)

    def sam_apply_release(self):
        """
        Event bound to the release of the "Multipoint" key
        """
        # TODO check other active tools
        if (not self.tool_active["wand"]) or (not self.sam_points): # empty lists are false
            return
        self.sam_apply(add=True) # multipoint only adds mask

    def manual_wand_preprocessing(self):
        values = PreprocessingAdjustments(self).values
        if values is not None:
            self.wand_brightness, self.wand_contrast, self.wand_gamma = values
            self.wand_brightness_lbl.configure(text=str(self.wand_brightness))
            self.wand_contrast_lbl.configure(text=str(self.wand_contrast))
            self.wand_gamma_lbl.configure(text=str(self.wand_gamma))
            # reload SAM image
            # deactivate tools
            self.deactivate_tools()
            if self.thread is None or not self.thread.is_alive():
                self.thread = threading.Thread(target=self.async_loader, daemon=True)
                self.thread.start()
             
    # NON-NEURAL METHODS
    # SCIPY REGION GROWING
    def region_growing(self, e):

        if self.image_orig is None or self.active_mask_id is None:
            return
        
        self.set_status("loading", "Applying region growing...")
        
        # Map coordinates
        x = int((e.x) * (self.view_w / self.canvas.winfo_width())) + self.view_x
        y = int((e.y) * (self.view_h / self.canvas.winfo_height())) + self.view_y
        
        params = {"threshold": self.wand_threshold, "grad_edge": self.wand_edge_tolerance}
            
        # image = None, since all the info needed is in preprocessing
        region = wand.region_growing_inference(None, [x, y], parameters=params, preprocessing=self.region_growing_preprocess)
        
        # update history, apply to mask and update display
        self.push_undo()
        
        self.mask_orig[region & (~self.mask_locked)] = self.active_mask_id
        self.set_modified(True)
        self.update_display(update_image=False)
        self.set_status("ready", "Ready")
        
    # CONNECTED COMPONENT
    def get_connected_component(self, mask, start_y, start_x, target_id):
        '''
        Computes and returns a boolean mask of all pixels connected to a 
        starting point that belong to the given mask ID, using a depth-first 
        search.
        '''
        if mask[start_y, start_x] != target_id:
            return np.zeros_like(mask, dtype=bool)
        h, w = mask.shape
        visited = np.zeros_like(mask, dtype=bool)
        component = np.zeros_like(mask, dtype=bool)
        stack = [(start_y, start_x)]
        while stack:
            y, x = stack.pop()
            if visited[y, x]:
                continue
            visited[y, x] = True
            if mask[y, x] == target_id:
                component[y, x] = True
                for ny, nx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                    if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                        stack.append((ny,nx))
        return component

    def connected_component_click(self, e, remove_only=True):
        '''
        Handles a click for the connected component tool, removing either only 
        the clicked component or all other pixels of the active mask, saving 
        the previous state for undo, and updating the display.
        '''
        if self.mask_orig is None or self.active_mask_id is None:
            return
        
        x = int((e.x)*(self.view_w/self.canvas.winfo_width())) + self.view_x
        y = int((e.y)*(self.view_h/self.canvas.winfo_height())) + self.view_y
        
        comp = self.get_connected_component(self.mask_orig, y, x, self.active_mask_id)
        if not comp.any():
            return
        
        self.push_undo()
        
        # notice that "connected component" only acts on active mask,
        # so the lock check is not needed
        if remove_only:
            self.mask_orig[comp] = 0
        else:
            self.mask_orig[(self.mask_orig==self.active_mask_id) & (~comp)] = 0
        
        self.set_modified(True)
        # Update locked status (we don't use self.update_lock() for performances)
        self.mask_locked[self.mask_orig==self.active_mask_id] = self.mask_widgets[self.active_mask_id].locked
        self.mask_locked[self.mask_orig==0] = False
        self.update_display(update_image=False)
        
    def fill_connected_component(self, e):
        '''
        Fill connected component
        '''
        
        if self.mask_orig is None or self.active_mask_id is None:
            return
        
        x = int((e.x) * (self.view_w / self.canvas.winfo_width())) + self.view_x
        y = int((e.y) * (self.view_h / self.canvas.winfo_height())) + self.view_y
        
        self.set_status("loading", "Applying filling...")
        
        # determine clicked connected component of active mask
        comp = self.get_connected_component(self.mask_orig, y, x, self.active_mask_id)
        if not comp.any():
            self.set_status("ready", "Ready")
            return
         
        # save for undo
        self.push_undo()
         
        # fill internal holes
        filled_comp = ndimage.binary_fill_holes(comp)
        
        # Assign active mask label
        self.mask_orig[filled_comp & (~self.mask_locked)] = self.active_mask_id
        
        # post-fill adjustments
        self.set_modified(True)
        # Update lock status
        self.mask_locked[self.mask_orig == self.active_mask_id] = self.mask_widgets[self.active_mask_id].locked
        self.mask_locked[self.mask_orig == 0] = False
        self.update_display(update_image=False)
        self.set_status("ready", "Ready")
        
    # SMOOTHING (EROSION + DILATION)
    def apply_smoothing(self, y, x, operation="dilation", size=3):
        '''
        Applies dilation or erosion to the connected component under the 
        clicked point, saving the previous state for undo, updating the mask 
        with the smoothed result, and refreshing the display.
        '''
        if self.mask_orig is None or self.active_mask_id is None:
            return
        
        self.set_status("loading", "Applying smoothing...")
        # Identify the connected component
        comp = self.get_connected_component(self.mask_orig, y, x, self.active_mask_id)
        if not comp.any():
            self.set_status("ready", "Ready")
            return
    
        self.push_undo()
    
        struct = np.ones((size, size), dtype=bool)
        
        comp_smooth = comp.copy()
        
        if operation == "dilation":
            for _ in range(self.smooth_iter):
                if self.smooth_n_erosions > 0:
                    comp_smooth = ndimage.binary_erosion(comp_smooth, structure=struct, iterations=self.smooth_n_erosions)
                if self.smooth_n_dilations > 0:
                    comp_smooth = ndimage.binary_dilation(comp_smooth, structure=struct, iterations=self.smooth_n_dilations)
        elif operation == "erosion":
            for _ in range(self.smooth_iter):
                if self.smooth_n_dilations > 0:
                    comp_smooth = ndimage.binary_dilation(comp_smooth, structure=struct, iterations=self.smooth_n_dilations)
                if self.smooth_n_erosions > 0:
                    comp_smooth = ndimage.binary_erosion(comp_smooth, structure=struct, iterations=self.smooth_n_erosions)
        else:
            return

        
        self.mask_orig[comp] = 0
        # (comp|(~self.mask_locked)) means: "during erosion, allow changes on
        # the old component even if active mask is locked"
        self.mask_orig[comp_smooth & (comp|(~self.mask_locked))] = self.active_mask_id
        
        self.set_modified(True)
        # Update locked status (we don't use self.update_lock() for performances)
        self.mask_locked[self.mask_orig==self.active_mask_id] = self.mask_widgets[self.active_mask_id].locked
        self.mask_locked[self.mask_orig==0] = False
        self.update_display(update_image=False)
        self.set_status("ready", "Ready")
    
    #%% BAYESIAN OPTIMIZATION
    def wand_update_bayesian(self):
        if not self.mask_labels:
            return
        OptimizerDialog(self)
