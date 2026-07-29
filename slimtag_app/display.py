"""Methods for the display responsibilities of SegmentationApp."""

import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk

from slimtag_app.constants import HIGHLIGHT_COLOR


class DisplayMixin:
    """Composable display behaviour for ``SegmentationApp``."""

    def show_preview_frame(self, preview):
        for nav in self.sub_canvas_frames:
            self.sub_canvas_frames[nav].grid_forget()
        # TODO implement for ortho
        # if hasattr(self.sub_canvas_frames[preview], "view1") ...
        if self.image_orig is not None:
            scale = max(self.orig_w, self.orig_h) / self.slimtag_config["view"]["preview_dim"]
            self.preview_scale = scale
            self.sub_canvas_frames["image"].canvas.configure(width=int(self.orig_w / scale), height=int(self.orig_h / scale))
            self.sub_canvas_image = ImageTk.PhotoImage(self.image_orig.resize((int(self.orig_w / scale), int(self.orig_h / scale)), Image.Resampling.LANCZOS))
            self.sub_canvas_frames["image"].canvas.create_image(0, 0, anchor="nw", image=self.sub_canvas_image, tag="image")
        self.current_preview_canvas = self.sub_canvas_frames["image"].canvas
        self.sub_canvas_frames[preview].grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.sub_canvas_frames[preview].tkraise()
    
    def update_preview_frame(self):
        self.current_preview_canvas.delete("rectangle")
        x = int(self.view_x / self.preview_scale)
        y = int(self.view_y / self.preview_scale)
        w = int(self.view_w / self.preview_scale)
        h = int(self.view_h / self.preview_scale)
        self.current_preview_canvas.create_rectangle(x, y, x+w, y+h, outline=HIGHLIGHT_COLOR, width=2, tag="rectangle")
    
    #%% UPDATE DISPLAY
    def update_display(self, update_image=True, update_blended=True):
        '''
        Aux method to update display whenever a change occurs.
        
        Valid argument for update_all:
            - "Global" updates both the background image and the mask overlay
            - "Mask" updates only the mask overlay
        '''
        if self.image_orig is None:
            return
        
        self.zoom_label_var.set(f"Zoom: {round(100*self.zoom)}%")
        
        # clean canvas if coming after pan & zoom events
        self.canvas.delete("preview_image")
        
        if update_image:
            # remove old info
            self.canvas.delete("background_image","mask")
            # create new image view and paste it on canvas
            self.image_disp = self.image_orig.crop([self.view_x, self.view_y, self.view_x+self.view_w, self.view_y+self.view_h]) \
                                             .resize((self.canvas.winfo_width(), self.canvas.winfo_height()), Image.NEAREST)
            self.tk_img = ImageTk.PhotoImage(self.image_disp)
            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img, tag="background_image")
        else:
            # delete only the mask to speed up computations
            self.canvas.delete("mask")
            
        # compute new mask view margins
        top = max(0, self.view_y)
        bottom = min(max(self.view_y+self.view_h, 0), self.orig_h)
        left = max(0,self.view_x)
        right = min(max(self.view_x+self.view_w,0), self.orig_w)
        
        self.cut_mask_orig = np.zeros((self.view_h, self.view_w), dtype=self.mask_orig.dtype)
        try:
            self.cut_mask_orig[top-self.view_y:bottom-self.view_y, left-self.view_x:right-self.view_x] = self.mask_orig[top:bottom, left:right]
        except ValueError: # in case we are out of image limits, in this case keep empty mask
            pass
        
        # resize new mask to canvas size (still in P mode for efficiency)
        resized = Image.fromarray(self.cut_mask_orig, mode="P").resize((self.canvas.winfo_width(), self.canvas.winfo_height()), Image.NEAREST)

        # create BINARY matrix to encode background & hidden masks
        # this is NOT an alpha channel with semi-transparency: too expensive
        hidden_values_list = [0] + [mid for mid in self.mask_colors if self.mask_widgets[mid].hidden]
        binary_mask = Image.fromarray((np.isin(np.array(resized), hidden_values_list)).astype("uint8") * 255, mode="L")

        # now we can convert resized to RGB
        palette = [0, 0, 0] * 256  # index 0 = black background
        for mid, color in self.mask_colors.items():
            palette[mid*3:mid*3+3] = list(color)
        resized.putpalette(palette)
        resized = resized.convert("RGB")
        
        # the trick: we put a UNIFORM alpha channel equal to the current mask opacity value
        # a "true" alpha channel is the computational bottlenck, but a single-value channel is fine.
        alpha = self.mask_opacity if len(self.sam_points) == 0 else (self.mask_opacity // 2)
        resized.putalpha(alpha)
        
        # now we compose the image:
        # - where binary_mask is 255, take from the FIRST image
        #   (in this case, the original image with full opacity)
        # - where binary_mask is 0, take from the SECOND image
        #   (so mask at mask_opacity)
        # this is fine since there is tk_img already drawn on the canvas
        resized = Image.composite(self.image_disp, resized, binary_mask)
        
        self.tk_ov = ImageTk.PhotoImage(resized)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_ov, tag="mask")
        
        if self.blended is None or update_blended:
            self.update_blended()

        # if SAM is active, create also multipoint preview
        if any(self.tool_active[tool] for tool in ["wand", "wand_multi"]):
            # create new mask view and populate
            cut_mask_preview = np.full((self.view_h, self.view_w), False)
            try:
                cut_mask_preview[top-self.view_y:bottom-self.view_y, left-self.view_x:right-self.view_x] = self.sam_preview[top:bottom, left:right]
            except ValueError: # in case we are out of image limits, in this case keep empty mask
                pass
            
            # TODO: if self.mask_outline.get(): change overlay as border only
            # but maybe not for SAM preview?
            # create overlay object and convert it to be pasted on canvas
            preview_alpha = max(min(int(self.mask_opacity + 0.35 * (255 - self.mask_opacity)), 255), 0)
            overlay_prev = np.zeros((self.view_h, self.view_w, 4), np.uint8)
            if not self.mask_widgets[self.active_mask_id].hidden:
                overlay_prev[cut_mask_preview] = [*self.mask_colors[self.active_mask_id], preview_alpha]
            self.sam_preview_pil = Image.fromarray(overlay_prev)
            resized_prev = self.sam_preview_pil.resize((self.canvas.winfo_width(), self.canvas.winfo_height()), Image.NEAREST)
            self.tk_sam_preview = ImageTk.PhotoImage(resized_prev)
            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_sam_preview, tag="mask")

            # raise back SAM multipoints if any
            self.display_wand_multipoints()
        
        # if brush or eraser is active, draw preview
        # there are some inconsistencies due to interactions with Shift and Ctrl, but whatever
        if self.tool_active["brush"] or self.tool_active["eraser"]:
            self._draw_brush_preview(self.mouse['x'], self.mouse['y'], shift_pressed=(self.shift_pressed or self.tool_active["eraser"]))

    def update_mask_opacity(self, v):
        self.mask_opacity = v
        self.update_display(update_image=False, update_blended=False)
        if self.update_opacity_id is not None:
            self.after_cancel(self.update_opacity_id)
        # if still resizing, schedule a new event
        self.update_opacity_id = self.after(300, lambda: self.update_display(update_image=False))
    
    def update_display_after_resize(self):
        self.view_h = int(self.canvas.winfo_height()/self.zoom)
        self.view_w = int(self.canvas.winfo_width()/self.zoom)
        self.update_display(update_image=True)
    
    def update_blended(self):
        """
        Update blended RGB image for fast pan & zoom
        """
        mask_disp = Image.fromarray(self.mask_orig, mode="P")
        palette = [0, 0, 0] * 256  # index 0 = black background
        for mid, color in self.mask_colors.items():
            palette[mid*3:mid*3+3] = list(color)
        mask_disp.putpalette(palette)
        mask_disp_rgb = mask_disp.convert("RGB")
        
        # create alpha mask
        hidden_values_list = [0] + [mid for mid in self.mask_colors if self.mask_widgets[mid].hidden]
        alpha = self.mask_opacity if len(self.sam_points) == 0 else (self.mask_opacity // 2)
        alpha_mask = Image.fromarray((1-np.isin(mask_disp, hidden_values_list).astype(np.uint8)) * alpha)
        mask_disp_rgb.putalpha(alpha_mask)
        
        # create composite image
        blended = Image.alpha_composite(self.image_orig.convert("RGBA"), mask_disp_rgb)
        if len(self.sam_points) > 0:
            # add preview image
            if not self.mask_widgets[self.active_mask_id].hidden: # if active mask is hidden, skip computation
                preview_alpha = max(min(int(self.mask_opacity + 0.35 * (255 - self.mask_opacity)), 255), 0)
                overlay_prev = np.zeros((self.orig_h, self.orig_w, 4), np.uint8)
                overlay_prev[self.sam_preview] = [*self.mask_colors[self.active_mask_id], preview_alpha]
                blended = Image.alpha_composite(blended, Image.fromarray(overlay_prev))
        self.blended = blended.convert("RGB")
    
    def display_blended(self):
        """
        Show precomputed preview during pan & zoom events
        """
        self.canvas.delete("background_image","mask")
        blended = self.blended.crop([self.view_x, self.view_y, self.view_x+self.view_w, self.view_y+self.view_h]) \
                      .resize((self.canvas.winfo_width(), self.canvas.winfo_height()), Image.NEAREST)
        self.tk_img = ImageTk.PhotoImage(blended)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img, tag="preview_image")
        self.display_wand_multipoints()
    
    def display_wand_multipoints(self):
        self.canvas.delete("sam_pt")
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        for i in range(len(self.sam_points)):
            x = int(((self.sam_points[i][0] - self.view_x) / self.view_w) * cw)
            y = int(((self.sam_points[i][1] - self.view_y) / self.view_h) * ch)
            if 0 <= x <= cw and 0 <= y <= ch:
                if self.sam_pt_labels[i] == 1:
                    # fg point: fill with active mask color, outline black
                    pt_fill = "#" + "".join([f"{c:02x}" for c in self.mask_colors[self.active_mask_id]])
                    pt_out = "black"
                else: # label == 0
                    # bg point: fill black, use the inverted active mask color for outline
                    pt_fill = "black"
                    pt_out = "#" + "".join([f"{255-c:02x}" for c in self.mask_colors[self.active_mask_id]])
                self.canvas.create_oval(x-3, y-3, x+3, y+3, fill=pt_fill, outline=pt_out, width=1, tag="sam_pt")
        self.canvas.tag_raise("sam_pt")
    
    #%% UI CANVAS METHODS
    def show_canvas_frame(self, frametype):
        # unbind events to currently visible canvas
        if self.canvas is not None:
            self.canvas.unbind("<Button-1>")
            self.canvas.unbind("<Button-2>")
            self.canvas.unbind("<Button-3>")
            self.canvas.unbind("<Button-4>")
            self.canvas.unbind("<Button-5>")
            self.canvas.unbind("<MouseWheel>")
            self.canvas.unbind("<Motion>")
            self.canvas.unbind("<B1-Motion>")
            self.canvas.unbind("<B2-Motion>")
            self.canvas.unbind("<B3-Motion>")
            self.canvas.unbind("<ButtonRelease-1>")
            self.canvas.unbind("<ButtonRelease-2>")
            self.canvas.unbind("<ButtonRelease-3>")
        # change canvas
        frame = self.canvas_frames[frametype]
        self.canvas = frame.canvas
        self.reset_bbox()
        if frametype == "volume":
            self.volume_zslider = frame.slider
        # bind events to newly visible canvas
        self.canvas.bind("<Button-1>", self.on_canvas_left)
        self.canvas.bind("<Button-2>", self.on_canvas_mid)
        self.canvas.bind("<Button-3>", self.on_canvas_right)
        self.canvas.bind("<Button-4>", self.wheel_up) # <Button-4> is scroll up for Linux
        self.canvas.bind("<Button-5>", self.wheel_down) # <Button-5> is scroll down for Linux
        self.canvas.bind("<MouseWheel>", self.wheel_evt)
        self.canvas.bind("<Motion>", self.draw_brush_preview, add="+")
        self.canvas.bind("<Motion>", self.on_canvas_track, add="+")
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<B2-Motion>", self.on_canvas_drag)
        self.canvas.bind("<B3-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_left_release)
        self.canvas.bind("<ButtonRelease-2>", self.on_canvas_mid_release)
        self.canvas.bind("<ButtonRelease-3>", self.on_canvas_right_release)
        frame.tkraise()
    
    def set_volume_slice(self, z):
        if not self.is_volume_loaded:
            return
        
        img = Image.fromarray(self.volume_disp[..., z]).convert("RGB")
        self.load_image(img, mask=self.volume_mask[..., z], reset_view=False) # don't change canvas, don't reset view

    def on_zslider_move(self, z):
        '''
        Update slider preview when moving.
        
        This is the argument of 'command=' of the slider.
        
        z is already cast to int.
        '''
        self.zlabel_var.set(f"z: {z}")
        if self.zslider_preview is not None:
            self.update_zslider_preview(z)

    def update_zslider_preview(self, z):
        '''
        Function that actually changes the image in the preview depending on z value.
        
        z is the slider value, already cast to int.
        '''
        arr = self.volume_preview[:, :, z]
        self.zslider_preview_img = ImageTk.PhotoImage(Image.fromarray(arr))
        self.zslider_preview.canvas.delete("all")
        self.zslider_preview.canvas.create_image(0, 0, anchor="nw", image=self.zslider_preview_img)

    def update_zslider_preview_position(self):
        '''
        Update preview window position based on knob position and slider dimensions.
        '''
        self.update_idletasks()
        # recover slider dimensions
        slider_x = self.volume_zslider.winfo_rootx()
        slider_y = self.volume_zslider.winfo_rooty()
        slider_w = self.volume_zslider.winfo_width()
        slider_h = self.volume_zslider.winfo_height()
        # get fraction of slider length corresponding to knob position
        min_val = self.volume_zslider.cget("from_")
        max_val = self.volume_zslider.cget("to")
        frac = (self.volume_zslider.get() - min_val) / (max_val - min_val)
        # compute actual position
        knob_radius = slider_h / 2
        usable_width = slider_w - 2 * knob_radius
        # x position
        preview_xc = slider_x + knob_radius + frac * usable_width # x of center
        preview_w = self.zslider_preview.winfo_width()
        x = int(preview_xc - preview_w / 2)
        # y position
        preview_h = self.zslider_preview.winfo_height()
        y = slider_y - (preview_h + 20) # padding of 20 px
        self.zslider_preview.geometry(f"+{x}+{y}")

    def start_zlider_preview(self, event):
        '''
        Function executed at <Button1> event on the slider.
        
        It creates the TopLevel object containing the preview.
        '''
        # create TopLevel if it does not exist
        if self.zslider_preview is None:
            self.zslider_preview = ctk.CTkToplevel(self)
            self.zslider_preview.overrideredirect(True) # remove window decorations
            self.zslider_preview.attributes("-topmost", True) # shadow-like appearance
            self.zslider_preview.canvas = ctk.CTkCanvas(self.zslider_preview, highlightthickness=0,
                                                        width=self.volume_preview.shape[1], height=self.volume_preview.shape[0])
            self.zslider_preview.canvas.grid(row=0, column=0, padx=4, pady=4)
            self.zslider_preview.grid_rowconfigure(0, weight=1)
            self.zslider_preview.grid_columnconfigure(0, weight=1)
            
        # call this once to set initial values
        self.move_zslider_preview(event)
    
    def move_zslider_preview(self, event):
        '''
        Function executed at <Button1-Motion> event on the slider.
        
        It moves the preview window and updates its content.
        '''
        z = round(self.volume_zslider.get())
        self.update_zslider_preview(z)
        self.update_zslider_preview_position()
    
    def end_zslider_preview(self, event):
        '''
        Function executed at <Button1-Release> event on the slider.
        
        It updates the main canvas and destroys the preview window.
        
        z is the slider value, already cast to int.
        '''
        z = round(self.volume_zslider.get())
        self.set_volume_slice(z)
        # destroy preview
        if self.zslider_preview is not None:
            self.zslider_preview.destroy()
            self.zslider_preview = None
