"""Methods for the events responsibilities of SegmentationApp."""

import time

import numpy as np

from slimtag_utils import MultiButtonDialog


class EventsMixin:
    """Composable events behaviour for ``SegmentationApp``."""

    def on_canvas_left(self, e):
        '''
        Handles left-clicks on the canvas, performing the active tool's action 
        (brush, magic wand, connected component, or smoothing) and using Shift 
        to modify behaviour.
        '''
        shift_pressed = (e.state & 0x0001) != 0 or self.b3_pressed
        ctrl_pressed = (e.state & 0x0004) != 0
        self._prev_brush_pos = None
        
        if not any(self.tool_active[tool] for tool in self.tool_active):
            self._pan_start = (e.x, e.y, self.view_x, self.view_y)
            return

        
        # Check position
        x_check = int((e.x)*(self.view_w/self.canvas.winfo_width())) + self.view_x
        y_check = int((e.y)*(self.view_h/self.canvas.winfo_height())) + self.view_y
        
        if (x_check < 0) or (x_check > self.orig_w) or (y_check < 0) or (y_check > self.orig_h):
            check_inside_image = False
        else:
            check_inside_image = True
        
        if self.tool_active["bbox"] and self.mid_pressed == False:
            if self.bbox[0] is None:
                self.bbox[0] = (e.x,e.y)
            return

        if self.tool_active["smooth"] and check_inside_image:
            x = int((e.x)*(self.view_w/self.canvas.winfo_width())) + self.view_x
            y = int((e.y)*(self.view_h/self.canvas.winfo_height())) + self.view_y
            op = "erosion" if shift_pressed else "dilation"
            self.apply_smoothing(y, x, operation=op)
            return
    
        if (self.tool_active["cut"] or self.tool_active["clean"]) and check_inside_image:
            self.connected_component_click(e, remove_only=(self.tool_active["cut"] and not shift_pressed))
            return
        
        if self.tool_active["fill"] and check_inside_image:
            self.fill_connected_component(e)
            return
        
        if self.tool_active["wand"] and check_inside_image: # TODO implement with match
            if self.wand_model_menu.get() == "Region growing":
                self.region_growing(e)
            elif self.wand_model_menu.get() in self.available_sam_models:
                self.sam_add_point(e, add=not shift_pressed, multipoint=ctrl_pressed)
            else:
                return
            return
        
        if self.tool_active["wand_multi"] and check_inside_image: # TODO implement with match
            if self.wand_model_menu.get() == "Region growing":
                MultiButtonDialog(self, message="Multipoint magic wand currently implemented for SAM models only", buttons=[("OK", None)])
                return
            elif self.wand_model_menu.get() in self.available_sam_models:
                self.sam_add_point(e, add=not shift_pressed, multipoint=True)
            else:
                return
            return
        
        if (self.tool_active["brush"] or self.tool_active["eraser"]) and check_inside_image:
            x = int((e.x)*(self.view_w/self.canvas.winfo_width())) + self.view_x
            y = int((e.y)*(self.view_h/self.canvas.winfo_height())) + self.view_y
            self.push_undo()
            self.brush_at(x, y, add=(self.tool_active["brush"] and not shift_pressed))
            self.update_display(update_image=False)
            self.draw_brush_preview(e)
            return

        
    def on_canvas_mid(self, e):
        self.mid_pressed = True
        self._pan_start = (e.x, e.y, self.view_x, self.view_y)
        self.reset_bbox()
   
    def on_canvas_mid_release(self, e):
        self.mid_pressed = False
        self._pan_start = None
        self.update_display(update_image=True)
        self.draw_brush_preview(e)

    def on_canvas_left_release(self, e):
        if self.tool_active["bbox"] and self.bbox[0] is not None:
            if self.bbox[1] is None:
                self.bbox[1] = (e.x,e.y)
            
            # Define the brush drag
            x0 = int((self.bbox[0][0])*(self.view_w/self.canvas.winfo_width())) + self.view_x
            y0 = int((self.bbox[0][1])*(self.view_h/self.canvas.winfo_height())) + self.view_y
            x1 = int((self.bbox[1][0])*(self.view_w/self.canvas.winfo_width())) + self.view_x
            y1 = int((self.bbox[1][1])*(self.view_h/self.canvas.winfo_height())) + self.view_y
            

            self.reset_bbox()

            #self.push_undo()            
            self.bbox_at([[x0, y0], [x1 ,y1]])

        self.last_brush_pos = None
        self._pan_start = None
        self._drag_counter = 0
        self.update_display(update_image=True)
        self.draw_brush_preview(e)
    
    def bbox_at(self, p):
        for i in range(len(p)):
            if p[i][0] < 0:
                p[i][0] = 0

            if p[i][0] > self.mask_orig.shape[0]:
                p[i][0] = self.mask_orig.shape[0]

            if p[i][1] < 0:
                p[i][1] = 0

            if p[i][1] > self.mask_orig.shape[1]:
                p[i][1] = self.mask_orig.shape[1]

        self.push_undo()

        p = np.array(p)    
        x0, y0 = p.min(axis=0)
        x1, y1 = p.max(axis=0)
        
        self.mask_orig[y0:y1, x0:x1] = self.active_mask_id

    def reset_bbox(self):
        self.bbox = [None, None]
        self.canvas.delete("bbox")
        

    def on_canvas_right(self, e):
        '''
        Handles right-clicks on the canvas, applying the active tool's removal 
        or erosion action without toggling tools.
        '''
        self.b3_pressed = True
        self.on_canvas_left(e)

    def on_canvas_right_release(self, e):
        self.b3_pressed = False
        self.on_canvas_left_release(e)

    def on_canvas_drag(self, e):
        '''
        Updates the brush continuously while dragging the mouse.
        Draws one circle per event, using add/subtract depending on Shift.
        No interpolation between points to avoid undesired smoothing.
        '''
        if self.image_orig is None:
            return
        
        shift_pressed = (e.state & 0x0001) != 0 or self.b3_pressed
        
        # Move the canvas if not tools selected
        if not any(self.tool_active[tool] for tool in self.tool_active) or self.mid_pressed:
            if self._pan_start is not None:
                x0, y0, ox0, oy0 = self._pan_start
                self.view_x = ox0 -int((e.x - x0)*(self.view_w/self.canvas.winfo_width()))
                self.view_y = oy0- int((e.y - y0)*(self.view_h/self.canvas.winfo_height()))
                self.display_blended()
                self.update_preview_frame()
            return
        
        # Check if the brush is not active (only draggable tool)
        # TODO implement other tools
        if not (self.tool_active["brush"] or self.tool_active["eraser"] or self.tool_active["bbox"]):
            return
        
        # Define the brush drag
        x1 = int((e.x)*(self.view_w/self.canvas.winfo_width())) + self.view_x
        y1 = int((e.y)*(self.view_h/self.canvas.winfo_height())) + self.view_y
        
        if(self.tool_active['bbox']):
            self.draw_bbox_preview(e)
            return

        if not hasattr(self, "_prev_brush_pos") or self._prev_brush_pos is None:
            self._prev_brush_pos = (x1, y1)
            # If brush is active, remove the starting point generated by first click event
            if self.tool_active["brush"]:
                self.undo()
            self.push_undo()
            self.brush_at(x1, y1, add=(self.tool_active["brush"] and not shift_pressed))
            self.update_display(update_image=False, update_blended=False)
            self.draw_brush_preview(e)
            
            return
        
        # Skip some updates when zooming
        now = time.monotonic()
        
        if not hasattr(self, "_last_brush_update"):
            self._last_brush_update = 0.0
        
        if now - self._last_brush_update >= self.slimtag_config["view"]["refresh_rate_brush"]:
            x0, y0 = self._prev_brush_pos
            dx = x1 - x0
            dy = y1 - y0
            dist = max(1, int(np.hypot(dx, dy))) # Distance between previous and current point (in pixel)
            r = max(1, self.brush_size // 2)
            steps = self.brush_line_ratio * 20 if self.brush_shape == 'Line' else max(3, dist*3 // r) # draw this number of mask shape along (x0, y0) and (x1, y1)
            for i in np.linspace(0, dist + 1, steps):
                xi = int(x0 + dx * i / dist)
                yi = int(y0 + dy * i / dist)
                self.brush_at(xi, yi, add=(self.tool_active["brush"] and not shift_pressed))

            self.update_display(update_image=False, update_blended=False) # update only mask
            self.draw_brush_preview(e)
            self._last_brush_update = now
            self._prev_brush_pos = (x1, y1)
    
    def on_canvas_track(self, e):
        '''
        Update label in statusbar depending on mouse position
        '''
        if self.image_orig is None:
            return
        
        x1 = int((e.x)*(self.view_w/self.canvas.winfo_width())) + self.view_x
        y1 = int((e.y)*(self.view_h/self.canvas.winfo_height())) + self.view_y
        
        self.pos_label_var.set(f"| x: {x1} | y: {y1} |")

    def wheel_evt(self, e):
        ctrl_pressed = (e.state & 0x0004) != 0
        if ((self.tool_active['brush'] or self.tool_active['eraser']) and ctrl_pressed):
            return self.brush_rotate(e)
        self.zoom_evt(e)

    def wheel_up(self, e):
        ctrl_pressed = (e.state & 0x0004) != 0
        if ((self.tool_active['brush'] or self.tool_active['eraser']) and ctrl_pressed):
            e.delta = self.brush_rot_delta
            return self.brush_rotate(e)
        self.zoom_in(e)

    def wheel_down(self, e):
        ctrl_pressed = (e.state & 0x0004) != 0
        if ((self.tool_active['brush'] or self.tool_active['eraser']) and ctrl_pressed):
            e.delta = -self.brush_rot_delta
            return self.brush_rotate(e)
        self.zoom_out(e)

    #%% KEYBOARD EVENTS
    def shift_key_pressed(self):
        self.shift_pressed = True
        self._draw_brush_preview(self.mouse['x'], self.mouse['y'], True)
    
    def shift_key_released(self):
        self.shift_pressed = False
        self._draw_brush_preview(self.mouse['x'], self.mouse['y'])
    # def shiftPressed(self):
    #     # in case brush is active, set preview to "dashed"
    #     self._draw_brush_preview(self.mouse['x'], self.mouse['y'], True)

    # def shiftReleased(self):
    #     # in case brush is active, set preview to "solid"
    #     self._draw_brush_preview(self.mouse['x'], self.mouse['y'])

    def tab(self):
        return self._tab(-1, 0, 1)

    def shift_tab(self):
        return self._tab(0, -1, -1)

    def _tab(self, id_key_to_check, id_key_to_get, increment):
        """
        Use TAB to cycle through masks
        """
        if len(self.mask_labels) == 0: # if there are no mask, do nothing
            return

        keys = list(self.mask_labels.keys())

        if len(self.mask_labels) == 1: 
            self.change_mask(keys[0])
            return

        if self.active_mask_id == keys[id_key_to_check]:
            self.change_mask(keys[id_key_to_get])
            return

        newIndex = keys.index(self.active_mask_id) + increment
        self.change_mask(keys[newIndex])

    #%% WINDOW EVENTS
    def on_resize(self, e):
        """
        Redraw canvas after window resize
        """
        if e.widget is self: # prevent firing during other events
            # while resizing, cancel the scheduled update_display event
            if self.resizing_event is not None:
                self.after_cancel(self.resizing_event)
            # if still resizing, schedule a new event
            self.resizing_event = self.after(300, self.update_display_after_resize)
    
    #%% PAN & ZOOM
    def pan_view(self, dx, dy):
        '''
        Pan view when distance is fixed. Used e.g. to bind keyboard arrows
        '''
        if self.image_orig is None:
            return
        self.view_x += dx
        self.view_y += dy
        self.update_display(update_image=True)
        self.update_preview_frame()
    
    def zoom_evt(self, e):
        '''
        Adjusts the zoom level of the displayed image based on mouse wheel 
        input and refreshes the display.
        '''
        if e.delta > 0:
            self.zoom_in(e)
        else:
            self.zoom_out(e)
        
    def zoom_in(self, e=None):
        '''
        Adjust zoom level (zoom in).
        '''
        # Check if an image is loaded
        if self.image_orig is None or self.bbox[0] is not None:
            return
        
        # change status while zoom function is inefficient, so that user is aware
        self.set_status("loading", "Zooming in...")
        
        # apply zoom
        if (self.zoom * 1.1) < self.zoom_max:
            self.zoom *= 1.1
        

        old_h = self.view_h
        old_w = self.view_w
        self.view_h = int(self.canvas.winfo_height()/self.zoom)
        self.view_w = int(self.canvas.winfo_width()/self.zoom)
        
        if e is not None:
            x = (e.x)/self.canvas.winfo_width()
            y = (e.y)/self.canvas.winfo_height()
        else:
            x = 0.5
            y = 0.5
        dx = round(x * (old_w - self.view_w))
        dy = round(y * (old_h - self.view_h))
        self.view_x += dx
        self.view_y += dy
        
        self.display_blended()
        if self.zoom_event_id is not None:
            self.after_cancel(self.zoom_event_id)
        # if still resizing, schedule a new event
        self.zoom_event_id = self.after(300, self.update_display)
        self.update_preview_frame()
        self.draw_brush_preview(e) # force redraw of brush preview during zoom event
        self.set_status("ready", "Ready")


    def zoom_out(self, e=None):
        '''
        Adjust zoom level (zoom out).
        '''
        # Check if an image is loaded
        if self.image_orig is None or self.bbox[0] is not None:
            return
        
        # change status while zoom function is inefficient, so that user is aware
        self.set_status("loading", "Zooming out...")
        
        # apply zoom
        if (self.zoom * 0.9) > self.zoom_min:
            self.zoom *= 0.9
        
        old_h = self.view_h
        old_w = self.view_w
        self.view_h = int(self.canvas.winfo_height()/self.zoom)
        self.view_w = int(self.canvas.winfo_width()/self.zoom)
        
        if e is not None:
            x = (e.x)/self.canvas.winfo_width()
            y = (e.y)/self.canvas.winfo_height()
        else: # use center of canvas
            x = 0.5
            y = 0.5
        dx = round(x * (old_w - self.view_w))
        dy = round(y * (old_h - self.view_h))
        self.view_x += dx
        self.view_y += dy

        self.display_blended()
        if self.zoom_event_id is not None:
            self.after_cancel(self.zoom_event_id)
        # if still resizing, schedule a new event
        self.zoom_event_id = self.after(300, self.update_display)
        self.update_preview_frame()
        self.draw_brush_preview(e)  # force redraw of brush preview during zoom event
        self.set_status("ready", "Ready")
        
        
    def reset_zoom(self, e=None):
        '''
        Reset zoom (Ctrl-0, Ctrl-Space).
        '''
        if self.image_orig is None:
            return
        self.zoom = 1.0
        
        self.view_x = 0
        self.view_y = 0
        self.view_w = self.canvas.winfo_width()#min(self.canvas.winfo_width(), self.orig_w)
        self.view_h = self.canvas.winfo_height()#min(self.canvas.winfo_height(), self.orig_h)
        
        self.update_display(update_image=True)
        self.update_preview_frame()
