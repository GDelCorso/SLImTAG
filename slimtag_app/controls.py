"""Methods for the controls responsibilities of SegmentationApp."""

import customtkinter as ctk

from slimtag_utils import Tooltip


class ControlsMixin:
    """Composable controls behaviour for ``SegmentationApp``."""

    def show_tool_frame(self, tool):
        frame = self.tool_opt_frame[tool]
        self.current_tool_frame = frame
        frame.tkraise()
        
    def deactivate_tools(self):
        '''
        Keep one tool button active at time.
        '''
        # TODO rework
        for tool in self.tools:
            self.tool_active[tool] = False
            self.tool_btn[tool].configure(border_width=0)
        self.show_tool_frame("empty")

    def set_controls_state(self, enabled: bool):
        '''
        Enable/disable all buttons.
        '''

        state = "normal" if enabled else "disabled"

        def simultaneous_apply():
            for tool, btn in self.tool_btn.items():
                btn.configure(state=state, image=self.tool_icon[tool][state])
    
            if not self.switch_computed_magic_wand:
                for tool in ["wand", "wand_all", "wand_multi", "wand_box"]:
                    self.tool_btn[tool].configure(
                        state="disabled",
                        image=self.tool_icon[tool]["disabled"]
                    )
            # TODO: deactivate the hard-coded always disabled
            always_disabled = [
                "polygon", "bucket",
                "denoise", "interpolate",
                "wand_all", "wand_box",
                "ruler", "area",
                "custom_1", "custom_2", "custom_3", "custom_4"
            ]
    
            for tool in always_disabled:
                self.tool_btn[tool].configure(
                    state="disabled",
                    image=self.tool_icon[tool]["disabled"]
                )
    
        self.left_panel.after_idle(simultaneous_apply)
    
    def set_hide_lock_all_btns(self, enabled: bool, propagate=True):
        '''
        Hard set state for "hide all masks" and "lock all masks" buttons.
        
        Put them in the "non-hidden" and "non-locked" state, and enable/disable
        the buttons depending on state.
        
        If propagate, change also the state of all masks to "non-hidden" and "non-locked".
        '''
        state = "normal" if enabled else "disabled"
        if propagate:
            self.toggle_all_masks_hide(False)
            self.toggle_all_masks_lock(False)
        else:
            self.hide_all_mask_btn.hidden = False
            self.lock_all_mask_btn.locked = False
        self.hide_all_mask_btn.configure(state=state, image=self.icons_dict["EyeOpen"][state])
        self.lock_all_mask_btn.configure(state=state, image=self.icons_dict["LockOpen"][state])

    def set_brush_rotation_slider(self, v):
        self.brush_rot = int(v)
        self.update_brush_rotation_slider()

    def update_brush_rotation_slider(self):
        self.brush_rot_lbl.configure(text=f"{self.brush_rot}°")
        self.brush_rot_slider.set(self.brush_rot)

    #%% TOOL BUTTONS
    def create_tool_button(self, tool, btn_frame, row, col, command=None, last_row=False, help_text=''):
        """
        Aux function to create button object from tool
        """
        assert tool in self.tools
        self.tool_btn[tool] = ctk.CTkButton(self.tools_btn_frame[btn_frame],
                                            width=44, height=44,
                                            text="", image=self.tool_icon[tool]["normal"],
                                            fg_color="transparent",
                                            command=(lambda: self.toggle_tool(tool)) if command is None else command)
        padx = (4, 2) if col == 0 else (2, 4) # col == 1
        pady = (4 if row == 0 else 2, 4 if last_row else 2)
        self.tool_btn[tool].grid(row=row, column=col, sticky="nsew", padx=padx, pady=pady)
        self.tool_btn[tool].help = Tooltip(self.tool_btn[tool], text=help_text)
    
    def toggle_tool(self, tool):
        if not self.image_is_loaded():
            return
        if tool == "undo": # just as a safeguard, "toggle" should not be defined for undo
            return
        
        assert tool in self.tools
        
        if self.tool_btn[tool].cget('state') != "disabled":
            if not self.tool_active[tool]:
                self.deactivate_tools()
                self.tool_active[tool] = True
                self.tool_btn[tool].configure(border_width=2)
                self.show_tool_frame(self.tool_opt_map[tool])
            else:
                self.tool_active[tool] = False
                self.tool_btn[tool].configure(border_width=0)
                self.show_tool_frame("empty")
        
        if tool in ["brush", "eraser"]:
            self._draw_brush_preview(self.mouse['x'], self.mouse['y'])
        
        # bind <Return> and <Escape> for multipoint wand only
        if tool == "wand_multi":
            self.sam_bind_enter = self.bind("<Return>", self.sam_apply)
            self.sam_bind_esc = self.bind("<Escape>", lambda e: self.sam_apply(cancel=True))
        else:
            # remove potential existing points
            self.sam_apply(cancel=True)
            if self.sam_bind_enter is not None:
                self.unbind("<Return>", self.sam_bind_enter)
            if self.sam_bind_esc is not None:
                self.unbind("<Escape>", self.sam_bind_esc)
