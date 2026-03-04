# utility functions for color management

import re

def contrasting_color(color, light_color='#ffffff', dark_color='#000000'):
    '''
    Return a color depending on another color.
    
    In particular, return lightColor if the relative luminance Y of
    the input color is low (hardcoded check: Y <= 0.179), and darkColor
    otherwise.

    Parameters
    ----------
    color : string of the form '#RRGGBB'
        Input color in HTML format. Here RR, GG, BB are hex digits.
    lightColor : string of the form '#RRGGBB', optional
        Output color for dark input colors. The default is '#ffffff'.
    darkColor : string of the form '#RRGGBB', optional
        Output color for light input colors. The default is '#000000'.

    Returns
    -------
    A string of the form '#RRGGBB', representing the output color.

    '''
    assert re.match(r"^#[0-9A-Fa-f]{6}$", color) is not None, "Input color not in format '#RRGGBB'"
    assert re.match(r"^#[0-9A-Fa-f]{6}$", light_color) is not None, "Light output color not in format '#RRGGBB'"
    assert re.match(r"^#[0-9A-Fa-f]{6}$", dark_color) is not None, "Dark output color not in format '#RRGGBB'"
    # assume input color is in sRGB space
    # each channel [0,255] is mapped to [0,1] and linearized
    cols = hex_to_rgb(color)
    R = _linearize_rgb_channel(cols[0] / 255)
    G = _linearize_rgb_channel(cols[1] / 255)
    B = _linearize_rgb_channel(cols[2] / 255)
    # compute relative luminance (Y) depending on (linear) RGB values
    Y = (0.2126 * R) + (0.7152 * G) + (0.0722 * B)
    return dark_color if Y > 0.179 else light_color

def _linearize_rgb_channel(ch):
    '''
    Given a value in [0, 1] of a channel in the (gamma-compressed) sRGB
    color space, return the (linearized) intensity of that channel
    (gamma decompression)
    
    Source: IEC 61966-2-1:1999/AMD1:2003
    '''
    if ch <= 0.04045:
        return ch / 12.92
    return pow((ch + 0.055) / 1.055, 2.4)

def hex_to_rgb(hstring):
    '''
    Given a string of the form '#RRGGBB', return the corresponding triplet
    (R, G, B) with integers in [0,255]
    '''
    assert re.match(r"^#[0-9A-Fa-f]{6}$", hstring) is not None, "Input string not in format '#RRGGBB'"
    return tuple(int(hstring[i:i+2], 16) for i in (1, 3, 5))

def rgb_to_hex(rgb):
    '''
    Given either a triplet (R, G, B) with integers in [0-255], or a string
    "(R,G,B)" with the same values, returns a string of the form '#RRGGBB'
    '''
    if isinstance(rgb, str):
        try:
            R, G, B = (int(ch) for ch in re.search(r"^\(( *[0-9]+ *),( *[0-9]+ *),( *[0-9]+ *)\)$", rgb).groups())
        except TypeError:
            raise ValueError("Input string cannot be interpreted as RGB color")
    else: # rgb is a 3-tuple
        R, G, B = tuple(int(ch) for ch in rgb)
    # clip values
    R = min(max(R, 0), 255)
    G = min(max(G, 0), 255)
    B = min(max(B, 0), 255)
    return f"#{R:02x}{G:02x}{B:02x}"
