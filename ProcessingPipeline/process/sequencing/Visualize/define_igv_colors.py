from PIL import ImageColor
import numpy as np

base_to_hsl = {
    'A': 'hsl(120, 100%, {0:d}%)'.format,
    'C': 'hsl(240, 100%, {0:d}%)'.format,
    'G': 'hsl(30, 75%, {0:d}%)'.format,
    'T': 'hsl(0, 100%, {0:d}%)'.format,
}

def _get_base_rgb(base):
    if base == 'N' or base == '-' or base == '.':
        rgb = (0, 0, 0)
    else:
        l = 45
        outline_hsl = base_to_hsl[base](l)
        rgb = ImageColor.getrgb(outline_hsl)
    return rgb

rgbs = {base: _get_base_rgb(base) for base in 'TCAGN-'}
normalized_rgbs = {b: np.array(rgb) / 255. for b, rgb in rgbs.items()}
