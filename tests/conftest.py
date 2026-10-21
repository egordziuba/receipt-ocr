"""Make the synthetic-text tests portable: map Windows font paths to an available font on other systems."""
import os
from pathlib import Path

from PIL import ImageFont

_FALLBACKS = {
    "arial.ttf": ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/Library/Fonts/Arial.ttf"],
    "arialbd.ttf": ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/Library/Fonts/Arial Bold.ttf"],
    "cour.ttf": ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "/Library/Fonts/Courier New.ttf"],
}

if os.name != "nt":
    _original = ImageFont.truetype

    def _truetype(font=None, *args, **kwargs):
        if isinstance(font, (str, Path)) and str(font).replace("\\", "/").lower().startswith("c:/windows/fonts/"):
            name = Path(str(font).replace("\\", "/")).name.lower()
            for candidate in _FALLBACKS.get(name, _FALLBACKS["arial.ttf"]):
                if Path(candidate).exists():
                    return _original(candidate, *args, **kwargs)
        return _original(font, *args, **kwargs)

    ImageFont.truetype = _truetype
