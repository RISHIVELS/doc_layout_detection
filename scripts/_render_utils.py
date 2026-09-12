"""
Small shared helper for the two scripts that draw class labels onto document
pages (prepare_dataset.py's label check, mine_failures.py's failure renders).

I pulled this out after fixing the same bug in both places once already: PIL's
bitmap default font is around 10px, which is fine on the raw 1025px page but
becomes an illegible smear the moment the image is resized for a notebook
display or a memo screenshot - which is the only place either render actually
gets looked at. Since both scripts need the identical fix, it belongs in one
place rather than two copies that can drift.
"""

from __future__ import annotations


def load_label_font(size: int):
    """
    Finds a real scalable font to draw labels with, trying the most reliable
    source first.

    A bare `ImageFont.truetype("DejaVuSans-Bold.ttf", size)` only works if
    that filename happens to resolve on whatever font search path the current
    machine has, which is not guaranteed - it works on my Windows dev machine
    but is not something I want to bet the actual Kaggle run on. `matplotlib`
    is already a pinned dependency in requirements.txt and ships its own copy
    of this exact font inside its package data, so that path is guaranteed to
    exist on any machine that can import matplotlib at all. I try that first,
    then a couple of common system locations, and only fall back to PIL's
    tiny bitmap default if every scalable option is somehow missing.
    """
    from PIL import ImageFont

    candidates = []

    try:
        import matplotlib

        candidates.append(f"{matplotlib.get_data_path()}/fonts/ttf/DejaVuSans-Bold.ttf")
    except Exception:
        pass

    candidates += [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # common on Debian/Ubuntu, incl. Kaggle
        "DejaVuSans-Bold.ttf",  # works if the OS's own font search resolves it (e.g. Windows)
    ]

    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue

    return ImageFont.load_default()
