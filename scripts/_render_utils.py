# Shared by prepare_dataset.py's label check and mine_failures.py's
# failure renders. PIL's bitmap default font (~10px) turns into an
# illegible smear once resized for a notebook or memo screenshot - which
# is the only place either render gets looked at.
from __future__ import annotations


def load_label_font(size: int):
    """Finds a real scalable font. matplotlib ships its own DejaVu copy
    and is already a pinned dependency, so that's guaranteed to exist -
    a bare "DejaVuSans-Bold.ttf" only works if the OS's font search
    happens to resolve it, which isn't something to bet a Kaggle run on."""
    from PIL import ImageFont

    candidates = []

    try:
        import matplotlib

        candidates.append(f"{matplotlib.get_data_path()}/fonts/ttf/DejaVuSans-Bold.ttf")
    except Exception:
        pass

    candidates += [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Debian/Ubuntu, Kaggle
        "DejaVuSans-Bold.ttf",  # works if OS font search resolves it (e.g. Windows)
    ]

    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue

    return ImageFont.load_default()
