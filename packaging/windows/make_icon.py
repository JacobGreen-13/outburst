"""Write the Windows icon used by the frozen app, the installer, and the shortcuts.

Run this after Outburst and PySide6 are installed. The icon is the PNG shipped with the window.
Windows shortcuts read the classic 32-bit bitmap images in an .ico file. A PNG stored inside
the .ico is skipped for those shortcuts, so each size is written as a device-independent bitmap.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

ICON_SIZES = (16, 24, 32, 48, 64, 128, 256)


def dib_32(width: int, height: int, rgba: bytes) -> bytes:
    """Pack one square image as a 32-bit bitmap for an .ico file.

    Inputs: width and height in pixels; rgba, top-down bytes, four per pixel, in R, G, B, A order.
    Output: a BITMAPINFOHEADER, bottom-up BGRA pixels, and a 1-bit AND mask.
    Uses: write_ico.
    Limitations: the buffer length must be width times height times 4. A fully transparent pixel is marked in the mask. Partial transparency stays in the alpha byte.
    """

    if width <= 0 or height <= 0 or len(rgba) != width * height * 4:
        raise ValueError("RGBA buffer does not match the icon size")
    xor_size = width * height * 4
    mask_stride = ((width + 31) // 32) * 4
    header = struct.pack(
        "<IiiHHIIiiII",
        40,
        width,
        height * 2,
        1,
        32,
        0,
        xor_size,
        0,
        0,
        0,
        0,
    )
    xor = bytearray(xor_size)
    mask = bytearray(mask_stride * height)
    for y in range(height):
        source_y = height - 1 - y
        for x in range(width):
            index = (source_y * width + x) * 4
            red, green, blue, alpha = rgba[index : index + 4]
            pixel = (y * width + x) * 4
            xor[pixel : pixel + 4] = bytes((blue, green, red, alpha))
            if alpha == 0:
                mask[y * mask_stride + (x // 8)] |= 1 << (7 - (x % 8))
    return header + xor + mask


def write_ico(destination: Path, images: list[tuple[int, bytes]]) -> None:
    """Write an .ico whose images are 32-bit bitmaps.

    Inputs: destination, the .ico path; images, pairs of pixel size and dib_32 bytes.
    Output: no return value. The file is created or replaced.
    Uses: the PyInstaller executable, the Inno Setup wizard, and the desktop shortcut.
    Limitations: each image is a square bitmap. The directory size byte is 0 when the image is 256 pixels, which is the ICO rule for that size.
    """

    count = len(images)
    header = struct.pack("<HHH", 0, 1, count)
    entries = bytearray()
    payload = bytearray()
    offset = 6 + (16 * count)
    for size, image in images:
        dimension = 0 if size >= 256 else size
        entries += struct.pack("<BBBBHHII", dimension, dimension, 0, 0, 1, 32, len(image), offset)
        payload += image
        offset += len(image)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(header + entries + payload)


def rgba_from_image(image: object) -> bytes:
    """Read a square QImage as tightly packed RGBA bytes.

    Inputs: image, a QImage already converted to RGBA8888.
    Output: top-down R, G, B, A bytes, one pixel after another.
    Uses: main, so a padded Qt row is not stored in the bitmap.
    Limitations: the image must already be Format_RGBA8888. The width and height are read from the image.
    """

    width = int(image.width())
    height = int(image.height())
    stride = int(image.bytesPerLine())
    raw = bytes(image.constBits())
    if stride == width * 4:
        return raw[: width * height * 4]
    rows = bytearray()
    for y in range(height):
        start = y * stride
        rows += raw[start : start + (width * 4)]
    return bytes(rows)


def main() -> None:
    """Draw the Outburst mark at several sizes and save an .ico.

    Inputs: the destination path in argv, or build/windows/outburst.ico when argv is empty.
    Output: no return value. Raises SystemExit when the path is missing or a size cannot be encoded.
    Uses: packaging/windows/build.ps1 before PyInstaller runs.
    Limitations: a QApplication is created so the PNG can be scaled. The process exits after the file is written.
    """

    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QApplication

    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("build/windows/outburst.ico")
    app = QApplication.instance() or QApplication([])
    del app
    from outburst.gui.styles import build_app_icon

    icon = build_app_icon()
    images: list[tuple[int, bytes]] = []
    for size in ICON_SIZES:
        image = icon.pixmap(size, size).toImage().convertToFormat(QImage.Format.Format_RGBA8888)
        if image.isNull() or image.width() != size or image.height() != size:
            raise SystemExit(f"Could not encode the {size}px icon")
        images.append((size, dib_32(size, size, rgba_from_image(image))))
    write_ico(destination, images)


if __name__ == "__main__":
    main()
