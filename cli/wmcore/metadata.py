"""Provenance metadata: embed author/copyright/watermark-ID as EXIF (JPEG) or
PNG text chunks when saving the final watermarked file, and read the ID back.
"""

from PIL import Image, PngImagePlugin

try:
    import piexif
except ImportError:
    piexif = None

# PNG text-chunk key and EXIF tag (Exif IFD "ImageUniqueID", 0xA420 - the tag
# EXIF defines for exactly this: a unique identifier for the image) that carry
# the watermark ID (wmcore.wmid). Like every metadata field, anyone can strip
# them; they are the cheapest layer and the first one platforms remove.
PNG_ID_KEY = "WatermarkID"


def writes_wm_id(output_path):
    """Whether save_with_metadata() can put a watermark ID into a file at this
    path: PNG always; JPEG only with piexif installed; other formats (WebP,
    BMP, TIFF) don't get metadata from this function at all."""
    low = output_path.lower()
    if low.endswith(".png"):
        return True
    if low.endswith((".jpg", ".jpeg")):
        return piexif is not None
    return False


def save_with_metadata(image, output_path, author=None, copyright_text=None, jpeg_quality=95, notify=print,
                       wm_id=None):
    is_jpeg = output_path.lower().endswith((".jpg", ".jpeg"))
    wants_metadata = bool(author or copyright_text or wm_id)

    if is_jpeg:
        rgb = image.convert("RGB")
        exif_bytes = None
        if wants_metadata and piexif is not None:
            exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
            if author:
                exif_dict["0th"][piexif.ImageIFD.Artist] = author.encode("utf-8")
            if copyright_text:
                exif_dict["0th"][piexif.ImageIFD.Copyright] = copyright_text.encode("utf-8")
            if wm_id:
                exif_dict["Exif"][piexif.ExifIFD.ImageUniqueID] = wm_id.encode("ascii")
            exif_bytes = piexif.dump(exif_dict)
        elif wants_metadata and piexif is None:
            notify("Note: install piexif (pip install piexif) to embed EXIF author/copyright/ID in JPEGs.")
        if exif_bytes:
            rgb.save(output_path, quality=jpeg_quality, exif=exif_bytes)
        else:
            rgb.save(output_path, quality=jpeg_quality)
    else:
        pnginfo = None
        if wants_metadata:
            pnginfo = PngImagePlugin.PngInfo()
            if author:
                pnginfo.add_text("Author", author)
            if copyright_text:
                pnginfo.add_text("Copyright", copyright_text)
            if wm_id:
                pnginfo.add_text(PNG_ID_KEY, wm_id)
        if pnginfo:
            image.save(output_path, pnginfo=pnginfo)
        else:
            image.save(output_path)


def read_wm_id_field(path):
    """The raw watermark-ID string stored in `path`'s metadata (PNG text chunk
    or JPEG EXIF ImageUniqueID), or None. Not validated - see wmcore.wmid.parse."""
    try:
        with Image.open(path) as img:
            fmt = img.format
            if fmt == "PNG":
                value = img.info.get(PNG_ID_KEY)
                return value if isinstance(value, str) else None
            if fmt == "JPEG" and piexif is not None:
                exif = img.info.get("exif")
                if not exif:
                    return None
                raw = piexif.load(exif).get("Exif", {}).get(piexif.ExifIFD.ImageUniqueID)
                return raw.decode("ascii", "replace") if raw else None
    except Exception:
        return None
    return None
