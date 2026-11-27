"""Image headers without decoding pixels: format, stored size and orientation.

The size comes from each format's header; the orientation is read the way Pillow reads it, because Hugging Face's
`load_image` decodes with Pillow and applies `ImageOps.exif_transpose`:
- **EXIF:** JPEG APP1 `Exif` segments (concatenated, as Pillow does), the PNG `eXIf` chunk or a "Raw profile type exif"
  text chunk, and the WebP `EXIF` chunk. The Orientation tag (0x0112) of IFD0 as Pillow loads it (`exif_orientation`):
  its first value, of any numeric type, is the orientation when it equals 2..8; anything else counts as 1.
- **XMP**, when there is no EXIF, it is empty, or its IFD0 has no Orientation tag: `tiff:Orientation="N"` or
  `<tiff:Orientation>N</tiff:Orientation>` in the JPEG APP1 XMP segment (the last one), the PNG iTXt
  `XML:com.adobe.xmp` chunk, or the WebP `XMP ` chunk. EXIF that Pillow cannot parse counts as 1, without the XMP.
- Orientations 5..8 swap width and height.

Formats: JPEG, PNG, GIF, WebP and BMP (their first frame, as Pillow opens it). Anything else is a `ProbeError` (a
400 for the client). The image pool checks the decoded size against the probe (`family.decode_rgb`), so a header this
module reads differently from Pillow fails the request instead of mis-sizing it.

**The size limit** (`check_source_pixels`, against decompression bombs): a stored size above `images.max_source_pixels`
(by default 178,956,970, Pillow's decompression-bomb bound) is a `ProbeError` before anything decodes the image. A JPEG
declares its size in 4 bytes of its SOF, and the decoder allocates what it says: a few hundred bytes can ask for
65,500² pixels, 12.9 GB of RGB. Pillow refuses an image at that size too, but only once it reaches the pool. A
compressed PNG text chunk that inflates past `MAX_TEXT_CHUNK` is a `ProbeError` too, as Pillow refuses it: the probe
runs on the frontend's event loop, and zlib inflates about 1,000 times.
"""

from __future__ import annotations

import contextlib
import math
import re
import struct
import zlib
from dataclasses import dataclass
from fractions import Fraction

_XMP_ORIENTATION = re.compile(rb'tiff:Orientation(="|>)([0-9])')
# The TIFF value types Pillow reads (`TiffImagePlugin`'s loaders): type -> (bytes per value, struct code of one value;
# None: BYTE, ASCII and UNDEFINED, which load as bytes or text; "R" and "r": RATIONAL and SRATIONAL, two LONGs).
# 13 is IFD, read as LONG
_TIFF_TYPES: dict[int, tuple[int, str | None]] = {
    1: (1, None), 2: (1, None), 3: (2, "H"), 4: (4, "I"), 5: (8, "R"), 6: (1, "b"), 7: (1, None), 8: (2, "h"),
    9: (4, "i"), 10: (8, "r"), 11: (4, "f"), 12: (8, "d"), 13: (4, "I"), 16: (8, "Q")}
# The TIFF headers Pillow's EXIF loader reads (`TiffImagePlugin.PREFIXES`): 42 in either byte order, and a big-endian
# 43, which it reads as classic TIFF; its other one, a little-endian BigTIFF, raises there (`Exif.load` reads 8 bytes)
_TIFF_HEADS = (b"MM\x00\x2a", b"II\x2a\x00", b"MM\x2a\x00", b"II\x00\x2a", b"MM\x00\x2b")
_SOF = frozenset({0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF})
# The most pixels an image may store (images.max_source_pixels' default and bound): Pillow's decompression-bomb bound
# (2 x Image.MAX_IMAGE_PIXELS)
MAX_SOURCE_PIXELS = 178_956_970
# The most bytes one PNG zTXt or iTXt chunk may inflate to: Pillow's MAX_TEXT_CHUNK (ImageFile.SAFEBLOCK, 1 MiB)
MAX_TEXT_CHUNK = 1 << 20


class ProbeError(ValueError):
    """The bytes are not an image this server reads (unsupported format, broken header or too many pixels)."""


@dataclass(frozen=True, slots=True)
class Header:
    format: str  # "jpeg", "png", "gif", "webp", "bmp"
    width: int  # as stored
    height: int
    orientation: int = 1  # Pillow's exif_transpose rule, 1..8

    @property
    def oriented(self) -> tuple[int, int]:
        """(width, height) as the model sees the image: after the orientation."""
        return (self.height, self.width) if self.orientation >= 5 else (self.width, self.height)


def probe(data: bytes | memoryview) -> Header:
    b = bytes(data[:4]) if len(data) >= 4 else bytes(data)
    if b[:3] == b"\xff\xd8\xff":
        return _jpeg(data)
    if bytes(data[:8]) == b"\x89PNG\r\n\x1a\n":
        return _png(data)
    if bytes(data[:6]) in (b"GIF87a", b"GIF89a"):
        return _gif(data)
    if bytes(data[:4]) == b"RIFF" and bytes(data[8:12]) == b"WEBP":
        return _webp(data)
    if bytes(data[:2]) == b"BM":
        return _bmp(data)
    raise ProbeError("unsupported image format (JPEG, PNG, WebP, GIF and BMP are)")


def _check(w: int, h: int, fmt: str) -> None:
    if w <= 0 or h <= 0:
        raise ProbeError(f"broken {fmt} header: size {w}x{h}")


def check_source_pixels(header: Header, limit: int = MAX_SOURCE_PIXELS) -> None:
    """A `ProbeError` when the image stores more than `limit` pixels (images.max_source_pixels; module docstring).
    The check and its words are vLLM's: the stored size, which an orientation does not change, strictly above the
    limit. vLLM answers such an image with a 500 in practice: Pillow's own DecompressionBombError, not a
    ValueError, fires first at its default and reaches the catch-all handler. Here it is a 400, as every other image
    failure is."""
    n = header.width * header.height
    if n > limit:
        raise ProbeError(f"Image dimensions {header.width}x{header.height} ({n} pixels) exceed the maximum of {limit} "
                         "pixels (images.max_source_pixels)")


# -- EXIF and XMP ----------------------------------------------------------------------------------------------------


def exif_orientation(exif: bytes) -> int | None:
    """The Orientation tag of IFD0 in a TIFF/EXIF block (behind up to two `Exif\\0\\0` prefixes, which Pillow strips),
    as Pillow 13 loads it (`Image.Exif.load`, `TiffImagePlugin.ImageFileDirectory_v2.load`) and `exif_transpose`
    applies it: the first entry of the tag with data of a type Pillow reads (inline up to 4 bytes, else at its offset,
    within Pillow's running bound of the block's length), and that data's first value, whatever the count. 2..8 when
    the value equals one (a SHORT or LONG, a signed or IFD type, a float or a rational: Pillow's lookup compares
    numbers), else 1 (bytes and text never match). 1 too when Pillow cannot parse the block (shorter than a TIFF header,
    or a header not in `_TIFF_HEADS`): `getexif` raises before it reads any XMP, and the image is served as stored.
    None when Pillow finds no Orientation: the block empty once stripped, no such tag, IFD0 or the entry's data past the
    block's end (Pillow warns and stops reading IFD0 there). Pillow stops there for an earlier tag's data too; that is
    not followed (rare)."""
    t = exif
    for _ in range(2):
        if t[:6] == b"Exif\x00\x00":
            t = t[6:]
    if not t:
        return None
    if len(t) < 8 or t[:4] not in _TIFF_HEADS:
        return 1
    e = "<" if t[:2] == b"II" else ">"
    ifd = struct.unpack_from(e + "I", t, 4)[0]
    if ifd + 2 > len(t):
        return None
    n = struct.unpack_from(e + "H", t, ifd)[0]
    stored: set[int] = set()  # the tags Pillow keeps: the first entry of each with data
    total = 0  # the bytes of the entries' data at an offset so far, which Pillow bounds by the block's length
    for k in range(n):
        p = ifd + 2 + 12 * k
        if p + 12 > len(t):
            return None
        tag, typ, count = struct.unpack_from(e + "HHI", t, p)
        kind = _TIFF_TYPES.get(typ)
        if tag in stored or kind is None:
            continue
        size = count * kind[0]
        if size > 4:
            total += size
            off = struct.unpack_from(e + "I", t, p + 8)[0]
            if total > len(t):
                continue
            if off + size > len(t):
                if tag == 0x0112:
                    return None
                continue
            data = t[off:off + size]
        else:
            data = t[p + 8:p + 8 + size]
        if not data:
            continue
        if tag == 0x0112:
            return _orientation_value(data, kind[1], e)
        stored.add(tag)
    return None


def _orientation_value(data: bytes, code: str | None, e: str) -> int:
    """The orientation `exif_transpose` applies for an Orientation entry's data: its first value when that equals 2..8
    (`code`: `_TIFF_TYPES`'), else 1."""
    if code is None:
        return 1
    if code in "Rr":
        num, den = struct.unpack_from(e + ("II" if code == "R" else "ii"), data)
        v: float | Fraction = Fraction(num, den) if den else math.nan
    else:
        v = struct.unpack_from(e + code, data)[0]
    return next((o for o in range(2, 9) if v == o), 1)


def _xmp_orientation(xmp: bytes | None) -> int | None:
    if not xmp:
        return None
    m = _XMP_ORIENTATION.search(xmp)
    if m is None:
        return None
    v = int(m.group(2))
    return v if 1 <= v <= 8 else 1


def _orientation(exif: bytes | None, xmp: bytes | None) -> int:
    o = exif_orientation(exif) if exif else None
    if o is None:
        o = _xmp_orientation(xmp)
    return o or 1


# -- formats ---------------------------------------------------------------------------------------------------------


def _jpeg(b: bytes | memoryview) -> Header:
    """Markers up to the start of scan: the first SOF's size, the APP1 Exif segments (concatenated) and the last
    APP1 XMP segment."""
    n = len(b)
    i = 2
    size: tuple[int, int] | None = None
    exif: bytes | None = None
    xmp: bytes | None = None
    while i + 4 <= n:
        if b[i] != 0xFF:
            raise ProbeError("broken JPEG header: marker expected")
        marker = b[i + 1]
        if marker == 0xFF:
            i += 1
            continue
        if marker == 0x01 or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if marker in (0xDA, 0xD9):
            break
        seg = struct.unpack_from(">H", b, i + 2)[0]
        end = min(n, i + 2 + seg)
        if marker in _SOF and size is None:
            if i + 9 > n:
                raise ProbeError("broken JPEG header: truncated SOF")
            h, w = struct.unpack_from(">HH", b, i + 5)
            size = (w, h)
        elif marker == 0xE1:
            s = bytes(b[i + 4:end])
            if s.startswith(b"Exif\x00\x00"):
                exif = s if exif is None else exif + s[6:]
            elif s.startswith(b"http://ns.adobe.com/xap/1.0/\x00"):
                xmp = s.split(b"\x00", 1)[1]
        i += 2 + seg
    if size is None:
        raise ProbeError("broken JPEG header: no frame header before the scan")
    _check(*size, "JPEG")
    return Header("jpeg", size[0], size[1], _orientation(exif, xmp))


def _png(b: bytes | memoryview) -> Header:
    """IHDR, then every chunk header (the image data never decompressed) for eXIf, a "Raw profile type exif" text chunk
    and the iTXt XMP packet: Pillow reads chunks after IDAT too when it applies the orientation."""
    n = len(b)
    if n < 33 or bytes(b[12:16]) != b"IHDR":
        raise ProbeError("broken PNG header: no IHDR")
    w, h = struct.unpack_from(">II", b, 16)
    _check(w, h, "PNG")
    exif: bytes | None = None
    xmp: bytes | None = None
    i = 8
    while i + 8 <= n:
        length = struct.unpack_from(">I", b, i)[0]
        kind = bytes(b[i + 4:i + 8])
        data = b[i + 8:i + 8 + length]
        if kind == b"eXIf":
            exif = b"Exif\x00\x00" + bytes(data)
        elif kind in (b"tEXt", b"zTXt", b"iTXt"):
            key, _, rest = bytes(data).partition(b"\x00")
            if key == b"Raw profile type exif" and exif is None:
                text = _png_text(kind, rest)
                if text is not None:
                    with contextlib.suppress(ValueError):
                        exif = bytes.fromhex("".join(text.decode("latin-1").split("\n")[3:]))
            elif key == b"XML:com.adobe.xmp" and kind == b"iTXt":
                xmp = _png_text(kind, rest)
        elif kind == b"IEND":
            break
        i += 12 + length
    return Header("png", w, h, _orientation(exif, xmp))


def _png_text(kind: bytes, rest: bytes) -> bytes | None:
    """The text of a tEXt/zTXt/iTXt chunk after its keyword."""
    try:
        if kind == b"tEXt":
            return rest
        if kind == b"zTXt":
            return _inflate(rest[1:])
        compressed = rest[0]  # iTXt: flag, method, language\0, translated keyword\0, text
        text = rest[2:].split(b"\x00", 2)
        if len(text) < 3:
            return None
        return _inflate(text[2]) if compressed else text[2]
    except (zlib.error, IndexError):
        return None


def _inflate(data: bytes) -> bytes:
    """Pillow's `_safe_zlib_decompress`: at most MAX_TEXT_CHUNK bytes, and a `ProbeError` past them."""
    d = zlib.decompressobj()
    out = d.decompress(data, MAX_TEXT_CHUNK)
    if d.unconsumed_tail:
        raise ProbeError(f"a PNG text chunk inflates past {MAX_TEXT_CHUNK >> 20} MiB")
    return out


def _gif(b: bytes | memoryview) -> Header:
    if len(b) < 10:
        raise ProbeError("broken GIF header")
    w, h = struct.unpack_from("<HH", b, 6)
    _check(w, h, "GIF")
    return Header("gif", w, h)


def _webp(b: bytes | memoryview) -> Header:
    """The canvas of a VP8X file (with its EXIF and XMP chunks), else the VP8 or VP8L frame size."""
    n = len(b)
    i = 12
    size: tuple[int, int] | None = None
    exif: bytes | None = None
    xmp: bytes | None = None
    while i + 8 <= n:
        kind = bytes(b[i:i + 4])
        length = struct.unpack_from("<I", b, i + 4)[0]
        d = i + 8
        if kind == b"VP8X" and d + 10 <= n:
            w = 1 + (b[d + 4] | b[d + 5] << 8 | b[d + 6] << 16)
            h = 1 + (b[d + 7] | b[d + 8] << 8 | b[d + 9] << 16)
            size = (w, h)
        elif kind == b"VP8 " and size is None and d + 10 <= n:
            if bytes(b[d + 3:d + 6]) != b"\x9d\x01\x2a":
                raise ProbeError("broken WebP header: no VP8 start code")
            w, h = struct.unpack_from("<HH", b, d + 6)
            size = (w & 0x3FFF, h & 0x3FFF)
        elif kind == b"VP8L" and size is None and d + 5 <= n:
            if b[d] != 0x2F:
                raise ProbeError("broken WebP header: no VP8L signature")
            bits = struct.unpack_from("<I", b, d + 1)[0]
            size = (1 + (bits & 0x3FFF), 1 + ((bits >> 14) & 0x3FFF))
        elif kind == b"EXIF":
            exif = bytes(b[d:d + length])
        elif kind == b"XMP ":
            xmp = bytes(b[d:d + length])
        i = d + length + (length & 1)
    if size is None:
        raise ProbeError("broken WebP header: no VP8, VP8L or VP8X chunk")
    _check(*size, "WebP")
    return Header("webp", size[0], size[1], _orientation(exif, xmp))


def _bmp(b: bytes | memoryview) -> Header:
    if len(b) < 26:
        raise ProbeError("broken BMP header")
    hdr = struct.unpack_from("<I", b, 14)[0]
    if hdr == 12:  # OS/2 BITMAPCOREHEADER
        w, h = struct.unpack_from("<HH", b, 18)
    else:
        w, h = struct.unpack_from("<ii", b, 18)
    h = abs(h)  # negative: top-down rows
    _check(w, h, "BMP")
    return Header("bmp", w, h)
