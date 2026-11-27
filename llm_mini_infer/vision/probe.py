"""Image headers without decoding pixels: the format and the stored size.

The probe reads what the frontend needs before a decode may start: the format and the stored size, for the pixel limit,
vLLM's decode budget and the sizing (`vision.family`). The orientation is the decode's: the image pool applies Pillow's
EXIF orientation (`family.decode_rgb`), and the frontend sizes every image as stored, which holds either way round (a
family's sizing is symmetric under a transpose). The probe runs on the frontend's event loop, so it reads a bounded
part of the header:
- **JPEG:** markers up to the first frame header (SOF), at most `MAX_JPEG_MARKERS` before it (a segment, or a
  standalone marker such as RSTn; the SOF itself is not counted) and `MAX_JPEG_FILL` fill bytes (each 0xFF before a
  marker's own, skipped a run at a time at C speed). A JPEG past either is a `ProbeError`: real files hold a few hundred
  markers at most before their frame header (an ICC profile takes up to 255 segments) and no fill to speak of.
- **PNG:** IHDR, the first chunk. **GIF:** the logical screen. **BMP:** the info header.
- **WebP:** its first chunk, which must be VP8X (the canvas), VP8 or VP8L (the frame), as Pillow requires.

Anything else is a `ProbeError` (a 400 for the client). The image pool checks the decoded size against the probe
(`family.decode_rgb`), so a header this module reads differently from Pillow fails the request instead of mis-sizing it.

**The size limit** (`check_source_pixels`, against decompression bombs): a stored size above `images.max_source_pixels`
(by default 178,956,970, Pillow's decompression-bomb bound) is a `ProbeError` before anything decodes the image. A JPEG
declares its size in 4 bytes of its SOF, and the decoder allocates what it says: a few hundred bytes can ask for
65,500² pixels, 12.9 GB of RGB. Pillow refuses an image at that size too, but only once it reaches the pool.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass

_SOF = frozenset({0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF})
_FILL = re.compile(rb"\xff+")
# The most pixels an image may store (images.max_source_pixels' default and bound): Pillow's decompression-bomb bound
# (2 x Image.MAX_IMAGE_PIXELS)
MAX_SOURCE_PIXELS = 178_956_970
# The most markers and fill bytes a JPEG may hold before its frame header (module docstring): together, under a
# millisecond of the walk
MAX_JPEG_MARKERS = 4096
MAX_JPEG_FILL = 1 << 20


class ProbeError(ValueError):
    """The bytes are not an image this server reads (unsupported format, broken header or too many pixels)."""


@dataclass(frozen=True, slots=True)
class Header:
    format: str  # "jpeg", "png", "gif", "webp", "bmp"
    width: int  # as stored
    height: int


def probe(data: bytes | memoryview) -> Header:
    b = bytes(data[:12])
    if b[:3] == b"\xff\xd8\xff":
        return _jpeg(data)
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        return _png(data)
    if b[:6] in (b"GIF87a", b"GIF89a"):
        return _gif(data)
    if b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        return _webp(data)
    if b[:2] == b"BM":
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


def _jpeg(b: bytes | memoryview) -> Header:
    """Markers up to the first SOF, at most `MAX_JPEG_MARKERS` markers and `MAX_JPEG_FILL` fill bytes before it (module
    docstring)."""
    n = len(b)
    i = 2
    markers = fill = 0
    while i + 4 <= n:
        if b[i] != 0xFF:
            raise ProbeError("broken JPEG header: marker expected")
        marker = b[i + 1]
        if marker == 0xFF:  # fill bytes: the run at once, the last 0xFF the marker's own
            end = _FILL.match(b, i + 1, i + 2 + MAX_JPEG_FILL - fill).end()
            fill += end - 1 - i
            i = end - 1
            if fill > MAX_JPEG_FILL:
                raise ProbeError(f"broken JPEG header: more than {MAX_JPEG_FILL >> 20} MiB of fill bytes before the "
                                 "frame header")
        elif marker in _SOF:
            if i + 9 > n:
                raise ProbeError("broken JPEG header: truncated SOF")
            h, w = struct.unpack_from(">HH", b, i + 5)
            _check(w, h, "JPEG")
            return Header("jpeg", w, h)
        elif marker in (0xDA, 0xD9):
            break
        else:
            markers += 1
            if markers > MAX_JPEG_MARKERS:
                raise ProbeError(f"broken JPEG header: more than {MAX_JPEG_MARKERS} markers before the frame header")
            i += 2 if marker == 0x01 or 0xD0 <= marker <= 0xD7 else 2 + struct.unpack_from(">H", b, i + 2)[0]
    raise ProbeError("broken JPEG header: no frame header before the scan")


def _png(b: bytes | memoryview) -> Header:
    if len(b) < 33 or bytes(b[12:16]) != b"IHDR":
        raise ProbeError("broken PNG header: no IHDR")
    w, h = struct.unpack_from(">II", b, 16)
    _check(w, h, "PNG")
    return Header("png", w, h)


def _gif(b: bytes | memoryview) -> Header:
    if len(b) < 10:
        raise ProbeError("broken GIF header")
    w, h = struct.unpack_from("<HH", b, 6)
    _check(w, h, "GIF")
    return Header("gif", w, h)


def _webp(b: bytes | memoryview) -> Header:
    """The first chunk: a VP8X file's canvas, else the VP8 or VP8L frame's size."""
    n = len(b)
    kind = bytes(b[12:16])
    d = 20
    if kind == b"VP8X" and d + 10 <= n:
        w = 1 + (b[d + 4] | b[d + 5] << 8 | b[d + 6] << 16)
        h = 1 + (b[d + 7] | b[d + 8] << 8 | b[d + 9] << 16)
    elif kind == b"VP8 " and d + 10 <= n:
        if bytes(b[d + 3:d + 6]) != b"\x9d\x01\x2a":
            raise ProbeError("broken WebP header: no VP8 start code")
        w, h = struct.unpack_from("<HH", b, d + 6)
        w, h = w & 0x3FFF, h & 0x3FFF
    elif kind == b"VP8L" and d + 5 <= n:
        if b[d] != 0x2F:
            raise ProbeError("broken WebP header: no VP8L signature")
        bits = struct.unpack_from("<I", b, d + 1)[0]
        w, h = 1 + (bits & 0x3FFF), 1 + ((bits >> 14) & 0x3FFF)
    else:
        raise ProbeError("broken WebP header: no VP8, VP8L or VP8X chunk first")
    _check(w, h, "WebP")
    return Header("webp", w, h)


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
