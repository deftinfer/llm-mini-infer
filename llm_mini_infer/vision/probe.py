"""Image headers without decoding pixels: format, stored size and orientation.

The size comes from each format's header. The orientation is the one Pillow's `ImageOps.exif_transpose` applies, as
vLLM and Hugging Face's `load_image` decode with Pillow and apply it (`orientation`):
- **Where:** each format's metadata as Pillow's plugin puts it in the image's info: the JPEG APP1 `Exif` segments
  (concatenated) and the last APP1 XMP segment; PNG's `eXIf` chunk and the tEXt, zTXt and iTXt chunks of the keys
  Pillow reads (`exif`, "Raw profile type exif", `XML:com.adobe.xmp`) up to where Pillow stops reading; a WebP's first
  `EXIF` and `XMP ` chunks, in a VP8X file whose flags declare them. GIF and BMP carry none that Pillow reads.
- **What:** IFD0's Orientation tag by Pillow's own EXIF loader (`PIL.Image.Exif`), or, when IFD0 has none, the XMP
  `tiff:Orientation`; a value equal to 2..8 is the orientation, anything else 1, and EXIF that Pillow cannot load
  counts as 1 (`exif_transpose` leaves the image as stored). Orientations 5..8 swap width and height.

Formats: JPEG, PNG, GIF, WebP and BMP (their first frame, as Pillow opens it). Anything else is a `ProbeError` (a
400 for the client). The image pool decodes with Pillow and checks the decoded size against the probe
(`family.decode_rgb`), so a header this module reads differently from Pillow fails the request instead of mis-sizing
it.

**On the event loop** (`api.images`), the probe's work is bounded: a stored size above `images.max_source_pixels` (by
default 178,956,970, Pillow's decompression-bomb bound) is a `ProbeError` before anything decodes the image (a JPEG
declares its size in 4 bytes of its SOF and the decoder allocates what it says: a few hundred bytes can ask for
65,500² pixels, 12.9 GB of RGB; Pillow refuses an image at that size too, but only in the pool); a PNG text chunk
that inflates past `MAX_TEXT_CHUNK`, or text past `MAX_TEXT_MEMORY`, is a `ProbeError`, as Pillow refuses the image
(zlib inflates about 1,000 times); and EXIF whose IFD0 has more than `MAX_IFD0_ENTRIES` entries counts as orientation
1 here, because Pillow's loader costs about a microsecond of Python an entry (the pool's decode still orients such an
image by Pillow, and an orientation of 5..8 then fails its size check: a 400).
"""

from __future__ import annotations

import re
import struct
import warnings
import zlib
from dataclasses import dataclass

_XMP_ORIENTATION = re.compile(rb'tiff:Orientation(="|>)([0-9])')  # Pillow's getexif: on the `xmp` bytes
_XMP_TEXT_ORIENTATION = re.compile(r'tiff:Orientation(="|>)([0-9])')  # and on the PNG `XML:com.adobe.xmp` text
_TRANSPOSED = {o: o for o in range(2, 9)}  # exif_transpose's lookup: a value equal to one (a float or rational too)
_SOF = frozenset({0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF})
# JPEG markers Pillow's JpegImagePlugin knows (0xC0 to 0xFE; any other fails its open), and those it reads no length for
_NO_LENGTH = frozenset({0xC8, *range(0xD0, 0xDA), *range(0xF0, 0xFE)})
# The most pixels an image may store (images.max_source_pixels' default and bound): Pillow's decompression-bomb bound
# (2 x Image.MAX_IMAGE_PIXELS)
MAX_SOURCE_PIXELS = 178_956_970
# The most bytes one PNG zTXt or iTXt chunk may inflate to, and all of an image's text: Pillow's MAX_TEXT_CHUNK
# (ImageFile.SAFEBLOCK, 1 MiB) and MAX_TEXT_MEMORY
MAX_TEXT_CHUNK = 1 << 20
MAX_TEXT_MEMORY = 64 * MAX_TEXT_CHUNK
# The most IFD0 entries the probe has Pillow read (module docstring): cameras write a few dozen; 1,000 take about 1 ms
MAX_IFD0_ENTRIES = 1000
_PNG_TEXT_KEYS = frozenset({b"exif", b"Raw profile type exif", b"XML:com.adobe.xmp"})  # the PNG text getexif reads
_PNG_CHUNK = re.compile(rb"\w{4}")  # a chunk type Pillow reads on (PngImagePlugin's is_cid)


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


# -- the orientation: Pillow's getexif and exif_transpose -------------------------------------------------------------


def orientation(info: dict[str, bytes | str]) -> int:
    """The orientation Pillow 13's `ImageOps.exif_transpose` applies to an image whose info holds these entries, read
    as `Image.getexif` reads them: `exif` (else "Raw profile type exif", in hex) by Pillow's EXIF loader; when IFD0 has
    no Orientation tag, the XMP (`XML:com.adobe.xmp` when set, else `xmp`); then exif_transpose's lookup of 2..8, else
    1. Also 1 when getexif raises (EXIF it cannot load: exif_transpose leaves the image as stored) and when IFD0 has
    more than MAX_IFD0_ENTRIES entries (module docstring). When exif_transpose fails afterwards, rewriting the EXIF
    once it has transposed (rare: crafted values), the pool serves the image as stored; reading that here would cost a
    rewrite of every sub-IFD on the event loop, so an orientation of 5..8 there fails the pool's size check instead."""
    from PIL import Image

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # corrupt entries: the pool's decode warns, as vLLM's does
            exif = info.get("exif")
            if exif is None and "Raw profile type exif" in info:
                exif = bytes.fromhex("".join(info["Raw profile type exif"].split("\n")[3:]))
            e = Image.Exif()
            if exif is not None:
                if _ifd0_entries(exif) > MAX_IFD0_ENTRIES:
                    return 1
                e.load(exif)
            if 0x0112 in e:
                return _TRANSPOSED.get(e.get(0x0112, 1), 1)
            xmp, pattern = info.get("XML:com.adobe.xmp"), _XMP_TEXT_ORIENTATION
            if not xmp and (xmp := info.get("xmp")):
                pattern = _XMP_ORIENTATION
            m = pattern.search(xmp) if xmp else None
            return _TRANSPOSED.get(int(m[2]), 1) if m else 1
    except Exception:
        return 1


def _ifd0_entries(exif: bytes | str) -> int:
    """The IFD0 entries Pillow would read in an EXIF block (after up to two `Exif\\0\\0` prefixes, as it strips them):
    the header's count, or the entries the block holds when it ends before them; 0 when the header gives no IFD0
    (Pillow then reads no entries, or raises)."""
    if not isinstance(exif, bytes):
        return 0
    p = 0
    while p < 12 and exif[p:p + 6] == b"Exif\x00\x00":
        p += 6
    if len(exif) < p + 8 or exif[p:p + 2] not in (b"II", b"MM") or exif[p + 2] == 0x2B:  # BigTIFF: Pillow raises
        return 0
    e = "<" if exif[p] == 0x49 else ">"
    ifd = p + struct.unpack_from(e + "I", exif, p + 4)[0]
    if ifd + 2 > len(exif):
        return 0
    return min(struct.unpack_from(e + "H", exif, ifd)[0], (len(exif) - ifd - 2) // 12)  # Pillow stops at the end


# -- formats ---------------------------------------------------------------------------------------------------------


def _jpeg(b: bytes | memoryview) -> Header:
    """Markers up to the start of scan, walked as Pillow's `JpegImageFile._open` walks them (a byte other than 0xFF
    skipped, 0xFFFF padding, 0xFF00 skipped, a segment's length below 2 read as 2): the first SOF's size, the APP1
    Exif segments (concatenated, as Pillow's `exif`) and the last APP1 XMP segment (its `xmp`)."""
    n = len(b)
    i = 2
    size: tuple[int, int] | None = None
    info: dict[str, bytes | str] = {}
    exif: list[bytes] = []  # joined once: appending to one bytes object copies it each segment
    while i + 2 <= n:
        if b[i] != 0xFF:
            i += 1
            continue
        marker = b[i + 1]
        if marker == 0xFF:
            i += 1
            continue
        if marker == 0x00:
            i += 2
            continue
        if marker < 0xC0:
            raise ProbeError("broken JPEG header: no marker found")
        if marker in _NO_LENGTH:
            i += 2
            continue
        if marker == 0xDA or i + 4 > n:
            break
        seg = max(2, struct.unpack_from(">H", b, i + 2)[0])
        end = min(n, i + 2 + seg)
        if marker in _SOF and size is None:
            if i + 9 > n:
                raise ProbeError("broken JPEG header: truncated SOF")
            h, w = struct.unpack_from(">HH", b, i + 5)
            size = (w, h)
        elif marker == 0xE1:
            s = bytes(b[i + 4:end])
            if s.startswith(b"Exif\x00\x00"):
                exif.append(s[6:] if exif else s)
            elif s.startswith(b"http://ns.adobe.com/xap/1.0/\x00"):
                info["xmp"] = s.split(b"\x00", 1)[1]
        i += 2 + seg
    if size is None:
        raise ProbeError("broken JPEG header: no frame header before the scan")
    _check(*size, "JPEG")
    if exif:
        info["exif"] = b"".join(exif)
    return Header("jpeg", size[0], size[1], orientation(info))


def _png(b: bytes | memoryview) -> Header:
    """IHDR, then every chunk header (the image data never decompressed) for the metadata Pillow puts in the image's
    info: `eXIf` and the text chunks (`_png_text`). Pillow reads the chunks after the image data too when it applies
    the orientation, and stops at a malformed chunk type, or, in an animated PNG, at the next frame's fcTL."""
    n = len(b)
    if n < 33 or bytes(b[12:16]) != b"IHDR":
        raise ProbeError("broken PNG header: no IHDR")
    w, h = struct.unpack_from(">II", b, 16)
    _check(w, h, "PNG")
    info: dict[str, bytes | str] = {}
    text = 0  # the text read, which Pillow bounds by MAX_TEXT_MEMORY
    frames: int | None = None  # acTL's frame count (Pillow's im_n_frames: a second acTL clears it)
    fctl = False  # a fcTL before the image data: it is the first frame, not a default image
    animated: bool | None = None  # decided at the image data, as Pillow decides it at open
    i = 8
    while i + 8 <= n:
        length = struct.unpack_from(">I", b, i)[0]
        kind = bytes(b[i + 4:i + 8])
        if kind == b"IEND" or not _PNG_CHUNK.fullmatch(kind) or (kind == b"fcTL" and animated):
            break
        if kind == b"eXIf":
            info["exif"] = b"Exif\x00\x00" + bytes(b[i + 8:i + 8 + length])
        elif kind in (b"tEXt", b"zTXt", b"iTXt"):
            text += _png_text(info, kind, bytes(b[i + 8:i + 8 + length]))
            if text > MAX_TEXT_MEMORY:
                raise ProbeError(f"PNG text chunks hold more than {MAX_TEXT_MEMORY >> 20} MiB")
        elif animated is None:
            if kind == b"acTL" and length >= 8 and i + 12 <= n:
                count = struct.unpack_from(">I", b, i + 8)[0]
                frames = None if frames is not None else count if 0 < count <= 0x80000000 else None
            elif kind == b"fcTL":
                fctl = True
            elif kind in (b"IDAT", b"fdAT"):  # the image data (Pillow takes a first fdAT as it)
                animated = frames is not None and frames + (not fctl) > 1
        i += 12 + length
    return Header("png", w, h, orientation(info))


def _png_text(info: dict[str, bytes | str], kind: bytes, s: bytes) -> int:
    """Pillow's tEXt, zTXt and iTXt handlers (`PngStream.chunk_*`) for the keys getexif reads: what each puts in
    `info`, and the length of the text it counts against MAX_TEXT_MEMORY. tEXt keeps the key `exif` as bytes; zTXt
    keeps the text of a stream zlib cannot inflate as ""; iTXt keeps nothing then, nor for another compression
    method or a text that is not UTF-8, but records an `XML:com.adobe.xmp` packet as `xmp` before that check."""
    if kind != b"iTXt":
        k, _, v = s.partition(b"\0")
        if k not in _PNG_TEXT_KEYS:
            return 0
        if kind == b"zTXt":
            if v and v[0] != 0:
                raise ProbeError(f"unknown compression method {v[0]} in a PNG zTXt chunk")  # as Pillow refuses it
            try:
                v = _inflate(v[1:])
            except zlib.error:
                v = b""
        text = v.decode("latin-1")
        info[k.decode("latin-1")] = v if kind == b"tEXt" and k == b"exif" else text
        return len(text)
    k, sep, r = s.partition(b"\0")
    if not sep or len(r) < 2 or k not in _PNG_TEXT_KEYS:
        return 0
    compressed, method, r = r[0], r[1], r[2:]
    parts = r.split(b"\0", 2)
    if len(parts) < 3:
        return 0
    lang, translated, v = parts
    if compressed:
        if method != 0:
            return 0
        try:
            v = _inflate(v)
        except zlib.error:
            return 0
    if k == b"XML:com.adobe.xmp":
        info["xmp"] = v
    try:
        lang.decode("utf-8")
        translated.decode("utf-8")
        text = v.decode("utf-8")
    except UnicodeError:
        return 0
    info[k.decode("latin-1")] = text
    return len(text)


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
    """The canvas of a VP8X file, else the VP8 or VP8L frame size, within the RIFF's size. Metadata as libwebp's
    demuxer gives it to Pillow: only in a VP8X file (its first chunk), the first EXIF and XMP chunks its flags declare
    (an empty one counts as none)."""
    n = min(len(b), 8 + struct.unpack_from("<I", b, 4)[0])  # libwebp reads no further than the RIFF's size
    i = 12
    size: tuple[int, int] | None = None
    flags = 0
    info: dict[str, bytes | str] = {}
    seen: set[bytes] = set()
    while i + 8 <= n:
        kind = bytes(b[i:i + 4])
        length = struct.unpack_from("<I", b, i + 4)[0]
        d = i + 8
        if kind == b"VP8X" and d + 10 <= n:
            w = 1 + (b[d + 4] | b[d + 5] << 8 | b[d + 6] << 16)
            h = 1 + (b[d + 7] | b[d + 8] << 8 | b[d + 9] << 16)
            size = (w, h)
            if i == 12:
                flags = b[d]
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
        elif kind in (b"EXIF", b"XMP ") and kind not in seen and flags & (0x08 if kind == b"EXIF" else 0x04):
            seen.add(kind)
            if length:
                info["exif" if kind == b"EXIF" else "xmp"] = bytes(b[d:d + length])
        i = d + length + (length & 1)
    if size is None:
        raise ProbeError("broken WebP header: no VP8, VP8L or VP8X chunk")
    _check(*size, "WebP")
    return Header("webp", size[0], size[1], orientation(info))


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
