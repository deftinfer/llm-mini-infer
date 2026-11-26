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
that inflates past `MAX_TEXT_CHUNK` is a `ProbeError`, as Pillow refuses the image. Bounds of its own, where Pillow
reads on, for crafted files (no encoder comes near them):
- a JPEG header of more than `MAX_JPEG_MARKERS` markers before its scan (segments, stray-byte and fill runs, markers
  without a length; a fill run counts one a KiB) is a `ProbeError`; a run is skipped at C speed;
- of PNG text, the probe reads only the values `getexif` uses, each from the last chunk Pillow keeps, and refuses
  past `MAX_TEXT_WORK`: bytes inflated, kept or not, text decoded or parsed, compressed bytes fed to zlib (which
  spends up to about 60 ns on a byte of crafted input), and the EXIF and XMP it then reads;
- EXIF whose IFD0 has more than `MAX_IFD0_ENTRIES` entries, and a "Raw profile type exif" longer than
  `MAX_TEXT_CHUNK`, count as orientation 1 (Pillow's loader costs about a microsecond of Python an entry); XMP is
  searched in its first `MAX_TEXT_CHUNK` bytes, up to `_XMP_DECOYS` `tiff:Orientation` that do not match; of the
  Orientation tag, one value is unpacked, whatever its count; meanwhile Pillow's TIFF loader logs nothing (one
  record an entry at debug level) and reads each tag's data at once (`_read_at_once`). The pool's decode still
  orients by Pillow, and an orientation the probe read otherwise then fails its size check if it swaps the sides (a
  400).
"""

from __future__ import annotations

import logging
import re
import struct
import warnings
import zlib
from collections import deque
from dataclasses import dataclass

# Pillow's getexif finds `re.search(r'tiff:Orientation(="|>)([0-9])', xmp)`: the probe finds the tag with memchr and
# reads what follows it (`_xmp_orientation`), on bytes (on text: the same match)
_XMP_TAG = b"tiff:Orientation"
_XMP_DECOYS = 64
_TRANSPOSED = {o: o for o in range(2, 9)}  # exif_transpose's lookup: a value equal to one (a float or rational too)
_SOF = frozenset({0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF})
# JPEG markers Pillow's JpegImagePlugin knows (0xC0 to 0xFE; any other fails its open), and those it reads no length for
_NO_LENGTH = frozenset({0xC8, *range(0xD0, 0xDA), *range(0xF0, 0xFE)})
# Pillow's JPEG walk by the byte after an 0xFF: 0 a segment the probe skips by its length, 1 a byte skipped with the
# 0xFF (0x00) or a marker without a length, 2 fill (0xFF), 3 a frame header, 4 APP1, 5 the start of scan, 6 no marker
_JPEG_STEP = bytes(2 if m == 0xFF else 1 if m == 0 or m in _NO_LENGTH else 6 if m < 0xC0 else 5 if m == 0xDA else
                   3 if m in _SOF else 4 if m == 0xE1 else 0 for m in range(256))
_FILL = re.compile(rb"\xff+")  # a run of fill: one match, charged a marker a KiB (`_jpeg`)
_U16 = struct.Struct(">H").unpack_from
_U32 = struct.Struct(">I").unpack_from
_U32LE = struct.Struct("<I").unpack_from
# The most markers the probe walks before a JPEG's scan (module docstring): a file has tens, hundreds with an ICC
# profile or XMP split over segments; each costs about 0.2 µs of Python
MAX_JPEG_MARKERS = 2048
# The most pixels an image may store (images.max_source_pixels' default and bound): Pillow's decompression-bomb bound
# (2 x Image.MAX_IMAGE_PIXELS)
MAX_SOURCE_PIXELS = 178_956_970
# The most bytes one PNG zTXt or iTXt chunk may inflate to: Pillow's MAX_TEXT_CHUNK (ImageFile.SAFEBLOCK, 1 MiB)
MAX_TEXT_CHUNK = 1 << 20
# The PNG text work the probe does at most (module docstring), about a nanosecond a unit: a byte inflated, kept or
# not (a stream zlib refuses counts the step it failed in); two a byte of a Raw profile's hex parsed; _DECODE_COST a
# byte decoded beyond ASCII and a 32nd a byte checked for ASCII; _FED_COST a compressed byte fed to zlib (in pieces of
# _TEXT_FEED: crafted deflate blocks cost zlib up to about 70 ns a byte); _TEXT_CHUNK_COST a chunk read; and, for what
# the probe then does with a PNG's metadata, _ENTRY_COST an IFD0 entry loaded (and a sixteenth of the block), one and
# a half a byte of XMP searched. Text of up to half Pillow's 1 MiB, or a compressed EXIF of tens of KiB, fits
MAX_TEXT_WORK = 5 * MAX_TEXT_CHUNK // 2
_FED_COST = 72
_DECODE_COST = 3
_TEXT_CHUNK_COST = 1 << 12
_ENTRY_COST = 1 << 12  # an IFD0 entry Pillow's loader reads, from PNG text or eXIf (up to about 4 µs)
_INFLATE_STEP = 1 << 14
_TEXT_FEED = 1 << 12
# The most IFD0 entries the probe has Pillow read (module docstring): cameras write a few dozen
MAX_IFD0_ENTRIES = 256
# The chunks the probe reads: 0 IEND, 1 eXIf, 2 text, 3 acTL, 4 fcTL, 5 the image data
_PNG_KIND = {b"IEND": 0, b"eXIf": 1, b"tEXt": 2, b"zTXt": 2, b"iTXt": 2, b"acTL": 3, b"fcTL": 4, b"IDAT": 5,
             b"fdAT": 5}
# The text keys getexif reads, by a chunk's first byte: the key with its NUL and the value it sets (0 `exif`, 1 "Raw
# profile type exif", 2 `XML:com.adobe.xmp`, 3 `xmp`, which an iTXt XMP packet sets too); and each key alone, which
# tEXt and zTXt read as the key with an empty value
_PNG_TEXT = {0x65: (b"exif\0", 0), 0x52: (b"Raw profile type exif\0", 1), 0x58: (b"XML:com.adobe.xmp\0", 2),
             0x78: (b"xmp\0", 3)}
_PNG_TEXT_KEYS = tuple(k for k, _ in _PNG_TEXT.values())
_PNG_BARE_KEYS = {len(k) - 1: k[:-1] for k in _PNG_TEXT_KEYS}


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


def orientation(info) -> int:
    """The orientation Pillow 13's `ImageOps.exif_transpose` applies to an image whose info holds these entries, read
    as `Image.getexif` reads them: `exif` (else "Raw profile type exif", in hex) by Pillow's EXIF loader; when IFD0 has
    no Orientation tag, the XMP (`XML:com.adobe.xmp` when set, else `xmp`); then exif_transpose's lookup of 2..8, else
    1. Also 1 when getexif raises (EXIF it cannot load, or text where it expects bytes: exif_transpose leaves the
    image as stored), and past the module docstring's bounds. `info` is a dict, or `_PngText`, which reads a value
    when asked for it. When exif_transpose fails afterwards, rewriting the EXIF once it has transposed (rare: crafted
    values), the pool serves the image as stored; reading that here would cost a rewrite of every sub-IFD on the
    event loop, so an orientation of 5..8 there fails the pool's size check instead."""
    from PIL import Image, ImageFile

    charge = getattr(info, "charge", None)  # _PngText counts this work against MAX_TEXT_WORK
    log = logging.getLogger("PIL.TiffImagePlugin")
    quiet, safe_read = log.disabled, getattr(ImageFile, "_safe_read", None)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # corrupt entries: the pool's decode warns, as vLLM's does
            log.disabled = True  # its debug record an entry (module docstring)
            exif = info.get("exif")
            if exif is None and (raw := info.get("Raw profile type exif")) is not None:
                if len(raw) > MAX_TEXT_CHUNK:
                    return 1
                exif = _raw_profile(raw, charge)
            e = Image.Exif()
            if exif is not None:
                entries = _ifd0_entries(exif)
                if entries > MAX_IFD0_ENTRIES:
                    return 1
                if charge:
                    charge(_ENTRY_COST * entries + len(exif) // 16)
                if safe_read is not None:
                    ImageFile._safe_read = _read_at_once
                e.load(exif)
            if 0x0112 in e:
                return _TRANSPOSED.get(_orientation_tag(e), 1)
            xmp = info.get("XML:com.adobe.xmp") or info.get("xmp")  # bytes: as getexif's match on the text
            if xmp and charge:
                charge(3 * min(len(xmp), MAX_TEXT_CHUNK) // 2)
            return _xmp_orientation(xmp) if xmp else 1
    except ProbeError:  # a bound of the module docstring, met reading the text
        raise
    except Exception:
        return 1
    finally:
        log.disabled = quiet
        if safe_read is not None:
            ImageFile._safe_read = safe_read


def _read_at_once(fp, size: int) -> bytes:
    """Pillow's `ImageFile._safe_read` on the BytesIO that Exif.load reads: the same bytes and the same error, in one
    read. Pillow's own reads a tag of more than a MiB in 1 MiB blocks and joins them, which on linux faults every block
    in afresh (a 15 MiB tag: about 20 ms). Set only while the probe's Exif.load runs (`orientation`)."""
    if size <= 0:
        return b""
    data = fp.read(size)
    if len(data) < size:
        raise OSError("Truncated File Read")
    return data


def _xmp_orientation(xmp: bytes) -> int:
    """exif_transpose's lookup of getexif's first `tiff:Orientation(="|>)([0-9])` in an XMP packet: each
    `tiff:Orientation` found by memchr in its first MAX_TEXT_CHUNK bytes, and what follows it; 1 after _XMP_DECOYS
    that do not match (module docstring)."""
    j = -len(_XMP_TAG)
    for _ in range(_XMP_DECOYS):
        j = xmp.find(_XMP_TAG, j + len(_XMP_TAG), MAX_TEXT_CHUNK)
        if j < 0:
            return 1
        k = j + len(_XMP_TAG)
        digit = xmp[k + 1:k + 2] if xmp[k:k + 1] == b">" else xmp[k + 2:k + 3] if xmp[k:k + 2] == b'="' else b""
        if digit.isdigit():
            return _TRANSPOSED.get(int(digit), 1)
    return 1


def _raw_profile(text: str, charge=None) -> bytes:
    """getexif's `bytes.fromhex("".join(text.split("\\n")[3:]))`, without a list of the lines or a str.replace (slow
    on text of many newlines): text other than ASCII is not hex, so a ValueError either way. `charge`: the hex parsed,
    two units a character (`_PngText.charge`)."""
    p = -1
    for _ in range(3):
        p = text.find("\n", p + 1)
        if p < 0:
            return b""
    if charge:
        charge(2 * (len(text) - p - 1))
    if not text.isascii():
        raise ValueError("a Raw profile's hex is ASCII")
    return bytes.fromhex(text[p + 1:].encode("ascii").translate(None, b"\n").decode("ascii"))


def _orientation_tag(e) -> object:
    """IFD0's Orientation as `e.get(0x0112)` gives it, from one value of the tag's data instead of all of them (a
    count filling the block would have Pillow unpack millions): BYTE, ASCII and UNDEFINED data loads as bytes or text,
    never 2..8, so 1; of the other types, Pillow keeps a one-value tag's first value whatever the count
    (`ImageFileDirectory_v2._setitem`), so the data cut to one unit gives the same. The tag's raw data is Pillow's
    private `_tagdata`: without it, the public lookup of every value."""
    ifd = getattr(e, "_info", None)
    raw = getattr(ifd, "_tagdata", None)
    if isinstance(raw, dict) and 0x0112 in raw:
        typ = ifd.tagtype.get(0x0112)
        if typ in (1, 2, 7):
            return 1
        unit = getattr(ifd, "_load_dispatch", {}).get(typ)
        if unit is not None:
            raw[0x0112] = raw[0x0112][:unit[0]]
    return e.get(0x0112, 1)


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


def _exif_block(b: bytes, s: int, e: int, prefixed: int = 0) -> bytes:
    """The EXIF block b[s:e], after `prefixed` `Exif\\0\\0` that Pillow's info already holds before it, cut once at
    the TIFF header: Exif.load strips up to two prefixes, by a copy (a fresh 20 MiB on linux is milliseconds of page
    faults), so the probe hands it the block without them when that is the same to Pillow; with a prefix past the
    two it strips, the block as Pillow's info holds it."""
    p = s
    for _ in range(2 - prefixed):
        if b.startswith(b"Exif\x00\x00", p, e):
            p += 6
    if not b.startswith(b"Exif\x00\x00", p, e):
        return b[p:e]
    return b"Exif\x00\x00" * prefixed + b[s:e]


# -- formats ---------------------------------------------------------------------------------------------------------


def _jpeg(b: bytes | memoryview) -> Header:
    """Markers up to the start of scan, walked as Pillow's `JpegImageFile._open` walks them (a byte other than 0xFF
    skipped, 0xFFFF padding, 0xFF00 skipped, a segment's length below 2 read as 2): the first SOF's size, the APP1
    Exif segments (concatenated, as Pillow's `exif`) and the last APP1 XMP segment (its `xmp`). A run of stray bytes or
    of fill is skipped at C speed (`bytes.find`, a regex match), and the walk takes at most MAX_JPEG_MARKERS (module
    docstring), a fill run counting one a KiB."""
    if not isinstance(b, bytes):
        b = bytes(b)
    n = len(b)
    last = n - 4  # the last offset at which a marker and its length fit
    i = 2
    size: tuple[int, int] | None = None
    exif: list[tuple[int, int]] = []  # the Exif segments' data, joined once at the end
    xmp: tuple[int, int] | None = None
    step, u16, markers = _JPEG_STEP, _U16, 0
    while i <= last:
        markers += 1
        if markers > MAX_JPEG_MARKERS:
            raise ProbeError(f"broken JPEG header: more than {MAX_JPEG_MARKERS} markers before the scan")
        if b[i] == 0xFF:
            k = step[b[i + 1]]
            if not k:  # a segment the probe does not read: the common case, kept short
                (seg,) = u16(b, i + 2)
                i += seg + 2 if seg > 1 else 4  # a length below 2 read as 2
            elif k > 2:
                if k == 5:
                    break
                if k == 6:
                    raise ProbeError("broken JPEG header: no marker found")
                (seg,) = u16(b, i + 2)
                end = i + 2 + max(2, seg)
                if k == 3:
                    if size is None:
                        if i + 9 > n:
                            raise ProbeError("broken JPEG header: truncated SOF")
                        h, w = struct.unpack_from(">HH", b, i + 5)
                        size = (w, h)
                elif b.startswith(b"Exif\x00\x00", i + 4, end):
                    exif.append((i + 10 if exif else i + 4, min(end, n)))  # Pillow keeps the first one's prefix
                elif b.startswith(b"http://ns.adobe.com/xap/1.0/\x00", i + 4, end):
                    xmp = (i + 33, min(end, n))
                i = end
            elif k == 1:
                i += 2
            else:  # fill: scanned no further than the markers left allow, a KiB each
                e = _FILL.match(b, i, i + ((MAX_JPEG_MARKERS - markers + 1) << 10)).end()
                markers += (e - i) >> 10
                if markers > MAX_JPEG_MARKERS:
                    raise ProbeError(f"broken JPEG header: more than {MAX_JPEG_MARKERS} markers before the scan")
                i = e - 1  # the run's last 0xFF starts the next marker
        else:
            i = b.find(b"\xff", i + 1)
            if i < 0:
                i = n
    else:  # fewer than 4 bytes left: no segment's length fits, as Pillow's walk reads on to the end
        while i + 2 <= n:
            if b[i] != 0xFF:
                i += 1
                continue
            k = step[b[i + 1]]
            if k == 6:
                raise ProbeError("broken JPEG header: no marker found")
            if k == 2 or k == 1:
                i += 3 - k
                continue
            break
    if size is None:
        raise ProbeError("broken JPEG header: no frame header before the scan")
    _check(*size, "JPEG")
    info: dict[str, bytes] = {}
    if xmp:
        info["xmp"] = b[xmp[0]:xmp[1]]
    if exif:
        view = memoryview(b)
        exif[0] = (exif[0][0] + 6, exif[0][1])  # one copy, without the prefix Exif.load would strip by copying
        block = b"".join(view[s:e] for s, e in exif)
        info["exif"] = block if not block.startswith(b"Exif\x00\x00") else b"Exif\x00\x00" + block
    return Header("jpeg", size[0], size[1], orientation(info))


def _png(b: bytes | memoryview) -> Header:
    """IHDR, then every chunk header (the image data never decompressed), noting the `eXIf` chunks and the text
    chunks of the keys getexif reads (`_PngText` reads them when asked). Pillow reads the chunks after the image data
    too when it applies the orientation, and stops at a malformed chunk type, or, in an animated PNG, at the next
    frame's fcTL."""
    if not isinstance(b, bytes):
        b = bytes(b)
    n = len(b)
    if n < 33 or b[12:16] != b"IHDR":
        raise ProbeError("broken PNG header: no IHDR")
    w, h = struct.unpack_from(">II", b, 16)
    _check(w, h, "PNG")
    found = _PngText(b)
    chunks = found.chunks
    frames: int | None = None  # acTL's frame count (Pillow's im_n_frames: a second acTL clears it)
    fctl = False  # a fcTL before the image data: it is the first frame, not a default image
    animated: bool | None = None  # decided at the image data, as Pillow decides it at open
    kinds, u32, keys, bare, text = _PNG_KIND, _U32, _PNG_TEXT_KEYS, _PNG_BARE_KEYS, _PNG_TEXT
    i = 8
    while i + 8 <= n:
        (length,) = u32(b, i)
        kind = b[i + 4:i + 8]
        t = kinds.get(kind)
        if t is None:  # a chunk the probe does not read; a malformed type ends Pillow's walk (its is_cid: \w{4})
            if not kind.isalnum() and not kind.replace(b"_", b"a").isalnum():
                break
        elif t == 2:
            p = i + 8
            if b.startswith(keys, p, p + length) or (length in bare and b[p:p + length] == bare[length]):
                key, which = text[b[p]]
                m = p + len(key)  # the compression method, when the chunk holds one
                if kind == b"zTXt" and length > len(key) and m < n and b[m]:
                    raise ProbeError(f"unknown compression method {b[m]} in a PNG zTXt chunk")  # Pillow's
                chunks[which].append(i)
        elif t == 1:
            chunks[0].append(i)
        elif not t:
            break
        elif animated is None:
            if t == 3:
                if length >= 8 and i + 12 <= n:
                    (count,) = u32(b, i + 8)
                    frames = None if frames is not None else count if 0 < count <= 0x80000000 else None
            elif t == 4:
                fctl = True
            else:  # the image data (Pillow takes a first fdAT as it)
                animated = frames is not None and frames + (not fctl) > 1
        elif t == 4 and animated:
            break
        i += 12 + length
    return Header("png", w, h, orientation(found))


class _PngText:
    """A PNG's metadata as Pillow's chunk handlers leave it in the image's info (`PngStream.chunk_eXIf`, `chunk_tEXt`,
    `chunk_zTXt`, `chunk_iTXt`), read for a key when `orientation` asks for it: from the key's last chunk back, as the
    last chunk that sets a value wins. `exif` comes as bytes (eXIf, tEXt), or as "" for text (zTXt, iTXt: getexif
    then raises); "Raw profile type exif" as text; `XML:com.adobe.xmp` and `xmp` as bytes, whose XMP match is the
    text's. tEXt and zTXt always set their value (zTXt: "" for a stream zlib refuses); iTXt sets nothing for another
    compression method, a stream zlib refuses or a text that is not UTF-8, but records an XMP packet as `xmp` before
    that check. Within MAX_TEXT_WORK (module docstring); the chunks noted per key are the last that the bound lets it
    read."""

    def __init__(self, b: bytes) -> None:
        self.b = b
        n = MAX_TEXT_WORK // _TEXT_CHUNK_COST + 1  # more would pass the bound before the first is reached
        self.chunks: tuple[deque[int], ...] = tuple(deque(maxlen=n) for _ in range(4))
        self.work = 0
        self.values: dict[str, object] = {}
        self.where: dict[int, tuple[bytes, int, int]] = {}
        self.read: dict[int, tuple[bytes, bytes]] = {}
        self.itxt: dict[int, tuple[bytes, bytes, bytes] | None] = {}

    def get(self, key: str, default: object = None) -> object:
        if key not in self.values:
            self.values[key] = {"exif": self._exif, "Raw profile type exif": self._raw,
                                "XML:com.adobe.xmp": self._xml, "xmp": self._xmp}[key]()
        v = self.values[key]
        return default if v is None else v

    def _exif(self) -> bytes | str | None:
        for i in reversed(self.chunks[0]):
            kind, s, e = self._at(i)
            if kind in (b"eXIf", b"tEXt"):  # eXIf: Pillow's info holds `Exif\0\0` and the data
                return _exif_block(self.b, s, e, kind == b"eXIf")
            kind, v = self._chunk(i)
            if kind == b"zTXt":
                self._inflate(v[1:])  # its text, whatever it holds: as Pillow refuses past MAX_TEXT_CHUNK
                return ""
            if (parts := self._itxt_parts(i)) is not None and self._utf8(*parts):
                return ""
        return None

    def _raw(self) -> str | None:
        for i in reversed(self.chunks[1]):
            kind, v = self._chunk(i)
            if kind == b"tEXt":
                return v.decode("latin-1")
            if kind == b"zTXt":
                return (self._inflate(v[1:]) or b"").decode("latin-1")
            if (parts := self._itxt_parts(i)) is not None and self._utf8(*parts[:2]):
                self._spend(len(parts[2]) >> 5)
                if not parts[2].isascii():
                    self._spend(_DECODE_COST * len(parts[2]))
                try:
                    return parts[2].decode("utf-8")  # decoded once: the text getexif reads
                except UnicodeError:
                    continue
        return None

    def _xml(self) -> bytes | None:
        chunks = self.chunks[2]
        for k in range(len(chunks) - 1, -1, -1):
            kind, v = self._chunk(chunks[k])
            if kind == b"tEXt":
                return v
            if kind == b"zTXt":
                return self._inflate(v[1:]) or b""
            parts = self._itxt_parts(chunks[k])
            if parts is None:
                continue
            if k == 0 and self.get("xmp") is parts[2]:
                return parts[2]  # the first XMP chunk, and the last that inflates: valid or not, getexif matches it
            if self._utf8(*parts):
                return parts[2]
        return None

    def _xmp(self) -> bytes | str | None:
        """An iTXt XMP packet's bytes, or the text of a chunk keyed `xmp` (getexif's bytes match then raises: 1)."""
        xml = set(self.chunks[2])
        for i in sorted(xml.union(self.chunks[3]), reverse=True):
            kind, v = self._chunk(i)
            if i in xml:
                if kind == b"iTXt" and (parts := self._itxt_parts(i)) is not None:
                    return parts[2]
            elif kind == b"tEXt":
                return v.decode("latin-1")
            elif kind == b"zTXt":
                return (self._inflate(v[1:]) or b"").decode("latin-1")
            elif (parts := self._itxt_parts(i)) is not None and self._utf8(*parts):
                return ""
        return None

    def _at(self, i: int) -> tuple[bytes, int, int]:
        """The chunk at `i`: its type and where its data lies (a text chunk's after its key's NUL), counted once."""
        if i not in self.where:
            self._spend(_TEXT_CHUNK_COST)
            b = self.b
            (length,) = _U32(b, i)
            kind = b[i + 4:i + 8]
            key = 0 if kind == b"eXIf" else len(_PNG_TEXT[b[i + 8]][0])
            self.where[i] = (kind, min(i + 8 + key, len(b)), min(i + 8 + length, len(b)))
        return self.where[i]

    def _chunk(self, i: int) -> tuple[bytes, bytes]:
        """The chunk at `i`: its type and its data, cut once."""
        if i not in self.read:
            kind, s, e = self._at(i)
            self.read[i] = (kind, self.b[s:e])
        return self.read[i]

    def _itxt_parts(self, i: int) -> tuple[bytes, bytes, bytes] | None:
        """An iTXt chunk's language, translated key and text (inflated), or None when Pillow sets nothing before its
        UTF-8 check; read once."""
        if i not in self.itxt:
            kind, r = self.read[i]
            parts = None
            j = r.find(b"\0", 2)  # Pillow's r[2:].split(b"\0", 2), with memchr
            k = r.find(b"\0", j + 1) if j >= 0 else -1
            if kind == b"iTXt" and k >= 0:
                v: bytes | None = r[k + 1:]
                if r[0]:
                    v = self._inflate(v) if r[1] == 0 else None
                if v is not None:
                    parts = (r[2:j], r[j + 1:k], v)
            self.itxt[i] = parts
        return self.itxt[i]

    def _utf8(self, *texts: bytes) -> bool:
        """Whether each is UTF-8, as Pillow's strict decode reads it: ASCII at once, other text counted as work."""
        for x in texts:
            self._spend(len(x) >> 5)
            if not x.isascii():
                self._spend(_DECODE_COST * len(x))
                try:
                    x.decode("utf-8")
                except UnicodeError:
                    return False
        return True

    def _inflate(self, data: bytes) -> bytes | None:
        """Pillow's `_safe_zlib_decompress`: at most MAX_TEXT_CHUNK bytes (a `ProbeError` past them), None for a
        stream zlib refuses; the input fed in pieces and the output taken in steps, each counted (module docstring)."""
        d = zlib.decompressobj()
        parts: list[bytes] = []
        out = pos = 0
        tail = b""
        while not d.eof:
            if not tail and pos < len(data):
                tail = data[pos:pos + _TEXT_FEED]
                pos += len(tail)
                self._spend(_FED_COST * len(tail))
            fed = not tail and pos >= len(data)  # zlib holds the whole input, and may still owe output
            if fed and out >= MAX_TEXT_CHUNK:
                break  # Pillow's one call ends there too, with no input left over: no error
            room = min(_INFLATE_STEP, MAX_TEXT_CHUNK - out) or 1  # past the cap, a byte more is Pillow's error
            try:
                part = d.decompress(tail, room)
            except zlib.error:
                self._spend(room)
                return None
            self._spend(len(part))
            out += len(part)
            if out > MAX_TEXT_CHUNK:
                raise ProbeError(f"a PNG text chunk inflates past {MAX_TEXT_CHUNK >> 20} MiB")
            parts.append(part)
            tail = d.unconsumed_tail
            if fed and not part:
                break
        return b"".join(parts)

    def charge(self, n: int) -> None:
        """Work `orientation` does with these values (loading the EXIF, searching the XMP), counted as text work."""
        self._spend(n)

    def _spend(self, n: int) -> None:
        self.work += n
        if self.work > MAX_TEXT_WORK:
            raise ProbeError("PNG text takes more work to read than the probe's bound (MAX_TEXT_WORK)")


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
    if not isinstance(b, bytes):
        b = bytes(b)
    n = min(len(b), 8 + struct.unpack_from("<I", b, 4)[0])  # libwebp reads no further than the RIFF's size
    i = 12
    size: tuple[int, int] | None = None
    flags = b[20] if n >= 30 and b[12:16] == b"VP8X" else 0  # the first chunk's
    info: dict[str, bytes] = {}
    seen: set[bytes] = set()
    u32 = _U32LE
    while i + 8 <= n:
        kind = b[i:i + 4]
        (length,) = u32(b, i + 4)
        d = i + 8
        if kind == b"VP8X" and d + 10 <= n:
            w = 1 + (b[d + 4] | b[d + 5] << 8 | b[d + 6] << 16)
            h = 1 + (b[d + 7] | b[d + 8] << 8 | b[d + 9] << 16)
            size = (w, h)
        elif kind == b"VP8 " and size is None and d + 10 <= n:
            if b[d + 3:d + 6] != b"\x9d\x01\x2a":
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
                info["exif" if kind == b"EXIF" else "xmp"] = _exif_block(b, d, d + length) if kind == b"EXIF" else \
                    b[d:d + length]
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
