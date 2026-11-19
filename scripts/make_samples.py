"""Generate sample test images, bundled comic font, and Edge extension icons."""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
USER_UPLOADED_JP = Path(
    r"C:\Users\ecmat\.gemini\antigravity\brain\a14219b1-1d67-4a8f-970a-a1ad8b0dcea1\.user_uploaded\media_1795062710266_cdab5fd3.jpg"
)

FONT_URLS = [
    "https://cdn.jsdelivr.net/npm/@fontsource/comic-neue@5.1.0/files/comic-neue-latin-700-normal.woff2",
    "https://cdn.jsdelivr.net/npm/@fontsource/comic-neue/files/comic-neue-latin-700-normal.woff2",
    "https://github.com/crozynski/comicneue/raw/master/Web/ComicNeue-Bold.woff2",
]


def copy_japanese_sample() -> Path:
    dest = PROJECT_ROOT / "assets" / "samples" / "jp_centipede.jpg"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if USER_UPLOADED_JP.exists():
        shutil.copy2(USER_UPLOADED_JP, dest)
        print(f"Copied Japanese sample -> {dest}")
    elif dest.exists():
        print(f"Japanese sample already present -> {dest}")
    else:
        print(f"WARNING: Source Japanese sample not found at {USER_UPLOADED_JP}")
    return dest


def _load_hangul_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        Path(r"C:\Windows\Fonts\malgunbd.ttf"),
        Path(r"C:\Windows\Fonts\malgun.ttf"),
    ):
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _draw_speech_bubble(
    draw: ImageDraw.ImageDraw,
    bbox: tuple[int, int, int, int],
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    tail_points: list[tuple[int, int]] | None = None,
) -> None:
    x1, y1, x2, y2 = bbox
    if tail_points:
        draw.polygon(tail_points, fill="#FFFFFF", outline="#111111")
    draw.rounded_rectangle(
        [x1, y1, x2, y2],
        radius=36,
        fill="#FFFFFF",
        outline="#111111",
        width=4,
    )
    if tail_points:
        # Cover internal stroke line at tail base
        cx = (tail_points[0][0] + tail_points[2][0]) // 2
        draw.line(
            [(tail_points[0][0] + 3, y2 - 2), (tail_points[2][0] - 3, y2 - 2)],
            fill="#FFFFFF",
            width=6,
        )
        _ = cx

    text_bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=10, align="center")
    tw = text_bbox[2] - text_bbox[0]
    th = text_bbox[3] - text_bbox[1]
    tx = x1 + (x2 - x1 - tw) // 2 - text_bbox[0]
    ty = y1 + (y2 - y1 - th) // 2 - text_bbox[1]
    draw.multiline_text((tx, ty), text, font=font, fill="#111111", spacing=10, align="center")


def generate_korean_webtoon_strip() -> Path:
    dest = PROJECT_ROOT / "assets" / "samples" / "ko_webtoon_strip.png"
    dest.parent.mkdir(parents=True, exist_ok=True)

    width, height = 800, 3800
    img = Image.new("RGB", (width, height), "#F4F5F8")
    draw = ImageDraw.Draw(img)

    # Draw 4 clean webtoon panels so it resembles a vertical scroll strip
    panels = [
        (40, 60, 760, 720, "#E9EEF6"),
        (40, 780, 760, 1320, "#F5EFEA"),
        (40, 1380, 760, 2600, "#EAECEF"),
        (40, 2680, 760, 3720, "#EDF2F7"),
    ]
    for px1, py1, px2, py2, bg in panels:
        draw.rectangle([px1, py1, px2, py2], fill=bg, outline="#22252A", width=3)

    # Subtle decorative panel silhouettes well clear of the dialogue bubbles
    draw.ellipse([540, 470, 720, 690], fill="#CBD5E1")
    draw.rectangle([80, 1130, 260, 1290], fill="#D6CCC2")
    draw.ellipse([260, 1820, 540, 2320], fill="#CBD5E1")
    draw.rectangle([100, 3380, 700, 3660], fill="#D1D9E6")

    font = _load_hangul_font(size=32)

    # Bubble 1: Slice 0
    _draw_speech_bubble(
        draw,
        (110, 170, 670, 400),
        "지금 바로 시작하자!\n늦기 전에!",
        font,
        tail_points=[(520, 398), (575, 455), (565, 398)],
    )

    # Bubble 2: Slice 0
    _draw_speech_bubble(
        draw,
        (130, 860, 670, 1080),
        "잠깐만, 저기 뭔가 있어...",
        font,
        tail_points=[(200, 1078), (175, 1135), (255, 1078)],
    )

    # Bubble 3: Placed right across the y=1520 slice seam (and 1600px chunk overlap)
    _draw_speech_bubble(
        draw,
        (100, 1415, 700, 1625),
        "조심해! 지네가 다가오고 있어!",
        font,
        tail_points=[(380, 1623), (400, 1685), (435, 1623)],
    )

    # Bubble 4: Slice 2
    _draw_speech_bubble(
        draw,
        (120, 2980, 680, 3220),
        "빨리 도망쳐! 지금 당장!",
        font,
        tail_points=[(360, 3218), (390, 3285), (425, 3218)],
    )

    img.save(dest, format="PNG")
    print(f"Generated Korean webtoon strip ({width}x{height}) -> {dest}")
    return dest


def bundle_comic_font() -> None:
    assets_font = PROJECT_ROOT / "assets" / "fonts" / "ComicNeue-Bold.woff2"
    ext_font = PROJECT_ROOT / "extension" / "fonts" / "ComicNeue-Bold.woff2"
    assets_font.parent.mkdir(parents=True, exist_ok=True)
    ext_font.parent.mkdir(parents=True, exist_ok=True)

    if assets_font.exists() and assets_font.read_bytes()[:4] == b"wOF2":
        shutil.copy2(assets_font, ext_font)
        print(f"ComicNeue-Bold.woff2 already present -> {ext_font}")
        return

    for url in FONT_URLS:
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = resp.read()
            if data[:4] == b"wOF2":
                assets_font.write_bytes(data)
                ext_font.write_bytes(data)
                print(f"Downloaded ComicNeue-Bold.woff2 ({len(data)} bytes)")
                return
        except Exception as exc:
            print(f"Font download attempt failed ({url}): {exc}")

    print("WARNING: Could not download ComicNeue-Bold.woff2; CSS will fall back to local comic/sans fonts.")


def generate_extension_icons() -> None:
    icons_dir = PROJECT_ROOT / "extension" / "icons"
    icons_dir.mkdir(parents=True, exist_ok=True)

    # Supersample at 512x512 then downsample cleanly to 16, 48, 128
    base_size = 512
    canvas = Image.new("RGBA", (base_size, base_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    # Rounded indigo badge
    draw.rounded_rectangle(
        [24, 24, 488, 488],
        radius=110,
        fill=(37, 41, 88, 255),
        outline=(110, 122, 247, 255),
        width=14,
    )

    # Crisp white speech bubble with tail
    draw.polygon(
        [(165, 335), (125, 415), (235, 345)],
        fill=(255, 255, 255, 255),
    )
    draw.rounded_rectangle(
        [92, 108, 420, 355],
        radius=72,
        fill=(255, 255, 255, 255),
        outline=(17, 24, 39, 255),
        width=12,
    )

    # Bold "A" glyph inside bubble
    font_path = Path(r"C:\Windows\Fonts\segoeuib.ttf")
    if not font_path.exists():
        font_path = Path(r"C:\Windows\Fonts\arialbd.ttf")
    font = (
        ImageFont.truetype(str(font_path), size=170)
        if font_path.exists()
        else ImageFont.load_default()
    )
    tb = draw.textbbox((0, 0), "A", font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    tx = 256 - tw // 2 - tb[0]
    ty = 230 - th // 2 - tb[1]
    draw.text((tx, ty), "A", font=font, fill=(37, 41, 88, 255))

    for size in (16, 48, 128):
        out_path = icons_dir / f"icon{size}.png"
        resized = canvas.resize((size, size), Image.Resampling.LANCZOS)
        resized.save(out_path, format="PNG")
        print(f"Generated extension icon -> {out_path}")


def main() -> None:
    copy_japanese_sample()
    generate_korean_webtoon_strip()
    bundle_comic_font()
    generate_extension_icons()


if __name__ == "__main__":
    main()
