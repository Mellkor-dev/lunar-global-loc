#!/usr/bin/env python3
"""Concise visual showing how DEM resolution changes global/local features."""

from pathlib import Path
from PIL import Image, ImageChops, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "poster" / "dem_resolution_effect.png"
W, H = 1800, 1120
NAVY = "#0b2f59"
CYAN = "#00a9ce"
ORANGE = "#f08a24"
INK = "#17212b"
MUTED = "#5a6976"
PALE = "#edf5f8"


def font(size, bold=False):
    p = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    return ImageFont.truetype(p, size)


def crop_plot(path):
    im = Image.open(path).convert("RGB")
    # Remove broad white plot margins while preserving axes/color information.
    bg = Image.new("RGB", im.size, "white")
    diff = ImageChops.difference(im, bg).convert("L").point(lambda x: 255 if x > 12 else 0)
    box = diff.getbbox()
    return im.crop(box) if box else im


def place_cover(canvas, path, box):
    x, y, w, h = box
    art = crop_plot(path)
    scale = max(w/art.width, h/art.height)
    art = art.resize((int(art.width*scale), int(art.height*scale)), Image.Resampling.LANCZOS)
    left = max(0, (art.width-w)//2); top = max(0, (art.height-h)//2)
    canvas.paste(art.crop((left, top, left+w, top+h)), (x, y))


def main():
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    d.rectangle((0,0,W,150), fill=NAVY)
    d.text((70,34), "DEM RESOLUTION CHANGES WHAT CAN BE MATCHED", font=font(46,True), fill="white")
    d.text((72,94), "Same Apollo 17 terrain · same Site 03 scan", font=font(25), fill="#cceaf4")

    configs = [
        ("FINE", "0.25 m/cell", "0p25m_px", 838, 2, "0 / 28", "too few local features", "#9aaeba"),
        ("BALANCED", "5 m/cell", "5m_px", 625, 14, "21 / 28", "best observed balance", CYAN),
        ("COARSE", "10 m/cell", "10m_px", 431, 10, "15 / 28", "terrain distinctions fade", ORANGE),
    ]
    margin=60; gap=28; cw=(W-2*margin-2*gap)//3; top=180
    for i,(tag,res,key,glob,local,avail,note,color) in enumerate(configs):
        x=margin+i*(cw+gap)
        d.rounded_rectangle((x,top,x+cw,H-90), 22, fill="#f8fafb", outline=color, width=5)
        d.rounded_rectangle((x+22,top+20,x+180,top+68), 15, fill=color)
        d.text((x+101,top+31),tag,font=font(22,True),fill="white",anchor="ma")
        d.text((x+cw-22,top+25),res,font=font(29,True),fill=NAVY,anchor="ra")

        d.text((x+22,top+94),"GLOBAL MAP",font=font(20,True),fill=MUTED)
        gp=ROOT/f"LargeScale_Implement/plots/{key}/global_features_preview.png"
        place_cover(im,gp,(x+22,top+124,cw-44,250))
        d.rectangle((x+22,top+326,x+cw-22,top+374),fill=(255,255,255))
        d.text((x+cw//2,top+338),f"{glob} detected features",font=font(24,True),fill=NAVY,anchor="ma")

        d.text((x+22,top+402),"LOCAL SCAN · SITE 03",font=font(20,True),fill=MUTED)
        lp=ROOT/f"LargeScale_Implement/plots/{key}/local_features/local_craters_site_03.png"
        place_cover(im,lp,(x+22,top+432,cw-44,250))
        d.rectangle((x+22,top+634,x+cw-22,top+682),fill=(255,255,255))
        d.text((x+cw//2,top+646),f"{local} detected features",font=font(24,True),fill=NAVY,anchor="ma")

        d.rounded_rectangle((x+22,top+710,x+cw-22,top+805),18,fill=PALE)
        d.text((x+42,top+729),"SITES LOCALIZED",font=font(18,True),fill=MUTED)
        d.text((x+cw-42,top+720),avail,font=font(35,True),fill=color,anchor="ra")
        d.text((x+cw//2,top+780),note,font=font(20,True),fill=INK,anchor="ma")

    # Bottom one-line takeaway and balance motif.
    d.line((90,H-58,560,H-58),fill="#a9bbc5",width=4)
    d.line((1240,H-58,1710,H-58),fill="#a9bbc5",width=4)
    d.text((W//2,H-72),"MATCHING NEEDS BOTH: enough local features  +  distinctive global terrain",font=font(25,True),fill=NAVY,anchor="ma")
    OUT.parent.mkdir(exist_ok=True)
    im.save(OUT)
    im.resize((1200,747),Image.Resampling.LANCZOS).save(OUT.with_name("dem_resolution_effect_preview.png"))
    print(OUT)


if __name__ == "__main__":
    main()
