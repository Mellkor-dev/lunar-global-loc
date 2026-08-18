#!/usr/bin/env python3
"""Build an A0-proportioned poster mockup from versioned experiment outputs."""

from pathlib import Path
import csv
import textwrap

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "poster"
W, H = 2480, 3508
NAVY = "#0b2f59"
BLUE = "#1479b8"
CYAN = "#00a9ce"
ORANGE = "#f08a24"
INK = "#17212b"
MUTED = "#526170"
PALE = "#eef5f8"
GREEN = "#237a57"


def font(size, bold=False):
    names = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for name in names:
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def fit_text(draw, text, box, size, color=INK, bold=False, spacing=8, align="left"):
    x, y, w, h = box
    f = font(size, bold)
    avg = max(8, int(w / max(size * 0.54, 1)))
    lines = []
    for para in text.split("\n"):
        lines += textwrap.wrap(para, width=avg) or [""]
    draw.multiline_text((x, y), "\n".join(lines), font=f, fill=color, spacing=spacing, align=align)


def crop_whitespace(im):
    im = im.convert("RGB")
    px = im.load(); xs=[]; ys=[]
    for y in range(0, im.height, 4):
        for x in range(0, im.width, 4):
            if min(px[x, y]) < 245:
                xs.append(x); ys.append(y)
    if not xs:
        return im
    pad = 18
    return im.crop((max(0,min(xs)-pad), max(0,min(ys)-pad), min(im.width,max(xs)+pad), min(im.height,max(ys)+pad)))


def place(im, src, box, contain=True):
    x, y, w, h = box
    art = crop_whitespace(Image.open(src))
    scale = min(w/art.width, h/art.height) if contain else max(w/art.width, h/art.height)
    art = art.resize((int(art.width*scale), int(art.height*scale)), Image.Resampling.LANCZOS)
    if not contain:
        left=max(0,(art.width-w)//2); top=max(0,(art.height-h)//2)
        art=art.crop((left,top,left+w,top+h))
    im.paste(art, (x+(w-art.width)//2, y+(h-art.height)//2))


def heading(draw, x, y, w, title, n):
    draw.rounded_rectangle((x,y,x+54,y+54), 12, fill=CYAN)
    draw.text((x+17,y+8), str(n), font=font(29,True), fill="white")
    draw.text((x+72,y+4), title, font=font(38,True), fill=NAVY)
    draw.line((x+72,y+57,x+w,y+57), fill=CYAN, width=5)


def make_resolution_chart():
    rows=[]
    with open(ROOT/"LargeScale_Implement/results/diagnostics/pipeline_summary.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["stage"]=="DARCES" and r["resolution"] in {"0p5m","1m","2m","5m","10m"}:
                rows.append(r)
    labels=[r["resolution"].replace("p", ".") for r in rows]
    availability=[float(r["availability_percent"]) for r in rows]
    median=[float(r["median_xy_error_m"]) for r in rows]
    fig, (a,b)=plt.subplots(1,2,figsize=(10.5,3.8),dpi=180)
    colors=["#aab9c5"]*len(labels); colors[3]=CYAN
    a.bar(labels, availability, color=colors)
    a.set_ylim(0,100); a.set_ylabel("sites localized [%]"); a.set_xlabel("DEM resolution [m/cell]")
    a.set_title("Availability peaks at 5 m")
    for i,v in enumerate(availability): a.text(i,v+2,f"{v:.0f}%",ha="center",fontsize=8)
    b.plot(labels,median,"o-",color=ORANGE,lw=2.5)
    b.set_yscale("log"); b.set_ylabel("median XY error [m, log]"); b.set_xlabel("DEM resolution [m/cell]")
    b.set_title("Typical solved-site error")
    b.scatter([3],[median[3]],s=110,facecolors="none",edgecolors=CYAN,lw=2.5)
    b.annotate("4.96 m",(3,median[3]),xytext=(10,12),textcoords="offset points",fontsize=9,fontweight="bold")
    for ax in (a,b):
        ax.spines[["top","right"]].set_visible(False); ax.grid(axis="y",alpha=.22)
    fig.tight_layout()
    p=OUT/"resolution_tradeoff.png"; fig.savefig(p,bbox_inches="tight",facecolor="white"); plt.close(fig)
    return p


def main():
    OUT.mkdir(exist_ok=True)
    chart=make_resolution_chart()
    im=Image.new("RGB",(W,H),"white"); d=ImageDraw.Draw(im)
    margin=110; gap=46; col=(W-2*margin-gap)//2

    # Header
    d.rectangle((0,0,W,335),fill=NAVY)
    d.rectangle((0,318,W,335),fill=CYAN)
    d.text((margin,55),"LUNAR CROSS-VIEW LOCALIZATION",font=font(67,True),fill="white")
    d.text((margin,139),"How map resolution controls crater-based global pose estimation",font=font(39),fill="#cdeaf4")
    d.text((margin,225),"Soumyadeep Chatterjee  ·  Machine Perception and Intelligence Lab  ·  GIST",font=font(27),fill="white")
    d.text((W-margin,78),"APOLLO 17",font=font(30,True),fill="#cdeaf4",anchor="ra")
    d.text((W-margin,126),"28 SITES",font=font(30,True),fill="white",anchor="ra")
    d.text((W-margin,174),"3 STAGES",font=font(30,True),fill="white",anchor="ra")

    # Hero finding
    y=375
    d.rounded_rectangle((margin,y,W-margin,y+220),24,fill=PALE,outline="#c7dbe5",width=3)
    d.text((margin+38,y+28),"KEY RESULT",font=font(27,True),fill=BLUE)
    d.text((margin+38,y+70),"5 m/cell gives the best balance",font=font(48,True),fill=NAVY)
    d.text((margin+38,y+136),"21/28 sites localized  ·  4.96 m median error  ·  13/21 solutions within 50 m",font=font(30),fill=INK)
    d.text((W-margin-30,y+35),"Typical accuracy is strong,\nbut the error tail is heavy.",font=font(29,True),fill=ORANGE,anchor="ra",spacing=8)

    # Row 1
    y=645; heading(d,margin,y,col,"Problem & experimental setup",1); heading(d,margin+col+gap,y,col,"Why resolution matters",2)
    y+=83
    fit_text(d,"Goal: recover a rover's global XY pose by matching crater-like terrain features seen in a local LiDAR elevation map to features extracted from an orbital DEM.", (margin,y,col,140),31)
    fit_text(d,"Apollo 17 terrain · synchronized 28-site traverse · DEMs from 0.25 to 10 m/cell · identical sites across resolutions · DARCES → RANSAC → MOGA evaluated against held-out truth.",(margin,y+145,col,180),29,color=MUTED)
    place(im,ROOT/"LargeScale_Implement/plots/dem_resolution_validation/dem_comparison_10m.png",(margin+col+gap,y,col,350),contain=False)
    d.text((margin+col+gap,y+358),"Fine DEM sources agree after common-grid resampling; localization still changes sharply with working resolution.",font=font(23),fill=MUTED)

    # Pipeline ribbon
    y=1160; heading(d,margin,y,W-2*margin,"Visual pipeline",3); y+=90
    steps=[("ORBITAL DEM","global terrain"),("CRATERS","feature catalog"),("LOCAL LiDAR","rover-centric map"),("DARCES","coarse hypotheses"),("RANSAC + MOGA","verify & refine")]
    sw=(W-2*margin-4*34)//5
    for i,(a,b) in enumerate(steps):
        x=margin+i*(sw+34)
        d.rounded_rectangle((x,y,x+sw,y+125),18,fill="#f5f8fa",outline="#b9ccd6",width=3)
        d.text((x+sw//2,y+24),a,font=font(28,True),fill=NAVY,anchor="ma")
        d.text((x+sw//2,y+72),b,font=font(22),fill=MUTED,anchor="ma")
        if i<4: d.polygon([(x+sw+8,y+52),(x+sw+27,y+62),(x+sw+8,y+72)],fill=CYAN)

    # Evidence images
    y=1430; heading(d,margin,y,col,"What the rover and map see",4); heading(d,margin+col+gap,y,col,"A successful cross-view match",5); y+=82
    half=(col-20)//2
    place(im,ROOT/"LargeScale_Implement/plots/5m_px/global_features_preview.png",(margin,y,half,560))
    place(im,ROOT/"LargeScale_Implement/plots/5m_px/local_features/local_craters_site_03.png",(margin+half+20,y,half,560))
    d.text((margin,y+565),"Global catalog (625 craters)",font=font(23,True),fill=NAVY)
    d.text((margin+half+20,y+565),"Site 03 local scan (14 craters)",font=font(23,True),fill=NAVY)
    place(im,ROOT/"LargeScale_Implement/plots/5m_px/ransac/ransac_site_03.png",(margin+col+gap,y,col,560))
    d.text((margin+col+gap,y+565),"Site 03: five geometrically consistent correspondences; estimate lies near evaluation truth.",font=font(23),fill=MUTED)

    # Results
    y=2130; heading(d,margin,y,W-2*margin,"Resolution trade-off across all 28 sites",6); y+=82
    place(im,chart,(margin,y,W-2*margin,500))

    # Traversal and interpretation
    y=2740; heading(d,margin,y,col+250,"Where localization succeeds—and fails",7); heading(d,margin+col+gap+250,y,W-margin-(margin+col+gap+250),"Takeaways",8); y+=82
    place(im,ROOT/"LargeScale_Implement/plots/5m_px/traversal/darces_traversal_map.png",(margin,y,col+250,490),contain=False)
    tx=margin+col+gap+250; tw=W-margin-tx
    take=[
      ("01","Use 5 m/cell as the operating point","Highest availability (75%) and low median error on solved sites."),
      ("02","Report availability with accuracy","Median alone hides unresolved sites and large false matches."),
      ("03","Refinement is not the main bottleneck","RANSAC/MOGA change aggregate error only slightly; coarse data association dominates."),
      ("04","Next: reject perceptual aliasing","Add confidence/consistency gates before accepting a global pose."),
    ]
    yy=y
    for n,a,b in take:
        d.ellipse((tx,yy,tx+55,yy+55),fill=ORANGE)
        d.text((tx+27,yy+13),n,font=font(22,True),fill="white",anchor="ma")
        d.text((tx+75,yy),a,font=font(26,True),fill=NAVY)
        fit_text(d,b,(tx+75,yy+39,tw-75,86),22,color=MUTED)
        yy+=118

    d.line((margin,3390,W-margin,3390),fill="#b9ccd6",width=3)
    d.text((margin,3420),"Metrics are conditional on available solutions. Truth is used for evaluation only, not localization.",font=font(22),fill=MUTED)
    d.text((W-margin,3420),"Data and plots: versioned repository outputs",font=font(22,True),fill=NAVY,anchor="ra")

    p=OUT/"lunar_localization_poster_mockup.png"; im.save(p,quality=95)
    im.resize((1240,1754),Image.Resampling.LANCZOS).save(OUT/"lunar_localization_poster_preview.png")
    print(p)


if __name__ == "__main__": main()
