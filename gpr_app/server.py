"""
GPR B-Scan Analyser — Backend Server
"""
import subprocess
subprocess.run(["pip", "install", "gdown", "-q"])
import gdown
import os

if not os.path.exists("best.pt") or os.path.getsize("best.pt") < 1000000:
    print("Downloading best.pt...")
    gdown.download(id="1q-BV-7_JvOfiyol4Aa8udG9A9kJTF5hO", output="best.pt", quiet=False)

if not os.path.exists("classifier_best.pth") or os.path.getsize("classifier_best.pth") < 1000000:
    print("Downloading classifier_best.pth...")
    gdown.download(id="15xEBkXlSOdh6QmY7QMk90YQ1wjFtodbp", output="classifier_best.pth", quiet=False)
import io
import os
import base64
import numpy as np
from pathlib import Path

# Force headless / no-display BEFORE any cv2 or torch import
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
os.environ["QT_QPA_PLATFORM"]          = "offscreen"
os.environ["DISPLAY"]                  = ""
os.environ["MPLBACKEND"]               = "Agg"

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
from torchvision import transforms, models

app = FastAPI(title="GPR B-Scan Analyser")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config ───────────────────────────────────────────────────────────────────
CLASS_NAMES    = ["cavity", "intact", "utility"]
DEVICE         = torch.device("cpu")   # Railway has no GPU
BASE_DIR       = Path(__file__).parent
YOLO_WEIGHTS   = str(BASE_DIR / "best.pt")
CLF_WEIGHTS    = str(BASE_DIR / "classifier_best.pth")
CONF_THRESHOLD = 0.25

BOX_COLORS = {
    "cavity":  (226, 75,  74,  220),
    "utility": (55,  138, 221, 220),
    "intact":  (99,  153, 34,  220),
}
LABEL_BG = {
    "cavity":  (226, 75,  74),
    "utility": (55,  138, 221),
    "intact":  (99,  153, 34),
}

# ── Load models at startup ───────────────────────────────────────────────────
yolo_model = None
clf_model  = None

def load_models():
    global yolo_model, clf_model

    # YOLOv8 — import here so env vars are set first
    if Path(YOLO_WEIGHTS).exists():
        try:
            from ultralytics import YOLO
            yolo_model = YOLO(YOLO_WEIGHTS)
            yolo_model.to("cpu")
            print(f"✅ YOLOv8 loaded from {YOLO_WEIGHTS}")
        except Exception as e:
            print(f"⚠️  YOLOv8 load failed: {e}")
    else:
        print(f"⚠️  {YOLO_WEIGHTS} not found — detection disabled")

    # ResNet18 classifier
    if Path(CLF_WEIGHTS).exists():
        try:
            m = models.resnet18(weights=None)
            m.fc = nn.Sequential(
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 3)
            )
            m.load_state_dict(torch.load(CLF_WEIGHTS, map_location=DEVICE))
            m.eval().to(DEVICE)
            clf_model = m
            print(f"✅ ResNet18 classifier loaded from {CLF_WEIGHTS}")
        except Exception as e:
            print(f"⚠️  Classifier load failed: {e}")
    else:
        print(f"⚠️  {CLF_WEIGHTS} not found — classification disabled")

load_models()

# ── Transforms ───────────────────────────────────────────────────────────────
clf_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── Helpers ───────────────────────────────────────────────────────────────────
def classify_image(img_rgb: Image.Image):
    if clf_model is None:
        return None, None
    t = clf_tf(img_rgb).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = clf_model(t)
        probs  = torch.softmax(logits, dim=1)[0]
    idx  = probs.argmax().item()
    conf = float(probs[idx])
    return CLASS_NAMES[idx], conf


def detect_objects(img_path: str):
    if yolo_model is None:
        return []
    try:
        results = yolo_model(img_path, conf=CONF_THRESHOLD, verbose=False)[0]
        detections = []
        iw, ih = results.orig_shape[1], results.orig_shape[0]
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf  = float(box.conf[0])
            cls   = int(box.cls[0])
            detections.append({
                "class": CLASS_NAMES[cls],
                "conf":  round(conf, 3),
                "x1": round(x1), "y1": round(y1),
                "x2": round(x2), "y2": round(y2),
                "x_pct": round(x1/iw*100, 1),
                "y_pct": round(y1/ih*100, 1),
                "w_pct": round((x2-x1)/iw*100, 1),
                "h_pct": round((y2-y1)/ih*100, 1),
            })
        return detections
    except Exception as e:
        print(f"Detection error: {e}")
        return []


def draw_results(img_rgb, detections, clf_label, clf_conf):
    img     = img_rgb.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)
    iw, ih  = img.size

    for d in detections:
        x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
        cls   = d["class"]
        conf  = d["conf"]
        color = BOX_COLORS.get(cls, (200, 200, 200, 200))
        bg    = LABEL_BG.get(cls, (150, 150, 150))
        lw    = max(2, iw // 200)

        for i in range(lw):
            draw.rectangle([x1+i, y1+i, x2-i, y2-i], outline=color)

        tick = max(10, iw // 40)
        tc   = color[:3] + (255,)
        for (ax, ay, dx, dy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            draw.line([(ax, ay), (ax+dx*tick, ay)], fill=tc, width=lw+1)
            draw.line([(ax, ay), (ax, ay+dy*tick)], fill=tc, width=lw+1)

        label = f"{cls}  {conf*100:.0f}%"
        fs    = max(12, iw // 55)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fs)
        except:
            font = ImageFont.load_default()
        tw  = draw.textlength(label, font=font)
        pad = 5
        lx  = x1
        ly  = y1 - fs - pad*2 - 2 if y1 > fs + pad*3 else y2 + 2
        draw.rounded_rectangle([lx, ly, lx+tw+pad*2, ly+fs+pad*2], radius=4, fill=bg+(230,))
        draw.text((lx+pad, ly+pad), label, fill=(255,255,255), font=font)

    img = Image.alpha_composite(img, overlay).convert("RGB")

    if clf_label:
        draw2 = ImageDraw.Draw(img)
        stamp = f"  {clf_label.upper()}  {clf_conf*100:.0f}%  "
        sfs   = max(13, iw // 45)
        try:
            sfont = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", sfs)
        except:
            sfont = ImageFont.load_default()
        sw  = draw2.textlength(stamp, font=sfont)
        bg2 = LABEL_BG.get(clf_label, (80, 80, 80))
        draw2.rounded_rectangle([8, 8, 8+sw+12, 8+sfs+12], radius=5, fill=bg2+(230,))
        draw2.text((14, 14), stamp, fill=(255,255,255), font=sfont)

    return img


# ── Routes ────────────────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path("static/index.html")
    if html_path.exists():
        return HTMLResponse(html_path.read_text())
    return HTMLResponse("<h1>GPR Analyser</h1><p>static/index.html not found</p>")


@app.post("/analyse")
async def analyse(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")

    tmp_path = "/tmp/gpr_upload.jpg"
    img.save(tmp_path, "JPEG", quality=95)

    clf_label, clf_conf = classify_image(img)
    detections          = detect_objects(tmp_path)

    cavity_count  = sum(1 for d in detections if d["class"] == "cavity")
    utility_count = sum(1 for d in detections if d["class"] == "utility")
    if cavity_count >= 2:
        risk = "high"
    elif cavity_count == 1 or utility_count >= 2:
        risk = "medium"
    else:
        risk = "low"

    annotated = draw_results(img, detections, clf_label, clf_conf or 0.0)
    buf = io.BytesIO()
    annotated.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    return JSONResponse({
        "overall_class":   clf_label or "unknown",
        "confidence":      round(clf_conf or 0.0, 3),
        "risk_level":      risk,
        "detections":      detections,
        "annotated_image": f"data:image/png;base64,{img_b64}",
        "models_loaded": {
            "classifier": clf_model  is not None,
            "detector":   yolo_model is not None,
        }
    })


@app.get("/status")
async def status():
    return {
        "classifier": clf_model  is not None,
        "detector":   yolo_model is not None,
        "device":     str(DEVICE),
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
