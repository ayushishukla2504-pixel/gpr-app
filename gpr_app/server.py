"""
GPR B-Scan Analyser — Backend Server (Classifier + YOLOv8)
"""
import subprocess
subprocess.run(["pip", "install", "gdown", "-q"], capture_output=True)
import gdown
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
os.environ["QT_QPA_PLATFORM"]          = "offscreen"
os.environ["DISPLAY"]                  = ":99"
os.environ["MPLBACKEND"]               = "Agg"

if not os.path.exists("best.pt") or os.path.getsize("best.pt") < 1000000:
    print("Downloading best.pt...")
    gdown.download(id="1q-BV-7_JvOfiyol4Aa8udG9A9kJTF5hO", output="best.pt", quiet=False)

if not os.path.exists("classifier_best.pth") or os.path.getsize("classifier_best.pth") < 1000000:
    print("Downloading classifier_best.pth...")
    gdown.download(id="15xEBkXlSOdh6QmY7QMk90YQ1wjFtodbp", output="classifier_best.pth", quiet=False)

import io
import base64
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
from torchvision import transforms, models

app = FastAPI(title="GPR B-Scan Analyser")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

CLASS_NAMES    = ["cavity", "intact", "utility"]
DEVICE         = torch.device("cpu")
CLF_WEIGHTS    = "classifier_best.pth"
YOLO_WEIGHTS   = "best.pt"
CONF_THRESHOLD = 0.25

LABEL_BG = {
    "cavity":  (226, 75,  74),
    "utility": (55,  138, 221),
    "intact":  (99,  153, 34),
}
BOX_COLORS = {
    "cavity":  (226, 75,  74,  220),
    "utility": (55,  138, 221, 220),
    "intact":  (99,  153, 34,  220),
}

clf_model  = None
yolo_model = None

def load_models():
    global clf_model, yolo_model

    # ResNet18 classifier
    if Path(CLF_WEIGHTS).exists() and os.path.getsize(CLF_WEIGHTS) > 1000000:
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
            m.load_state_dict(torch.load(CLF_WEIGHTS, map_location=DEVICE, weights_only=False))
            m.eval().to(DEVICE)
            clf_model = m
            print("✅ ResNet18 classifier loaded")
        except Exception as e:
            print(f"⚠️  Classifier load failed: {e}")
    else:
        size = os.path.getsize(CLF_WEIGHTS) if Path(CLF_WEIGHTS).exists() else 0
        print(f"⚠️  classifier_best.pth not ready — size: {size}")

    # YOLOv8 detector
    if Path(YOLO_WEIGHTS).exists() and os.path.getsize(YOLO_WEIGHTS) > 1000000:
        try:
            from ultralytics import YOLO
            yolo_model = YOLO(YOLO_WEIGHTS)
            yolo_model.to("cpu")
            print("✅ YOLOv8 detector loaded")
        except Exception as e:
            print(f"⚠️  YOLOv8 load failed: {e}")
    else:
        size = os.path.getsize(YOLO_WEIGHTS) if Path(YOLO_WEIGHTS).exists() else 0
        print(f"⚠️  best.pt not ready — size: {size}")

load_models()

clf_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def classify_image(img_rgb):
    if clf_model is None:
        return None, None, [0.0, 0.0, 0.0]
    t = clf_tf(img_rgb).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(clf_model(t), dim=1)[0]
    idx = probs.argmax().item()
    return CLASS_NAMES[idx], float(probs[idx]), probs.tolist()

def detect_objects(img_path):
    if yolo_model is None:
        return []
    try:
        results = yolo_model(img_path, conf=CONF_THRESHOLD, verbose=False)[0]
        detections = []
        iw, ih = results.orig_shape[1], results.orig_shape[0]
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls  = int(box.cls[0])
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

def draw_results(img_rgb, clf_label, clf_conf, all_probs, detections):
    img     = img_rgb.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)
    iw, ih  = img.size

    # Draw bounding boxes
    for d in detections:
        x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
        cls   = d["class"]
        conf  = d["conf"]
        color = BOX_COLORS.get(cls, (200, 200, 200, 220))
        bg    = LABEL_BG.get(cls, (150, 150, 150))
        lw    = max(2, iw // 200)

        for i in range(lw):
            draw.rectangle([x1+i, y1+i, x2-i, y2-i], outline=color)

        tick = max(10, iw // 40)
        tc   = color[:3] + (255,)
        for (ax, ay, dx, dy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            draw.line([(ax, ay), (ax+dx*tick, ay)], fill=tc, width=lw+1)
            draw.line([(ax, ay), (ax, ay+dy*tick)], fill=tc, width=lw+1)

        fs = max(12, iw // 55)
        try:
            font = ImageFont.truetype("/run/current-system/sw/share/X11/fonts/TTF/DejaVuSans-Bold.ttf", fs)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fs)
            except:
                font = ImageFont.load_default()

        label = f"{cls}  {conf*100:.0f}%"
        tw    = draw.textlength(label, font=font)
        pad   = 5
        lx    = x1
        ly    = y1 - fs - pad*2 - 2 if y1 > fs + pad*3 else y2 + 2
        draw.rounded_rectangle([lx, ly, lx+tw+pad*2, ly+fs+pad*2], radius=4, fill=bg+(230,))
        draw.text((lx+pad, ly+pad), label, fill=(255,255,255), font=font)

    img = Image.alpha_composite(img, overlay).convert("RGB")
    draw2 = ImageDraw.Draw(img)

    # Overall classification stamp
    fs2 = max(14, iw // 40)
    try:
        font2  = ImageFont.truetype("/run/current-system/sw/share/X11/fonts/TTF/DejaVuSans-Bold.ttf", fs2)
        sfont2 = ImageFont.truetype("/run/current-system/sw/share/X11/fonts/TTF/DejaVuSans.ttf", max(11, fs2-4))
    except:
        try:
            font2  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fs2)
            sfont2 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", max(11, fs2-4))
        except:
            font2  = ImageFont.load_default()
            sfont2 = font2

    bg2   = LABEL_BG.get(clf_label, (80, 80, 80))
    stamp = f"  {clf_label.upper()}  {clf_conf*100:.0f}%  "
    sw    = draw2.textlength(stamp, font=font2)
    pad2  = 8
    draw2.rounded_rectangle([8, 8, 8+sw+pad2, 8+fs2+pad2+4], radius=6, fill=bg2)
    draw2.text((8+pad2//2, 10), stamp, fill=(255,255,255), font=font2)

    # Probability bars
    bar_x = 12
    bar_y = 8 + fs2 + pad2 + 14
    bar_w = max(140, iw // 4)
    bar_h = max(14, fs2 - 2)
    gap   = bar_h + 6

    for i, (cls, prob) in enumerate(zip(CLASS_NAMES, all_probs)):
        color = LABEL_BG.get(cls, (120, 120, 120))
        y = bar_y + i * gap
        draw2.rounded_rectangle([bar_x, y, bar_x+bar_w, y+bar_h], radius=3, fill=(50,50,50))
        fill_w = int(bar_w * prob)
        if fill_w > 4:
            draw2.rounded_rectangle([bar_x, y, bar_x+fill_w, y+bar_h], radius=3, fill=color)
        draw2.text((bar_x+6, y+1), f"{cls}  {prob*100:.0f}%", fill=(255,255,255), font=sfont2)

    return img

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def index():
    p = Path("static/index.html")
    return HTMLResponse(p.read_text() if p.exists() else "<h1>GPR Analyser</h1>")

@app.post("/analyse")
async def analyse(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")

    tmp_path = "/tmp/gpr_upload.jpg"
    img.save(tmp_path, "JPEG", quality=95)

    clf_label, clf_conf, all_probs = classify_image(img)
    detections = detect_objects(tmp_path)

    cavity_count  = sum(1 for d in detections if d["class"] == "cavity")
    utility_count = sum(1 for d in detections if d["class"] == "utility")
    if cavity_count >= 2:
        risk = "high"
    elif cavity_count == 1 or utility_count >= 2:
        risk = "medium"
    else:
        risk = "low"

    annotated = draw_results(img, clf_label or "unknown", clf_conf or 0.0, all_probs, detections)
    buf = io.BytesIO()
    annotated.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    return JSONResponse({
        "overall_class":   clf_label or "unknown",
        "confidence":      round(clf_conf or 0.0, 3),
        "risk_level":      risk,
        "detections":      detections,
        "annotated_image": f"data:image/png;base64,{img_b64}",
        "models_loaded":   {
            "classifier": clf_model  is not None,
            "detector":   yolo_model is not None,
        },
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