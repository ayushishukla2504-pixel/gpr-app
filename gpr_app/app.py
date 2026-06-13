import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
os.environ["QT_QPA_PLATFORM"]          = "offscreen"
os.environ["DISPLAY"]                  = ":99"
os.environ["MPLBACKEND"]               = "Agg"

import io
import base64
import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import numpy as np

CLASS_NAMES = ["cavity", "intact", "utility"]
DEVICE      = torch.device("cpu")

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

# Load classifier
clf_model = None
try:
    m = models.resnet18(weights=None)
    m.fc = nn.Sequential(
        nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 3)
    )
    m.load_state_dict(torch.load("classifier_best.pth", map_location=DEVICE, weights_only=False))
    m.eval()
    clf_model = m
    print("✅ Classifier loaded")
except Exception as e:
    print(f"⚠️ Classifier failed: {e}")

# Load YOLOv8
yolo_model = None
try:
    from ultralytics import YOLO
    yolo_model = YOLO("best.pt")
    yolo_model.to("cpu")
    print("✅ YOLOv8 loaded")
except Exception as e:
    print(f"⚠️ YOLOv8 failed: {e}")

clf_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def analyse(image):
    img = image.convert("RGB")

    # Classify
    clf_label, clf_conf, all_probs = "unknown", 0.0, [0.0, 0.0, 0.0]
    if clf_model:
        t = clf_tf(img).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(clf_model(t), dim=1)[0]
        idx = probs.argmax().item()
        clf_label, clf_conf, all_probs = CLASS_NAMES[idx], float(probs[idx]), probs.tolist()

    # Detect
    detections = []
    if yolo_model:
        try:
            tmp = "/tmp/input.jpg"
            img.save(tmp)
            results = yolo_model(tmp, conf=0.25, verbose=False)[0]
            iw, ih = results.orig_shape[1], results.orig_shape[0]
            for box in results.boxes:
                x1,y1,x2,y2 = box.xyxy[0].tolist()
                detections.append({
                    "class": CLASS_NAMES[int(box.cls[0])],
                    "conf": float(box.conf[0]),
                    "x1": round(x1), "y1": round(y1),
                    "x2": round(x2), "y2": round(y2),
                })
        except Exception as e:
            print(f"Detection error: {e}")

    # Draw
    out     = img.copy().convert("RGBA")
    overlay = Image.new("RGBA", out.size, (0,0,0,0))
    draw    = ImageDraw.Draw(overlay)
    iw, ih  = out.size

    for d in detections:
        x1,y1,x2,y2 = d["x1"],d["y1"],d["x2"],d["y2"]
        color = BOX_COLORS.get(d["class"], (200,200,200,220))
        bg    = LABEL_BG.get(d["class"], (150,150,150))
        lw    = max(2, iw//200)
        for i in range(lw):
            draw.rectangle([x1+i,y1+i,x2-i,y2-i], outline=color)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", max(12,iw//55))
        except:
            font = ImageFont.load_default()
        label = f"{d['class']} {d['conf']*100:.0f}%"
        tw = draw.textlength(label, font=font)
        ly = y1 - max(12,iw//55) - 12 if y1 > 30 else y2 + 2
        draw.rounded_rectangle([x1, ly, x1+tw+10, ly+max(12,iw//55)+8], radius=4, fill=bg+(230,))
        draw.text((x1+5, ly+4), label, fill=(255,255,255), font=font)

    out = Image.alpha_composite(out, overlay).convert("RGB")
    draw2 = ImageDraw.Draw(out)

    # Stamp
    bg2 = LABEL_BG.get(clf_label, (80,80,80))
    fs  = max(14, iw//40)
    try:
        f2 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fs)
        f3 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", max(11,fs-4))
    except:
        f2 = f3 = ImageFont.load_default()

    stamp = f"  {clf_label.upper()}  {clf_conf*100:.0f}%  "
    sw = draw2.textlength(stamp, font=f2)
    draw2.rounded_rectangle([8,8,8+sw+10,8+fs+12], radius=6, fill=bg2)
    draw2.text((13,10), stamp, fill=(255,255,255), font=f2)

    bar_x, bar_y = 12, 8+fs+20
    bar_w, bar_h = max(140,iw//4), max(14,fs-2)
    for i,(cls,prob) in enumerate(zip(CLASS_NAMES, all_probs)):
        color = LABEL_BG.get(cls,(120,120,120))
        y = bar_y + i*(bar_h+6)
        draw2.rounded_rectangle([bar_x,y,bar_x+bar_w,y+bar_h], radius=3, fill=(50,50,50))
        fw = int(bar_w*prob)
        if fw > 4:
            draw2.rounded_rectangle([bar_x,y,bar_x+fw,y+bar_h], radius=3, fill=color)
        draw2.text((bar_x+6,y+1), f"{cls}  {prob*100:.0f}%", fill=(255,255,255), font=f3)

    risk = "HIGH" if clf_label=="cavity" else ("MEDIUM" if clf_label=="utility" else "LOW")
    det_text = "\n".join([f"#{i+1} {d['class']} — {d['conf']*100:.0f}% at ({d['x1']},{d['y1']})" for i,d in enumerate(detections)]) or "No anomalies detected"
    summary = f"Class: {clf_label.upper()} | Confidence: {clf_conf*100:.0f}% | Risk: {risk} | Detections: {len(detections)}"

    return out, summary, det_text

demo = gr.Interface(
    fn=analyse,
    inputs=gr.Image(type="pil", label="Upload GPR B-Scan"),
    outputs=[
        gr.Image(type="pil", label="Annotated Scan"),
        gr.Textbox(label="Classification Result"),
        gr.Textbox(label="Detected Anomalies"),
    ],
    title="GPR B-Scan Analyser",
    description="Upload a GPR B-scan image to detect and classify subsurface anomalies (cavities, utilities, intact zones).",
    theme=gr.themes.Soft()
)

demo.launch()