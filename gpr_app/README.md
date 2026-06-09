# GPR B-Scan Analyser

A standalone web app that runs your trained YOLOv8 + ResNet18 models locally
and shows detection boxes + classification directly on your GPR scan image.

---

## Setup (one time)

### 1. Install dependencies
```bash
pip install fastapi uvicorn python-multipart torch torchvision ultralytics pillow numpy
```

### 2. Place your trained weights in this folder
```
gpr_app/
├── server.py
├── best.pt                  ← your YOLOv8 detection weights
├── classifier_best.pth      ← your ResNet18 classification weights
├── static/
│   └── index.html
└── README.md
```

`best.pt` comes from:
  `runs/detect/gpr_detector/weights/best.pt`

`classifier_best.pth` comes from:
  `models/classifier_best.pth`

---

## Run

```bash
cd gpr_app
python server.py
```

Then open your browser at:

```
http://localhost:8000
```

That's it. The app opens, you upload any GPR B-scan image, and it will:
- Classify the overall scan (cavity / intact / utility)
- Draw bounding boxes on every detected anomaly
- Show confidence scores and risk level
- Give a plain-English interpretation

---

## What the colours mean

| Colour | Class   | Meaning                              |
|--------|---------|--------------------------------------|
| Red    | Cavity  | Hyperbolic arch = subsurface void    |
| Blue   | Utility | Pipe or cable reflection             |
| Green  | Intact  | No significant anomaly               |

---

## Notes

- Works on CPU if no GPU available (slightly slower)
- Confidence threshold default: 0.25 (edit `CONF_THRESHOLD` in server.py)
- The app auto-checks server status every 15 seconds
- To use on another device on your network, replace `localhost` with your machine's IP
