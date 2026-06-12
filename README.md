# 🧠 Real-Time Head Counter & People Tracker

> Detects and tracks unique people in CCTV / crowd footage using YOLOv5 + CrowdHuman weights + Centroid Tracker — built entirely on CPU.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU-red?logo=pytorch)
![YOLOv5](https://img.shields.io/badge/Model-YOLOv5m-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 What It Does

| Feature | Description |
|---|---|
| 🟢 Head Detection | Detects every head in each frame using CrowdHuman-trained YOLOv5m |
| 🔵 Unique Person Tracking | Assigns each person a stable ID — same person across 200 frames = counted once |
| 📊 Cumulative Count | Shows total unique people seen from frame 0 to current frame |
| 👁️ In-Frame Count | Shows how many people are visible right now |
| 💾 Works on CPU | No GPU needed — runs on Intel/AMD CPU |

---

## 🎬 Demo

> *(Add your output video or GIF here — see instructions below)*

<!-- Replace this comment with your GIF:
![Demo](assets/demo.gif)
-->

---

## 🗂️ Project Structure

```
yolov5-crowdhuman/
│
├── detect.py                  ← main script (detection + tracker + overlay)
├── weights/
│   └── crowdhuman_yolov5m.pt  ← pretrained model (download separately)
├── models/                    ← YOLOv5 model architecture
├── utils/                     ← YOLOv5 utilities
├── test/
│   └── your_video.mp4         ← your input CCTV footage
└── runs/detect/               ← output saved here automatically
```

---

## ⚙️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/Naazpathan111/headcount-tracker.git
cd headcount-tracker
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

### 3. Install PyTorch (CPU)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install requirements
```bash
pip install -r requirements.txt
pip install scipy
```

### 5. Download model weights
Download `crowdhuman_yolov5m.pt` from the [releases page](https://github.com/Naazpathan111/headcount-tracker/releases) and place it in `weights/`.

---

## 🚀 Run

```bash
# On a video file
python detect.py --weights weights/crowdhuman_yolov5m.pt --source test/your_video.mp4 --heads --view-img

# On webcam
python detect.py --weights weights/crowdhuman_yolov5m.pt --source 0 --heads --view-img
```

---

## 📺 What You See on Screen

```
┌─────────────────────────────────────────────┐
│  TOTAL UNIQUE (all frames):  23             │  ← grows from frame 0, never drops
│  In frame right now:          4             │  ← current frame only
└─────────────────────────────────────────────┘
```

- 🟢 **Green box** = detected head
- 🟠 **Orange dot + #ID** = tracked person with unique ID
- Counter updates every frame

---

## 🧠 How the Tracker Works

```
Frame N:   detect heads → get bounding boxes
               ↓
           compute centroids (center of each box)
               ↓
           match centroids to known IDs using distance
           (if distance > 150px → new person, new ID)
               ↓
           add ID to seen_total set (set ignores duplicates)
               ↓
           len(seen_total) = true unique count
```

The key insight: a Python `set` automatically ignores duplicates, so the same person appearing in 200 frames still only counts as 1.

---

## 🛠️ Built With

- [YOLOv5](https://github.com/ultralytics/yolov5) — object detection
- [CrowdHuman Dataset](https://www.crowdhuman.org/) — pretrained weights for crowd scenes
- [PyTorch](https://pytorch.org/) — deep learning framework
- [OpenCV](https://opencv.org/) — video processing & overlay
- [SciPy](https://scipy.org/) — centroid distance matching

---

## 📸 How to Add a Demo GIF to GitHub

1. Run detection on your video
2. Find output in `runs/detect/exp/`
3. Convert to GIF using [ezgif.com](https://ezgif.com/video-to-gif)
4. Create an `assets/` folder in your repo
5. Upload `demo.gif` there
6. Uncomment the `![Demo]` line in this README

---

## 👤 Author

**Naazpathan111**  
[GitHub](https://github.com/Naazpathan111)

---

## 📄 License

MIT License — free to use and modify.
