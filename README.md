# 🦺 PPE Detection Dashboard

A **professional-grade AI-powered web application** for monitoring Personal Protective Equipment (PPE) compliance using **YOLOv8** and **Streamlit**. This system detects safety helmets, vests, and violations in uploaded videos and provides a modern, enterprise-style UI with real-time terminal logs.

---

## 🚀 Features

* 🎯 **YOLOv8-based Detection** – Accurate object detection for PPE classes
* 🖥️ **Professional Dashboard UI** – Clean, modern, SaaS-style interface
* 📹 **Video Upload & Processing** – Supports MP4, AVI, MOV
* 🔁 **Real-time Frame Processing** – Frame-by-frame inference
* 📊 **Detection Summary** – Auto-generated counts per class
* 🧾 **Live Terminal Logs** – See model inference status in real-time
* 💾 **Download Processed Video** – Browser-safe MP4 output (H.264)
* ⚙️ **Production-Safe Video Pipeline** – FFmpeg-based encoding

---

## 🛠️ Tech Stack

| Component        | Technology                  |
| ---------------- | --------------------------- |
| Frontend         | Streamlit                   |
| Backend          | Python                      |
| Model            | YOLOv8 (Ultralytics)        |
| Video Processing | OpenCV                      |
| Encoding         | FFmpeg (via imageio-ffmpeg) |
| UI Styling       | Custom CSS                  |

---

## 📂 Project Structure

```
PPE-Detection-Dashboard/
│
├── app.py                 # Main Streamlit app
├── model/
│   └── best.pt            # Trained YOLOv8 model
├── uploads/               # Uploaded input videos
├── outputs/               # Processed output videos
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## 🔧 Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/ppe-detection-dashboard.git
cd ppe-detection-dashboard
```

### 2️⃣ Create a Virtual Environment (Recommended)

```bash
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # Mac/Linux
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing, install manually:

```bash
pip install ultralytics streamlit opencv-python imageio-ffmpeg
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

The dashboard will open in your browser.

---

## 🧪 How It Works

1. Upload a video (MP4, AVI, MOV)
2. Click **Run PPE Detection**
3. The model processes each frame using YOLOv8
4. Bounding boxes and labels are drawn
5. Output video is re-encoded to browser-safe MP4
6. Processed video is displayed + downloadable
7. Terminal shows real-time detection logs

---

## 📊 Example Terminal Output

```
[FRAME 12] {'person': 3, 'helmet': 2, 'vest': 2}
[FRAME 13] {'person': 3, 'helmet': 1, 'vest': 2, 'no-helmet': 1}
...
```

---

## 🎨 UI Design Philosophy

This dashboard is designed like a **real SaaS product**, not a student project:

* Glassmorphism layout
* Gradient hero section
* Card-based components
* Clean spacing & typography
* Dark enterprise theme
* KPI-style metrics

---

## 🔐 Use Cases

* Industrial safety monitoring
* Construction site surveillance
* Factory compliance tracking
* Smart CCTV systems
* Research & academic demos

---

## 📌 Future Enhancements

* 🔴 Violation alerts (No helmet, No vest)
* 📈 Compliance percentage
* ⏱️ Timestamped reports
* ☁️ Cloud deployment
* 🔐 Authentication & roles
* 📄 PDF / CSV report export

---

## 📜 License

This project is for educational and research purposes.

---

## 🤝 Acknowledgements

* [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
* [Streamlit](https://streamlit.io/)
* OpenCV
* FFmpeg

---

## 👨‍💻 Author

Developed as a **professional AI safety monitoring system**.

If you need help, deployment, or feature upgrades — feel free to ask!

---

⭐ *If you like this project, give it a star!*
