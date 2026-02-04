# 🌿 Ne3Na3 - Senior Medical AI Brain Tumor Segmentation

<div align="center">

![Ne3Na3 Logo](frontend/public/ne3na3-icon.svg)

**Multi-Modal BraTS Segmentation with AttUnet**

[![Python](https://img.shields.io/badge/Python-3.9+-00A676?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-00A676?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18.2-00A676?style=for-the-badge&logo=react&logoColor=white)](https://react.dev)
[![MONAI](https://img.shields.io/badge/MONAI-1.3+-00A676?style=for-the-badge)](https://monai.io)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Reference](#-api-reference)
- [Safety & Ethics](#-safety--ethics)

---

## 🧠 Overview

**Ne3Na3** is a cutting-edge medical AI system for multi-modal brain tumor segmentation using the BraTS dataset format. It processes four MRI modalities (T1, T1ce, T2, FLAIR) and outputs multi-class segmentation masks for:

- **NCR** - Necrotic Core
- **ED** - Peritumoral Edema  
- **ET** - Enhancing Tumor

### Tumor Regions

| Region | Description | Labels |
|--------|-------------|--------|
| **WT** (Whole Tumor) | Complete tumor extent | NCR + ED + ET |
| **TC** (Tumor Core) | Solid tumor mass | NCR + ET |
| **ET** (Enhancing Tumor) | Active tumor | ET only |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Ne3Na3 SYSTEM ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐  │
│  │   React Frontend │◄────►│  FastAPI Backend │◄────►│  AttUnet Model  │  │
│  │   (Port 3000)    │ HTTP │   (Port 8000)    │      │  (PyTorch)      │  │
│  └────────┬────────┘      └────────┬────────┘      └────────┬────────┘  │
│           │                        │                        │           │
│  ┌────────▼────────┐      ┌────────▼────────┐      ┌────────▼────────┐  │
│  │ • File Upload   │      │ • /api/segment   │      │ • 3D Conv Blocks │  │
│  │ • Insights Panel│      │ • /api/insights  │      │ • Attention Gates│  │
│  │ • Safe-Bot Chat │      │ • /api/chat      │      │ • Sliding Window │  │
│  │ • Explainability│      │ • /api/attention │      │ • TTA Processing │  │
│  └─────────────────┘      └─────────────────┘      └─────────────────┘  │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                         INFERENCE PIPELINE                         │   │
│  │                                                                    │   │
│  │   Input (4 NIfTI) ──► Normalize ──► Sliding Window (96³) ──►      │   │
│  │   TTA (Flips) ──► AttUnet ──► Anatomical Consistency ──► Output   │   │
│  │                                                                    │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### AttUnet Architecture

```
Input (4, D, H, W)
    │
    ▼
┌───────────┐
│ Encoder 1 │──────────────────────────────┐
│ 4 → 32    │                              │
└─────┬─────┘                              │ Attention Gate
      │ MaxPool                            │
┌─────▼─────┐                              │
│ Encoder 2 │─────────────────────┐        │
│ 32 → 64   │                     │        │
└─────┬─────┘                     │ AG     │
      │ MaxPool                   │        │
┌─────▼─────┐                     │        │
│ Encoder 3 │────────────┐        │        │
│ 64 → 128  │            │ AG     │        │
└─────┬─────┘            │        │        │
      │ MaxPool          │        │        │
┌─────▼─────┐            │        │        │
│ Encoder 4 │───┐        │        │        │
│ 128 → 256 │   │ AG     │        │        │
└─────┬─────┘   │        │        │        │
      │ MaxPool │        │        │        │
┌─────▼─────┐   │        │        │        │
│ Bottleneck│   │        │        │        │
│ 256 → 512 │   │        │        │        │
└─────┬─────┘   │        │        │        │
      │ Upsample│        │        │        │
┌─────▼─────┐   │        │        │        │
│ Decoder 4 │◄──┘        │        │        │
│ 512 → 256 │            │        │        │
└─────┬─────┘            │        │        │
      │ Upsample         │        │        │
┌─────▼─────┐            │        │        │
│ Decoder 3 │◄───────────┘        │        │
│ 256 → 128 │                     │        │
└─────┬─────┘                     │        │
      │ Upsample                  │        │
┌─────▼─────┐                     │        │
│ Decoder 2 │◄────────────────────┘        │
│ 128 → 64  │                              │
└─────┬─────┘                              │
      │ Upsample                           │
┌─────▼─────┐                              │
│ Decoder 1 │◄─────────────────────────────┘
│ 64 → 32   │
└─────┬─────┘
      │
┌─────▼─────┐
│ Output    │
│ 32 → 4    │
└───────────┘
    │
    ▼
Output (4, D, H, W) → Softmax → Segmentation
```

---

## ✨ Features

### 🧠 Ne3Na3 Segmentation Engine
- **MONAI Sliding Window Inference** (ROI: 96×96×96, overlap: 0.5)
- **Test-Time Augmentation** with axis flips
- **Anatomical Consistency** enforcement (TC ⊂ WT, ET ⊂ TC)

### 📊 Insight Engine
- Tumor volume in mm³/cm³
- Bounding box dimensions
- Asymmetry scores (left vs right hemisphere)
- Surface area estimation

### 🔬 Explainability Sidebar
- Attention map visualization from AttentionBlocks
- Modality importance statistics
- Decoder layer attention breakdown

### 💬 Ne3Na3 Safe-Bot
- Grounded in analysis JSON only
- **Safety-first design**: Refuses diagnosis/treatment advice
- Calm, green-themed UI cues

---

## 🚀 Installation

### Prerequisites

- Python 3.9+
- Node.js 18+
- CUDA (optional, for GPU acceleration)

### Backend Setup

```bash
# Navigate to backend
cd ne3na3/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Add model weights (optional)
# Place your .pth file in model_weights/attunet_brats.pth
```

### Frontend Setup

```bash
# Navigate to frontend
cd ne3na3/frontend

# Install dependencies
npm install
```

---

## 📖 Usage

### Start the Backend

```bash
cd ne3na3/backend

# Development mode
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Start the Frontend

```bash
cd ne3na3/frontend

# Development mode
npm run dev

# Build for production
npm run build
npm run preview
```

### Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

---

## 📡 API Reference

### Health Check
```http
GET /health
```

### Run Segmentation
```http
POST /api/segment
Content-Type: multipart/form-data

- t1: NIfTI file
- t1ce: NIfTI file
- t2: NIfTI file
- flair: NIfTI file
- use_tta: boolean (default: true)
- enforce_consistency: boolean (default: true)
```

### Get Insights
```http
GET /api/insights
```

### Chat with Safe-Bot
```http
POST /api/chat
Content-Type: application/json

{
  "message": "What are the tumor volumes?"
}
```

### Run Demo
```http
POST /api/demo
```

---

## 🛡 Safety & Ethics

### ⚠️ Important Disclaimers

1. **Research Use Only**: Ne3Na3 is designed for educational and research purposes only. It is NOT a medical device and should NOT be used for clinical diagnosis.

2. **No Medical Advice**: The system explicitly refuses to provide:
   - Medical diagnoses
   - Treatment recommendations
   - Prognosis predictions
   - Drug/medication advice

3. **Healthcare Professional Required**: All results should be reviewed and interpreted by qualified healthcare professionals.

4. **Data Privacy**: 
   - MRI data is processed locally and not stored permanently
   - No PHI is transmitted to external servers
   - Compliant with research ethics guidelines

### Safe-Bot System Prompt

```
You are Ne3Na3 Safe-Bot, a helpful medical imaging assistant.

🛡️ SAFETY RULES (NON-NEGOTIABLE):
1. NEVER provide medical diagnoses
2. NEVER suggest treatments or medications
3. NEVER predict patient outcomes
4. ALWAYS recommend consulting healthcare professionals
5. ALWAYS clarify this is for research/educational purposes

💚 You CAN explain:
- Volume measurements
- Tumor region definitions
- MRI modality information
- Technical analysis results
```

---

## 🎨 Design System

### Color Palette

| Color | Hex | Usage |
|-------|-----|-------|
| **Healing Green** | `#00A676` | Primary actions, highlights |
| **Mint Frost** | `#E6F4F1` | Light backgrounds |
| **Deep Green** | `#004D40` | Dark accents |
| **Neon Mint** | `#00FFB3` | AI hotspots, glow effects |

### UI Principles

- **Glassmorphism**: Frosted glass effects with backdrop blur
- **Pill Shapes**: Rounded buttons and badges
- **Dark Mode**: High-contrast for clinical precision
- **Calm Aesthetics**: Soothing green tones for healthcare context

---

## 📁 Project Structure

```
ne3na3/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI application
│   │   ├── inference.py         # Inference engine
│   │   ├── insights.py          # Metrics computation
│   │   ├── chatbot.py           # Safe-Bot implementation
│   │   ├── schemas.py           # Pydantic models
│   │   └── models/
│   │       ├── __init__.py
│   │       ├── attunet.py       # AttUnet architecture
│   │       └── model_loader.py  # Weight loading logic
│   ├── model_weights/           # Place .pth files here
│   └── requirements.txt
│
├── frontend/
│   ├── public/
│   │   └── ne3na3-icon.svg
│   ├── src/
│   │   ├── main.jsx
│   │   ├── App.jsx              # Main application
│   │   ├── index.css            # Global styles
│   │   ├── components/
│   │   │   ├── FileUploadZone.jsx
│   │   │   ├── InsightsPanel.jsx
│   │   │   ├── SafeBot.jsx
│   │   │   ├── ProcessingOverlay.jsx
│   │   │   └── ExplainabilityPanel.jsx
│   │   └── services/
│   │       └── api.js           # API client
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── postcss.config.js
│
└── README.md
```

---

## 🙏 Acknowledgments

- [MONAI](https://monai.io/) - Medical Open Network for AI
- [BraTS Challenge](https://www.med.upenn.edu/cbica/brats/) - Brain Tumor Segmentation
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [React](https://react.dev/) - UI library

---

<div align="center">

**🌿 Ne3Na3** — *Fresh, Clinically Precise, Calming*

Made with 💚 for the Hackathon

</div>
