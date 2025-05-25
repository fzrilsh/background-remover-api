# Background Remover API

A Flask-based API service for removing the background of images using OpenCV with multiple background removal techniques: GrabCut, Contour-based, and Advanced segmentation.

## ✨ Features

- 🔁 Supports three background removal methods:
  - GrabCut algorithm
  - Contour and edge-based segmentation
  - Advanced combination of color, edge, and contour segmentation
- 🖼️ Returns the image in RGBA format with transparent background
- 🌐 Simple REST API using Flask
- 🔄 CORS-enabled for frontend access

---

## 📦 Installation

1. Clone the repository:

```bash
git clone https://github.com/fzrilsh/background-remover-api.git
cd background-remover-api