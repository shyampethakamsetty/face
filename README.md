# Real-Time Face-Based Age & Gender Detection Matrimony Platform

## Overview
This project is a full-stack web application for real-time face-based age and gender detection, designed for a matrimony platform to prevent fraud. It uses state-of-the-art AI models to estimate age and gender from a user's webcam image.

---

## Folder Structure
```
/
  frontend/   # React app (webcam, UI)
  backend/    # Flask app (API, InsightFace)
  README.md   # This file
```

---

## Backend (Flask + InsightFace)

### Setup
1. Navigate to the backend folder:
   ```sh
   cd backend
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Download the InsightFace model (buffalo_l) if not auto-downloaded on first run.

### Run
```sh
python app.py
```
The backend will start on `http://localhost:5000` by default.

---

## Frontend (React)

### Setup
1. Navigate to the frontend folder:
   ```sh
   cd frontend
   ```
2. Install dependencies:
   ```sh
   npm install
   ```

### Run
```sh
npm start
```
The frontend will start on `http://localhost:3000` by default.

---

## Usage
1. Open the frontend in your browser.
2. Allow webcam access.
3. Click "Detect Age & Gender" to capture your image and get results.

---

## Notes
- Ensure both frontend and backend are running.
- The backend uses InsightFace (buffalo_l) for robust detection, suitable for Indian faces.
- CORS is enabled for local development. 