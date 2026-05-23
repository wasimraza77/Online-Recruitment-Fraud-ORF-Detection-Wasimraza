# 🛡️ Online Recruitment Fraud (ORF) Detection System

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-blue?style=for-the-badge&logo=render)](https://online-recruitment-fraud-orf-detection-a9kg.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.9+-green?style=for-the-badge&logo=python)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1.3-orange?style=for-the-badge&logo=flask)](https://flask.palletsprojects.com/)
[![PostgreSQL](https://img.shields.io/badge/Database-PostgreSQL%20%7C%20SQLite-blue?style=for-the-badge&logo=postgresql)](https://neon.tech/)

An advanced AI-powered web application designed to detect fraudulent job postings and protect job seekers from recruitment scams. Built as a **B.Tech Final Year Project**, this system implements a high-performance deep learning pipeline with instant classification and an integrated AI chatbot assistant.

🔗 **Live Deployment Link:** [https://online-recruitment-fraud-orf-detection-a9kg.onrender.com](https://online-recruitment-fraud-orf-detection-a9kg.onrender.com)

---

## 🌟 Key Features

*   **📊 Live Interactive Dashboard:** Real-time metrics on parsed jobs, flagged scam rates, and cluster insights.
*   **📂 Batch CSV Analysis:** Upload a CSV file containing multiple job postings to scan them all at once.
*   **✍️ Manual Job Check:** Paste or type a single job description to get a real-time risk assessment.
*   **🤖 AI Fraud Assistant Chatbot:** Embedded conversational scanner powered by **Google Gemini 2.5 Flash** (with fallback to an optimized local rule-based engine).
*   **📧 Welcoming System:** An asynchronous, non-blocking email service that sends HTML registration credentials upon user sign-up.
*   **🔒 Secure User Auth:** Robust sign-up and login mechanisms with encrypted password hashing.

---

## 🧠 ML/DL Architecture & Pipeline

The system uses a custom-engineered pipeline for optimal speed, security, and accuracy:
1.  **Text Preprocessing:** Automated stop-word filtering, punctuation removal, and tokenization.
2.  **Sentence Embeddings:** Text is mapped into dense vector spaces using `SentenceTransformers` (`distilbert-base-nli-stsb-mean-tokens`).
3.  **Inference Model:** A high-performance **2D Convolutional Neural Network (CNN)** running optimized numpy-based tensor operations (`cnn2d_weights.hdf5` & `bert_X.npy`).
4.  **Classification:** Output predictions classified as `Real Job` or `Fraudulent Job` with high-confidence ratings.

---

## ⚙️ Environment Configuration

Create a `.env` file in the root directory to store your secret tokens:

```env
# Google Gemini API key for the AI Chatbot Assistant
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL_NAME=gemini-2.5-flash

# (Optional) External PostgreSQL Database URL (e.g., Neon.tech / Supabase)
# If left blank, the app automatically falls back to local SQLite (users.db)
DATABASE_URL=postgresql://neondb_owner:...

# (Optional) Welcome Email SMTP credentials
SMTP_EMAIL=your_gmail_address@gmail.com
SMTP_PASSWORD=your_gmail_app_password
```

---

## 💻 Local Setup & Development

### Prerequisites
*   Python 3.9 or higher (recommended: 64-bit Python 3.9)

### Installation
1.  **Clone / Open the project directory:**
    ```bash
    cd FraudRecruitment
    ```
2.  **Create and activate a virtual environment:**
    *   **Windows:**
        ```cmd
        py -3.9 -m venv .venv
        .venv\Scripts\activate
        ```
    *   **macOS/Linux:**
        ```bash
        python3 -m venv .venv
        source .venv/bin/activate
        ```
3.  **Install dependencies:**
    ```bash
    python -m pip install --upgrade pip setuptools wheel
    pip install -r requirements-runtime.txt
    ```
4.  **Run the Flask application:**
    ```bash
    python app.py
    ```
5.  Access the interface at **`http://127.0.0.1:5000`**.

### 📝 Default Admin Credentials
*   **Email:** `admin@fraud.local`
*   **Password:** `Admin123`

---

## 📓 Jupyter Notebook Setup (Model Training)

To run the Jupyter training kernel:
1.  Activate your virtual environment and install standard requirements:
    ```bash
    pip install -r requirements-windows-jupyter.txt
    ```
2.  Register the Jupyter kernel:
    ```bash
    python -m ipykernel install --user --name fraudrecruitment --display-name "Python (.venv FraudRecruitment)"
    ```
3.  Start Jupyter:
    ```bash
    python -m notebook
    ```
4.  Open `FraudJobDetection.ipynb` and select the **`Python (.venv FraudRecruitment)`** kernel to train the CNN and evaluate predictions.

---

## ☁️ Cloud Deployment Steps (Render + Neon DB)

This application is ready to deploy directly on **Render** (Web Service) using **Neon.tech** (PostgreSQL):

### 1. Database Setup (Neon)
1.  Sign up for a free database on [Neon.tech](https://neon.tech/).
2.  Create a project named `fraud_recruitment`.
3.  Copy the connection string (starts with `postgresql://`).

### 2. Render Deployment
1.  Link your GitHub repository to [Render](https://dashboard.render.com/).
2.  Create a new **Web Service** with the following configurations:
    *   **Runtime:** `Python 3`
    *   **Build Command:** `pip install -r requirements-runtime.txt`
    *   **Start Command:** `gunicorn app:app`
3.  Add the following **Environment Variables**:
    *   `PYTHON_VERSION` = `3.10.13`
    *   `DATABASE_URL` = *(Your Neon PostgreSQL connection string)*
    *   `GEMINI_API_KEY` = *(Your Google Gemini key)*
4.  Click **Deploy Web Service**. Render will automatically compile, connect to the database, and serve your app.

---

## 🔒 Security & Performance Features

*   **⚡ Non-Blocking Emailing:** Welcome emails are dispatched asynchronously inside a background daemon thread with a strict 10-second timeout, completely preventing Gunicorn worker timeouts or connection blocks.
*   **🛡️ Multi-Backend Database:** Uses a smart hybrid adapter. Connects securely to PostgreSQL in production while maintaining native zero-configuration SQLite for local development.
*   **🔑 Password Encryption:** Safe storage of user credentials using secure password hashing (`werkzeug.security`).
