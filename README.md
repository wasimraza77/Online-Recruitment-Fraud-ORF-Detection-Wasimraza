# Online Recruitment Fraud Detection

A robust, AI-powered web application that detects fraudulent job postings to protect job seekers from scams. This system uses Natural Language Processing (NLP) with BERT embeddings and a Convolutional Neural Network (CNN) to classify job descriptions as "Real" or "Fraudulent".

## 🚀 Live Demo
**[Try the Live Application Here!](https://online-recruitment-fraud-orf-detection-a9kg.onrender.com)**

## 🌟 Features
* **Batch CSV Upload:** Upload datasets of job postings to generate a comprehensive fraud detection report.
* **Manual Check:** Copy and paste a single job description to get an instant risk assessment.
* **AI Chatbot Assistant:** An interactive Gemini-powered chatbot to help users identify red flags in job offers.
* **Interactive Dashboards:** Visual statistics, confidence scores, and risk insights.
* **User Authentication:** Secure login and registration system.

## 🛠️ Technology Stack
* **Frontend:** HTML5, Vanilla CSS (Custom Design System), JavaScript
* **Backend:** Python, Flask, Gunicorn
* **Database:** SQLite (Local) / PostgreSQL (Production)
* **Machine Learning:** 
  * `sentence-transformers` (distilbert) for semantic text embeddings
  * Keras/TensorFlow (exported as NumPy) for CNN classification
* **AI Integration:** Google Gemini API for the chatbot

## 💻 Local Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/wasimraza77/Online-Recruitment-Fraud-ORF-Detection-Wasimraza.git
cd Online-Recruitment-Fraud-ORF-Detection-Wasimraza
```

### 2. Create a Virtual Environment
```bash
# On Windows
python -m venv .venv
.venv\Scripts\activate

# On macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements-runtime.txt
```

### 4. Environment Variables
Create a `.env` file in the root directory and add your API keys:
```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```
*(Optional) To use a remote PostgreSQL database, add `DATABASE_URL=postgresql://...`*

### 5. Run the Application
```bash
python app.py
```
Open your browser and navigate to `http://127.0.0.1:5000/`.

## ☁️ Deployment (Render)
This project is configured to be deployed on Render:
1. Connect your GitHub repository to Render as a **Web Service**.
2. **Build Command:** `pip install -r requirements-runtime.txt`
3. **Start Command:** `gunicorn app:app`
4. Add your `GEMINI_API_KEY` to the Environment Variables.
5. *(Optional)* Add a `DATABASE_URL` pointing to a Neon/Supabase PostgreSQL database. Note: If using Neon, remove `&channel_binding=require` from the connection string to avoid compatibility issues.

## ⚠️ Notes for Free Tier Hosting
If you are hosting this application on a free tier service (like Render Free with 512MB RAM), uploading **new** custom CSV files may cause the server to crash due to Out-Of-Memory (OOM) errors when loading the PyTorch AI model. 
* To demonstrate the batch upload feature on the free tier, please upload the provided `Dataset/fake_job_postings.csv` file, which utilizes cached embeddings to bypass memory limits.

---
*Developed for B.Tech Final Year Project*
