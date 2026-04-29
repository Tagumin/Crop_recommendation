# Crop Recommendation AI - Flask API

This system is a Machine Learning-based **Flask API** application for crop recommendation based on soil and climate conditions, with a web interface and Docker support.

---

## 📦 Features

* Crop prediction using Random Forest model
* Inputs: N, P, K, pH, temperature, humidity, rainfall
* Detailed crop information (pests, diseases, treatments)
* Flask API backend
* HTML + JS frontend
* Docker & non-Docker support

---

# 🚀 How to Run WITHOUT Docker

## 1. Clone the project

```bash
git clone <your-repo>
cd project-folder
```

## 2. Create virtual environment

```bash
python -m venv env
```

## 3. Activate environment

### Windows:

```bash
env\Scripts\activate
```

### Mac/Linux:

```bash
source env/bin/activate
```

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

## 5. Run Flask app

```bash
python app.py
```

## 6. Open application

```
http://localhost:5000
```

---

# 🐳 How to Run with Docker

## 1. Build Docker image

```bash
docker build -t crop-app .
```

## 2. Run container

```bash
docker run -d -p 5000:5000 crop-app
```

## 3. Check running container

```bash
docker ps
```

## 4. Open application

```
http://localhost:5000
```

---

# ⚙️ API Structure

## 🔹 Health Check

```
GET /health
```

Response:

```json
{
  "status": "ok",
  "model_accuracy": 96.5
}
```

---

## 🔹 Crop Prediction

```
POST /predict
```

### Request Body

```json
{
  "N": 90,
  "P": 42,
  "K": 43,
  "temperature": 20.5,
  "humidity": 80,
  "ph": 6.5,
  "rainfall": 200
}
```

### Response

```json
{
  "crop": "rice",
  "emoji": "🌾",
  "confidence": 95.2,
  "model_accuracy": 96.5
}
```

---

## 🔹 Crop Info

```
GET /crop-info/<crop_name>
```

Example:

```
/crop-info/rice
```

---

## 🔹 Pest Info

```
GET /pest-info/<pest_name>
```

Example:

```
/pest-info/aphids
```

---

# 📁 Project Structure

```
.
├── app.py
├── crop_model.pkl
├── requirements.txt
├── index.html
├── pest_image/
└── Dockerfile
```

---

# ⚠️ Important Notes

* Make sure port **5000 is not used by another application**
* If Docker fails, check logs:

```bash
docker logs <container_id>
```

* Ensure `.pkl` model file exists before running

---


