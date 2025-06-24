

# API Inference Services & Predictions

---

### API Inference Services

After deployment, the project exposes a robust **FastAPI-based prediction service** for real-time and batch inference. This API enables business applications, dashboards, and external clients to obtain instant, reliable forecasts.

**Key Features:**

* **Live REST API:**
  Available at `http://localhost:8000` (or your cloud endpoint).
* **Interactive Documentation:**
  Auto-generated at `/docs` (Swagger UI).
* **Multiple Model Support:**
  Select and query different registered models (`best_model`, `random_forest`, etc.).
* **Dynamic Feature Handling:**
  Automatically extracts required features for any model.
* **Endpoints:**

  * `/predict`: Single prediction (POST, JSON input)
  * `/batch_predict`: Batch prediction
  * `/health`: Health check/status
  * `/metrics`: Performance metrics (latency, usage)
  * `/models`: List available models

**Example Usage:**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Total_Quantity": 150.5,
    "Avg_Price": 18.50,
    "Transaction_Count": 25,
    "Month": 6,
    "DayOfWeek": 2,
    "IsWeekend": 0,
    "Wholesale_Price": 14.0,
    "Loss_Rate": 8.5
  }'
```

**API Test Results:**

* 100% test pass rate (health, single/batch prediction, edge cases, model switching, load testing).
* **Latency:** \~73ms per request.
* **All models available:** 14 models, with full dynamic feature order support.
* **Swagger docs:** Easy to explore and interact with API.

---

## **API Service Diagram**
![API Inference Services](api_inf_1.png)
<br/><br/>
![API Inference Services](api_inf_2.png)

---



