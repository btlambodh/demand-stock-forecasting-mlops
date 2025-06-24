

# Performance Monitoring & Drift Detection

---

### **What It Does**

The monitoring system in this project ensures **your deployed AI models remain fast, healthy, and accurate**. It’s a *critical MLOps capability* for any business relying on real-time ML predictions.

---

### **Performance Monitoring**

**Continuous, automated process** that checks:

* **System Resources:** CPU, memory, disk, uptime.
* **API & Model Health:** Model count, response latency, error rates, API status.
* **Data Health:** Quality score, freshness (hours since last validation), data validation checks.

**Outputs:**

* Live dashboard (Dash/Flask): Visualizes system and model health, available at e.g. `http://localhost:51453`.
* Health summary (auto-generated, see logs):

  ```
  System Health: HEALTHY
  CPU: 0.1% | Memory: 1.7% | Disk Free: 37.0GB
  API Health: 14 models loaded, 200ms response time
  Data Quality: 99.3%, Freshness: 0.7 hours
  Active Alerts: 0
  ```
* **Alerts:** Any resource, latency, or health anomaly triggers an alert and is logged for downstream action.

---

### **Drift Detection**

**Purpose:** Detects when your input data distribution changes (“drifts”) so much that it may threaten model reliability.

**How it works:**

* **Reference Data:** Uses the same features from your training set.
* **Current Data:** Continuously analyzes new data (e.g., validation, live API inputs).
* **Checks:** Every 60 minutes (customizable), runs statistical tests (e.g., KS, PSI) on all features.
* **Thresholds:** Example: If >25% of features drift (drift ratio), triggers a drift alert.
* **Reports:** Saves a detailed HTML drift report listing drifted features and summary metrics.

**Drift detection output (sample):**

* Overall drift: **True**
* Drift ratio: **54.8%**
* Top drifted features: Item Code, Date, Days\_Since\_Epoch, Avg\_Quantity, Year
* Alert saved: `data/monitoring/reports/alert_data_drift_xxxx.json`

---

### **Monitoring & Drift Dashboard**

* Launches automatically, see link in terminal (e.g., `http://localhost:51453`).
* **Dashboards display:**

  * Real-time resource metrics
  * Model latency and status
  * Data health and quality
  * Drift state, drift ratio, and recent alerts

---

### **Testing and Status**

* `make monitoring-test`: Verifies all monitoring systems, reports “HEALTHY” or issues.
* `make monitoring-status`: Checks all running monitors and dashboard ports.
* **Alert testing:** Triggers alert samples and verifies local alert logging.

---

### **Why It Matters for the Business**

* **Keeps ML trustworthy:** No “silent failure”—you always know when predictions need attention.
* **Proactive:** Early warning for data or system issues means faster business response.
* **Audit-ready:** All drift events, health checks, and resource stats are logged and reportable.

---
![Performance Monitor Drift Detection](perf_monitor_drift_detect.png)