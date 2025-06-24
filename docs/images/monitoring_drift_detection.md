# Performance Monitoring & Drift Detection - Demand Stock Forecasting MLOps (Chinese Produce Market RMB)

## Overview

Provides comprehensive, automated monitoring capabilities ensuring deployed AI models remain fast, healthy, and accurate in production environments. Implements continuous performance tracking, statistical drift detection, and proactive alerting systems critical for maintaining reliable MLOps operations. Features real-time dashboards, automated health checks, and statistical analysis to detect data distribution changes that could threaten model reliability. Built with enterprise-grade monitoring infrastructure supporting business-critical decision making through early warning systems and detailed audit trails for regulatory compliance.

---

## The Monitoring Architecture

The monitoring system provides enterprise-grade observability across all MLOps pipeline components with automated alerting and comprehensive reporting:

| Monitoring Component | Description | Metrics Tracked | Alert Triggers |
| -------------------- | ----------- | --------------- | -------------- |
| **System Performance** | Resource utilization monitoring | CPU, Memory, Disk, Uptime | Resource thresholds exceeded |
| **API Health** | Service availability and performance | Response time, error rates, model count | Latency >2s, error rate >5% |
| **Data Quality** | Input data validation and freshness | Quality scores, validation checks | Quality <95%, data staleness |
| **Drift Detection** | Statistical distribution analysis | Feature drift ratios, KS tests | >25% features drifting |

### **Monitoring Configuration and Setup**

* **Monitoring Interval**: `MONITORING_INTERVAL = 60` seconds (configurable)
* **Dashboard Port**: Dynamic port assignment (8050+ range)
* **Reference Data**: `REFERENCE_DATA = data/processed/train.parquet`
* **Current Data**: `CURRENT_DATA = data/processed/validation.parquet`
* **Local Mode**: Supports both local and cloud deployment configurations

---

## Performance Monitoring Capabilities

**1. System Resource Monitoring**

Continuous automated monitoring of critical system metrics:

* **CPU Utilization**: Real-time CPU usage tracking with trend analysis
* **Memory Management**: RAM consumption monitoring with memory leak detection
* **Disk Space**: Storage utilization monitoring with capacity planning alerts
* **System Uptime**: Service availability tracking with downtime notifications

**2. API and Model Health Monitoring**

| Health Metric | Monitoring Frequency | Normal Range | Alert Threshold |
| ------------- | :------------------: | :----------: | :-------------: |
| **Response Time** | Real-time | <200ms | >2000ms |
| **Error Rate** | Per minute | <1% | >5% |
| **Model Count** | Continuous | 14+ models | Model unavailable |
| **API Status** | Health checks | 200 OK | Non-200 responses |

**3. Data Health and Quality Monitoring**

* **Data Quality Scores**: Automated validation against expected schema and business rules
* **Data Freshness**: Monitoring time since last data validation and ingestion
* **Completeness Checks**: Missing value detection and completeness scoring
* **Anomaly Detection**: Statistical outlier identification in incoming data streams

---

## Drift Detection System

**1. Statistical Drift Analysis**

Comprehensive drift detection using multiple statistical methods:

* **Kolmogorov-Smirnov Tests**: Distribution comparison for continuous features
* **Population Stability Index (PSI)**: Measures population shifts over time
* **Chi-Square Tests**: Categorical feature distribution analysis
* **Jensen-Shannon Divergence**: Probabilistic distribution comparison

**2. Drift Detection Configuration**

| Parameter | Default Value | Description | Customizable |
| --------- | :-----------: | ----------- | :----------: |
| **Detection Interval** | 60 minutes | Frequency of drift analysis | Yes |
| **Drift Threshold** | 25% | Percentage of features before alert | Yes |
| **Reference Period** | Training data | Baseline for comparison | Yes |
| **Feature Coverage** | All features | Features monitored for drift | Yes |

**3. Drift Detection Results**

Sample drift detection output showing comprehensive analysis:

```
Overall Drift Status: DETECTED
Drift Ratio: 54.8% (49 of 89 features)
Statistical Significance: p < 0.001

Top Drifted Features:
- Item_Code: PSI = 0.89 (HIGH)
- Date: KS = 0.67 (MEDIUM)  
- Days_Since_Epoch: PSI = 0.72 (HIGH)
- Avg_Quantity: KS = 0.43 (MEDIUM)
- Year: PSI = 0.56 (MEDIUM)

Alert Generated: data/monitoring/reports/alert_data_drift_20250624_153000.json
Report Saved: data/monitoring/reports/drift_analysis_20250624_153000.html
```

---

## Real-time Dashboard and Visualization

**1. Interactive Monitoring Dashboard**

* **Live System Metrics**: Real-time CPU, memory, and disk utilization graphs
* **API Performance**: Response time trends, error rate monitoring, throughput metrics
* **Model Health**: Model availability status, prediction accuracy trends
* **Drift Visualization**: Feature drift heatmaps, statistical test results, historical trends

**2. Health Summary Reports**

Automated health summaries generated every monitoring cycle:

```
System Health Status: HEALTHY
Resource Utilization:
  CPU: 0.1% | Memory: 1.7% | Disk Free: 37.0GB
API Performance:
  Models Loaded: 14 | Avg Response Time: 73ms
Data Quality:
  Quality Score: 99.3% | Data Freshness: 0.7 hours
Active Alerts: 0 | System Uptime: 99.9%
```

**3. Alert Management System**

* **Threshold-Based Alerts**: Configurable thresholds for all monitored metrics
* **Alert Prioritization**: Critical, high, medium, and low severity levels
* **Alert Aggregation**: Intelligent grouping to prevent alert fatigue
* **Alert History**: Complete audit trail of all alerts and resolutions

---

## Automated Monitoring Workflow & Makefile Integration

Comprehensive monitoring automation with intelligent process management and health checking. Makefile targets provide complete monitoring lifecycle management with enhanced debugging and reporting capabilities.

```bash
# Start complete monitoring system with all components
make monitoring-start

# Check detailed status of all monitoring systems
make monitoring-status

# Stop all monitoring processes with cleanup
make monitoring-stop

# Restart monitoring systems (stop + start)
make monitoring-restart

# Run one-time drift detection analysis
make detect-drift

# Generate comprehensive performance report
make performance-report

# View monitoring logs and system status
make monitoring-logs

# Test monitoring system components
make monitoring-test

# Debug monitoring issues with detailed diagnostics
make monitoring-debug

# Clean monitoring temporary files and logs
make monitoring-clean

# Individual component management
make monitoring-dashboard-only      # Start only dashboard
make monitoring-performance-only    # Start only performance monitoring
make monitoring-drift-only         # Start only drift detection

# Emergency operations
make emergency-stop                # Stop all services including monitoring
```

**Monitoring Configuration Variables (from Makefile):**

```bash
# Monitoring system configuration
MONITORING_INTERVAL=60            # Health check interval (seconds)
DASHBOARD_PORT=8050              # Default dashboard port (auto-assigned)
REFERENCE_DATA=data/processed/train.parquet      # Baseline data
CURRENT_DATA=data/processed/validation.parquet   # Current data for comparison

# Custom configuration examples:
make monitoring-start MONITORING_INTERVAL=30     # Faster monitoring
make detect-drift REFERENCE_DATA=custom_baseline.parquet
```

**Key Scripts Referenced:**

* [`src/monitoring/performance_monitor.py`](src/monitoring/performance_monitor.py)
* [`src/monitoring/drift_detector.py`](src/monitoring/drift_detector.py)
* [`config.yaml`](config.yaml)

---

## Business Impact & Production Operations

* **Proactive Issue Detection**: Early warning systems prevent production failures and maintain business continuity
* **Model Reliability Assurance**: Continuous validation ensures AI predictions remain trustworthy for critical business decisions
* **Regulatory Compliance**: Comprehensive audit trails and monitoring logs support governance and compliance requirements
* **Cost Optimization**: Resource monitoring and automated scaling prevent over-provisioning and reduce operational costs
* **Business Intelligence**: Rich monitoring data enables data-driven infrastructure and model optimization decisions

---

## Alert System & Incident Response

**1. Alert Categories and Thresholds**

| Alert Type | Severity | Trigger Condition | Response Time SLA | Escalation |
| ---------- | :------: | ----------------- | :---------------: | :--------: |
| **System Resources** | Critical | CPU >90%, Memory >95% | 2 minutes | Immediate |
| **API Performance** | High | Response time >2s | 5 minutes | 15 minutes |
| **Model Drift** | Medium | >25% features drifting | 1 hour | 24 hours |
| **Data Quality** | Medium | Quality score <95% | 30 minutes | 4 hours |

**2. Automated Response Actions**

* **Resource Alerts**: Automatic scaling recommendations and resource optimization suggestions
* **Performance Alerts**: Service health checks and automatic restart procedures
* **Drift Alerts**: Model retraining recommendations and data pipeline validation
* **Quality Alerts**: Data source validation and input pipeline debugging

**3. Incident Documentation**

* **Alert Logs**: Complete timestamp and severity tracking for all alerts
* **Resolution Tracking**: Time to resolution and remediation action logging
* **Root Cause Analysis**: Automated analysis of alert patterns and contributing factors
* **Performance Trending**: Historical analysis for capacity planning and optimization

---

## Monitoring Testing & Validation

**1. Comprehensive Testing Framework**

* **Component Testing**: Individual monitoring component validation and health checks
* **Integration Testing**: End-to-end monitoring workflow validation
* **Performance Testing**: Monitoring system overhead and resource consumption analysis
* **Stress Testing**: High-load monitoring system behavior validation

**2. Test Coverage Results**

| Test Category | Coverage | Status | Performance Impact |
| ------------- | :------: | :----: | :----------------: |
| **Performance Monitor** | 100% | HEALTHY | <1% CPU overhead |
| **Drift Detection** | 100% | HEALTHY | <2% memory overhead |
| **Dashboard System** | 100% | HEALTHY | <0.5% system impact |
| **Alert System** | 100% | HEALTHY | Real-time response |

**3. Monitoring System Validation**

```bash
# Comprehensive monitoring system test
make monitoring-test

# Example test output:
Testing Performance Monitor: PASSED
Testing Drift Detector: PASSED  
Testing Dashboard System: PASSED
Testing Alert System: PASSED
All monitoring components validated successfully!
```

---

## Advanced Analytics & Reporting

**1. Drift Analysis Reports**

* **Feature-Level Analysis**: Individual feature drift scores and statistical significance
* **Temporal Trends**: Drift patterns over time with seasonal analysis
* **Impact Assessment**: Model performance correlation with detected drift
* **Recommendations**: Automated suggestions for model retraining and data pipeline updates

**2. Performance Analytics**

* **Resource Utilization Trends**: Historical resource consumption patterns and optimization opportunities
* **API Performance Metrics**: Response time distributions, error rate analysis, and throughput optimization
* **Model Health Scores**: Aggregated model performance metrics and reliability indicators
* **System Reliability**: Uptime analysis, failure pattern identification, and resilience metrics

**3. Business Intelligence Integration**

* **Executive Dashboards**: High-level KPIs and business impact metrics
* **Operational Reports**: Detailed technical metrics for DevOps and ML engineering teams
* **Compliance Reports**: Regulatory compliance documentation and audit trail generation
* **Cost Analysis**: Infrastructure cost tracking and optimization recommendations

---

## Performance Monitoring Dashboard

![Performance Monitor Drift Detection](./images/perf_monitor_drift_detect.png)

---

## Quick Start

1. **Start the complete monitoring system** with all components:
   ```bash
   make monitoring-start
   ```

2. **Check monitoring system status** to verify all components are running:
   ```bash
   make monitoring-status
   ```

3. **Access the monitoring dashboard** (port will be displayed in terminal output):
   ```
   Dashboard available at: http://localhost:[dynamic-port]
   ```

4. **Run drift detection analysis** to check for data distribution changes:
   ```bash
   make detect-drift
   ```

5. **Generate performance report** for comprehensive system analysis:
   ```bash
   make performance-report
   ```

6. **Test monitoring components** to validate system health:
   ```bash
   make monitoring-test
   ```

7. **View monitoring logs** for detailed system information:
   ```bash
   make monitoring-logs
   ```

---

## Questions?

For monitoring system configuration, alert tuning, or production deployment guidance, please contact [Bhupal Lambodhar](mailto:btiduwarlambodhar@sandiego.edu) or open an issue in the [GitHub repository](https://github.com/btlambodh/demand-stock-forecasting-mlops).