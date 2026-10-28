from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import pickle
import time
import random
import numpy as np
from database import log_prediction, get_recent_scores, get_recent_logs

app = FastAPI(title="TechM Enterprise AI Core", version="2.0")

class ChurnModel(nn.Module):
    def __init__(self, input_dim):
        super(ChurnModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc2(self.relu(self.fc1(x))))

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

model = ChurnModel(input_dim=4)
model.load_state_dict(torch.load("churn_model.pth"))
model.eval()

class ClientData(BaseModel):
    usage_hours: float
    support_tickets: int
    contract_months: int
    monthly_bill: float

@app.get("/health")
def health_check():
    return {"status": "healthy", "engine": "PyTorch v2.x", "database": "Connected"}

@app.post("/predict")
def predict_churn(data: ClientData):
    start_time = time.time()
    
    features = np.array([[data.usage_hours, data.support_tickets, data.contract_months, data.monthly_bill]])
    scaled = scaler.transform(features)
    tensor_in = torch.tensor(scaled, dtype=torch.float32)

    with torch.no_grad():
        score = model(tensor_in).item()
    
    prediction = 1 if score > 0.5 else 0
    latency_ms = round((time.time() - start_time) * 1000, 2)

    log_prediction(
        data.usage_hours, data.support_tickets, data.contract_months,
        data.monthly_bill, round(score, 4), prediction, latency_ms
    )

    return {
        "churn_risk_score": round(score, 4),
        "prediction": "HIGH RISK" if prediction == 1 else "STABLE",
        "latency_ms": latency_ms
    }

@app.get("/drift-check")
def drift_check():
    scores = get_recent_scores(limit=50)
    baseline_mean = 0.38
    current_mean = float(np.mean(scores))
    drift_delta = round(current_mean - baseline_mean, 4)
    drift_detected = abs(drift_delta) > 0.18

    return {
        "samples_analyzed": len(scores),
        "current_mean_risk": round(current_mean, 4),
        "baseline_mean_risk": baseline_mean,
        "drift_delta": drift_delta,
        "drift_detected": drift_detected,
        "system_status": "FLAGGED: Retraining Trigger Recommended" if drift_detected else "OPTIMAL: Serving in Bounds"
    }

@app.get("/logs")
def get_logs():
    raw_logs = get_recent_logs(limit=20)
    return [{
        "id": r[0], "time": r[1], "usage": r[2], "tickets": r[3],
        "bill": r[4], "score": r[5], "prediction": r[6], "latency": r[7]
    } for r in raw_logs]

@app.post("/simulate-batch")
def simulate_batch():
    # Simulates 15 random customer requests
    for _ in range(15):
        usage = random.uniform(5.0, 100.0)
        tickets = random.randint(0, 9)
        contract = random.choice([1, 6, 12, 24])
        bill = random.uniform(30.0, 280.0)
        
        feat = scaler.transform([[usage, tickets, contract, bill]])
        with torch.no_grad():
            score = model(torch.tensor(feat, dtype=torch.float32)).item()
        
        pred = 1 if score > 0.5 else 0
        lat = round(random.uniform(4.5, 9.8), 2)
        log_prediction(usage, tickets, contract, bill, round(score, 4), pred, lat)
    
    return {"message": "Generated 15 simulated production requests successfully."}