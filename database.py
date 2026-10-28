import sqlite3
import random
from datetime import datetime, timedelta

DB_NAME = "enterprise_ai.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS inference_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            usage_hours REAL,
            support_tickets INTEGER,
            contract_months INTEGER,
            monthly_bill REAL,
            churn_risk_score REAL,
            churn_prediction INTEGER,
            latency_ms REAL
        )
    ''')
    conn.commit()

    # Pre-seed initial historical logs if table is empty
    cursor.execute('SELECT COUNT(*) FROM inference_logs')
    count = cursor.fetchone()[0]
    if count == 0:
        base_time = datetime.now() - timedelta(days=2)
        mock_data = []
        for i in range(15):
            log_time = str(base_time + timedelta(hours=i * 3))
            usage = round(random.uniform(20.0, 90.0), 1)
            tickets = random.randint(0, 4)
            contract = random.choice([6, 12, 24])
            bill = round(random.uniform(40.0, 180.0), 2)
            score = round(random.uniform(0.18, 0.48), 4)
            pred = 1 if score > 0.5 else 0
            lat = round(random.uniform(6.5, 12.0), 2)
            mock_data.append((log_time, usage, tickets, contract, bill, score, pred, lat))
        
        cursor.executemany('''
            INSERT INTO inference_logs 
            (timestamp, usage_hours, support_tickets, contract_months, monthly_bill, churn_risk_score, churn_prediction, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', mock_data)
        conn.commit()
    conn.close()

def log_prediction(usage_hours, support_tickets, contract_months, monthly_bill, score, prediction, latency_ms):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO inference_logs (timestamp, usage_hours, support_tickets, contract_months, monthly_bill, churn_risk_score, churn_prediction, latency_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), usage_hours, support_tickets, contract_months, monthly_bill, score, prediction, latency_ms))
    conn.commit()
    conn.close()

def get_recent_logs(limit=50):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, timestamp, usage_hours, support_tickets, monthly_bill, churn_risk_score, churn_prediction, latency_ms 
        FROM inference_logs ORDER BY id DESC LIMIT ?
    ''', (limit,))
    rows = cursor.fetchall()
    conn.close()
    return rows

def get_recent_scores(limit=50):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('SELECT churn_risk_score FROM inference_logs ORDER BY id DESC LIMIT ?', (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [r[0] for r in rows]

init_db()