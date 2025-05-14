from flask import Flask, render_template, request
import sqlite3
import os
from datetime import datetime, timedelta
import time

app = Flask(__name__)

def get_db_connection():
    db_path = os.path.join(os.path.dirname(__file__), '..', 'api', 'metrics.db')
    print(f"Attempting to connect to database at: {db_path}")
    if not os.path.exists(db_path):
        print(f"Database file not found at: {db_path}. Creating empty database.")
        conn = sqlite3.connect(db_path)
        conn.close()
        return sqlite3.connect(db_path)
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # Allow accessing columns by name
        print("Database connection established successfully")
        return conn
    except Exception as e:
        print(f"Error connecting to database: {str(e)}")
        return None

@app.route('/')
def dashboard():
    print(f"Received request for dashboard at {datetime.now().strftime('%H:%M:%S')}")
    refresh = request.args.get('refresh', 'false').lower() == 'true'
    conn = get_db_connection()
    if not conn:
        print("Failed to connect to database, returning empty data")
        return render_template('index.html', metrics=[], transition_logs=[],
                              vision_success_rate=0.0, user_correction_rate=0.0,
                              avg_vit_accuracy_card=0.0, avg_vit_accuracy_expiry=0.0, avg_vit_accuracy_security=0.0,
                              is_ready_card=False, is_ready_expiry=False, is_ready_security=False,
                              accuracy_trends={'card_number': [], 'expiry_date': [], 'security_number': []},
                              training_logs=[])

    metrics = []
    transition_logs = []
    accuracy_trends = {'card_number': [], 'expiry_date': [], 'security_number': []}
    training_logs = []

    try:
        cursor = conn.cursor()

        # Fetch metrics
        cursor.execute('''
            SELECT id, timestamp, side, vision_success, vit_accuracy_card, vit_accuracy_expiry, vit_accuracy_security,
                   vision_confidence_card, vision_confidence_expiry, vision_confidence_security,
                   vit_confidence_card, vit_confidence_expiry, vit_confidence_security,
                   user_correction, used_vit_card, used_vit_expiry, used_vit_security
            FROM metrics
            ORDER BY timestamp DESC
        ''')
        metrics = cursor.fetchall()
        print(f"Fetched {len(metrics)} rows from metrics table")

        # Fetch transition logs
        cursor.execute('''
            SELECT id, timestamp, field, used_vit, user_correction, vit_accuracy, vision_accuracy
            FROM transition_log
            ORDER BY timestamp DESC
        ''')
        transition_logs = cursor.fetchall()
        print(f"Fetched {len(transition_logs)} rows from transition_log table")

        # Compute summary statistics
        total_metrics = len(metrics) if metrics else 1  # Avoid division by zero
        vision_success_count = sum(1 for row in metrics if row['vision_success'] == 1)
        user_correction_count = sum(1 for row in metrics if row['user_correction'] == 1)

        vision_success_rate = (vision_success_count / total_metrics) * 100 if total_metrics > 0 else 0.0
        user_correction_rate = (user_correction_count / total_metrics) * 100 if total_metrics > 0 else 0.0

        # Compute average ViT accuracies
        avg_vit_accuracy_card = sum(row['vit_accuracy_card'] for row in metrics) / total_metrics * 100 if total_metrics > 0 else 0.0
        avg_vit_accuracy_expiry = sum(row['vit_accuracy_expiry'] for row in metrics) / total_metrics * 100 if total_metrics > 0 else 0.0
        avg_vit_accuracy_security = sum(row['vit_accuracy_security'] for row in metrics) / total_metrics * 100 if total_metrics > 0 else 0.0

        # Compute transition readiness
        fourteen_days_ago = (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute('''
            SELECT field, timestamp, vit_accuracy, vision_accuracy
            FROM transition_log
            WHERE timestamp >= ?
            ORDER BY timestamp ASC
        ''', (fourteen_days_ago,))
        transition_data = cursor.fetchall()

        # Group transition data by field
        card_number_data = [(row['timestamp'], row['vit_accuracy'], row['vision_accuracy'])
                            for row in transition_data if row['field'] == 'card_number']
        expiry_date_data = [(row['timestamp'], row['vit_accuracy'], row['vision_accuracy'])
                            for row in transition_data if row['field'] == 'expiry_date']
        security_number_data = [(row['timestamp'], row['vit_accuracy'], row['vision_accuracy'])
                              for row in transition_data if row['field'] == 'security_number']

        # Populate accuracy trends
        accuracy_trends['card_number'] = card_number_data
        accuracy_trends['expiry_date'] = expiry_date_data
        accuracy_trends['security_number'] = security_number_data

        # Check transition readiness (ViT >= 99% and better than Vision API for 14 days)
        is_ready_card = (len(card_number_data) >= 14 and
                         all(row[1] >= 0.99 and row[1] > row[2] for row in card_number_data))
        is_ready_expiry = (len(expiry_date_data) >= 14 and
                           all(row[1] >= 0.99 and row[1] > row[2] for row in expiry_date_data))
        is_ready_security = (len(security_number_data) >= 14 and
                             all(row[1] >= 0.99 and row[1] > row[2] for row in security_number_data))

        # Fetch training logs with updated columns
        try:
            cursor.execute('''
                SELECT id, timestamp, step, epoch, total_loss, card_loss, expiry_loss, security_loss, learning_rate
                FROM training_logs
                ORDER BY timestamp DESC
            ''')
            training_logs = cursor.fetchall()
            print(f"Fetched {len(training_logs)} rows from training_logs table")
        except sqlite3.OperationalError as e:
            print(f"Training log table not found or empty: {str(e)}. Initializing empty training logs.")
            training_logs = []

        # Simulate a small delay if refresh is requested to ensure data stability
        if refresh:
            time.sleep(1)

    except Exception as e:
        print(f"Error querying database: {str(e)}")
    finally:
        conn.close()
        print("Database connection closed")

    # Debug the data being passed to the template
    print("Metrics data sample:")
    for row in metrics[:2]:
        print(dict(row))
    print("Transition logs sample:")
    for row in transition_logs[:2]:
        print(dict(row))
    print(f"Vision Success Rate: {vision_success_rate}%")
    print(f"User Correction Rate: {user_correction_rate}%")
    print(f"Average ViT Card Accuracy: {avg_vit_accuracy_card}%")
    print(f"Average ViT Expiry Accuracy: {avg_vit_accuracy_expiry}%")
    print(f"Average ViT Security Accuracy: {avg_vit_accuracy_security}%")
    print(f"Transition Readiness - Card: {'Ready' if is_ready_card else 'Not Ready'}")
    print(f"Transition Readiness - Expiry: {'Ready' if is_ready_expiry else 'Not Ready'}")
    print(f"Transition Readiness - Security: {'Ready' if is_ready_security else 'Not Ready'}")

    return render_template('index.html', metrics=metrics, transition_logs=transition_logs,
                          vision_success_rate=vision_success_rate,
                          user_correction_rate=user_correction_rate,
                          avg_vit_accuracy_card=avg_vit_accuracy_card,
                          avg_vit_accuracy_expiry=avg_vit_accuracy_expiry,
                          avg_vit_accuracy_security=avg_vit_accuracy_security,
                          is_ready_card=is_ready_card,
                          is_ready_expiry=is_ready_expiry,
                          is_ready_security=is_ready_security,
                          accuracy_trends=accuracy_trends,
                          training_logs=training_logs)

if __name__ == '__main__':
    print("Starting monitoring dashboard...")
    app.run(debug=True, host='0.0.0.0', port=8000)