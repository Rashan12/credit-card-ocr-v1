import sqlite3

db_path = r"C:\Users\PM_User\Documents\credit-card-ocr\api\metrics.db"

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create tables
cursor.execute('''
CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    side TEXT,
    vision_success INTEGER,
    vit_accuracy_card REAL,
    vit_accuracy_expiry REAL,
    vit_accuracy_security REAL,
    vision_confidence_card REAL,
    vision_confidence_expiry REAL,
    vision_confidence_security REAL,
    vit_confidence_card REAL,
    vit_confidence_expiry REAL,
    vit_confidence_security REAL,
    user_correction INTEGER,
    used_vit_card INTEGER,
    used_vit_expiry INTEGER,
    used_vit_security INTEGER
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS transition_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    field TEXT,
    used_vit INTEGER,
    user_correction INTEGER,
    vit_accuracy REAL,
    vision_accuracy REAL
)
''')

# Insert sample data
cursor.execute('''
INSERT INTO metrics (timestamp, side, vision_success, vit_accuracy_card, vit_accuracy_expiry, vit_accuracy_security, vision_confidence_card, vision_confidence_expiry, vision_confidence_security, vit_confidence_card, vit_confidence_expiry, vit_confidence_security, user_correction, used_vit_card, used_vit_expiry, used_vit_security)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
''', ('2025-05-14 11:00:00', 'front', 1, 0.95, 0.90, 0.85, 0.90, 0.85, 0.80, 0.98, 0.95, 0.92, 0, 1, 0, 0))

cursor.execute('''
INSERT INTO metrics (timestamp, side, vision_success, vit_accuracy_card, vit_accuracy_expiry, vit_accuracy_security, vision_confidence_card, vision_confidence_expiry, vision_confidence_security, vit_confidence_card, vit_confidence_expiry, vit_confidence_security, user_correction, used_vit_card, used_vit_expiry, used_vit_security)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
''', ('2025-05-14 11:01:00', 'back', 0, 0.88, 0.92, 0.87, 0.85, 0.88, 0.82, 0.96, 0.94, 0.90, 1, 0, 1, 0))

cursor.execute('''
INSERT INTO transition_log (timestamp, field, used_vit, user_correction, vit_accuracy, vision_accuracy)
VALUES (?, ?, ?, ?, ?, ?)
''', ('2025-05-14 11:00:00', 'card_number', 1, 0, 0.98, 0.90))

cursor.execute('''
INSERT INTO transition_log (timestamp, field, used_vit, user_correction, vit_accuracy, vision_accuracy)
VALUES (?, ?, ?, ?, ?, ?)
''', ('2025-05-14 11:01:00', 'expiry_date', 1, 1, 0.95, 0.88))

# Commit changes and close
conn.commit()
conn.close()
print("Database initialized with sample data.")