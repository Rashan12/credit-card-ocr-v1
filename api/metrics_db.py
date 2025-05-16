import sqlite3
from threading import Lock

class MetricsDB:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(MetricsDB, cls).__new__(cls)
                    cls._instance.conn = sqlite3.connect("metrics.db", check_same_thread=False)
                    cls._instance.initialize()
        return cls._instance

    def initialize(self):
        cursor = self.conn.cursor()
        # Drop old tables if they exist to ensure schema consistency
        cursor.execute('DROP TABLE IF EXISTS metrics')
        cursor.execute('DROP TABLE IF EXISTS training_logs')
        cursor.execute('DROP TABLE IF EXISTS transition_log')

        cursor.execute('''
            CREATE TABLE metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
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
            CREATE TABLE training_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                step INTEGER,
                batch_size INTEGER,
                total_loss REAL,
                card_loss REAL,
                expiry_loss REAL,
                security_loss REAL,
                learning_rate REAL
            )
        ''')
        cursor.execute('''
            CREATE TABLE transition_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                field TEXT,
                used_vit INTEGER,
                user_correction INTEGER,
                vit_accuracy REAL,
                vision_accuracy REAL
            )
        ''')
        self.conn.commit()

    def log_metrics(self, is_front, side, vision_success, vit_accuracy_card, vit_accuracy_expiry, vit_accuracy_security,
                    vision_confidence_card, vision_confidence_expiry, vision_confidence_security,
                    vit_confidence_card, vit_confidence_expiry, vit_confidence_security,
                    user_correction, used_vit_card, used_vit_expiry, used_vit_security):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO metrics (side, vision_success, vit_accuracy_card, vit_accuracy_expiry, vit_accuracy_security,
                                vision_confidence_card, vision_confidence_expiry, vision_confidence_security,
                                vit_confidence_card, vit_confidence_expiry, vit_confidence_security,
                                user_correction, used_vit_card, used_vit_expiry, used_vit_security)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (side, vision_success, vit_accuracy_card, vit_accuracy_expiry, vit_accuracy_security,
              vision_confidence_card, vision_confidence_expiry, vision_confidence_security,
              vit_confidence_card, vit_confidence_expiry, vit_confidence_security,
              user_correction, used_vit_card, used_vit_expiry, used_vit_security))
        self.conn.commit()

    def log_training(self, step, batch_size, total_loss, card_loss, expiry_loss, security_loss, learning_rate):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO training_logs (step, batch_size, total_loss, card_loss, expiry_loss, security_loss, learning_rate)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (step, batch_size, total_loss, card_loss, expiry_loss, security_loss, learning_rate))
        self.conn.commit()

    def log_transition(self, field, used_vit, user_correction, vit_accuracy, vision_accuracy):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO transition_log (field, used_vit, user_correction, vit_accuracy, vision_accuracy)
            VALUES (?, ?, ?, ?, ?)
        ''', (field, used_vit, user_correction, vit_accuracy, vision_accuracy))
        self.conn.commit()

    def close(self):
        with self._lock:
            if self.conn:
                self.conn.close()
                self._instance = None