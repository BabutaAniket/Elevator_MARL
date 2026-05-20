import sqlite3
import os
import json
from datetime import datetime


DB_PATH = os.path.join(os.path.dirname(__file__), 'simulation.db')


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS simulation_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        algorithm TEXT NOT NULL,
        start_time TEXT NOT NULL,
        end_time TEXT,
        sim_duration_ticks INTEGER DEFAULT 0,
        sim_start_hour REAL DEFAULT 9.0,
        speed_multiplier REAL DEFAULT 1.0,
        custom_arrival_rate REAL,
        status TEXT DEFAULT 'running',
        config_json TEXT
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS tick_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER NOT NULL,
        tick INTEGER NOT NULL,
        total_waiting INTEGER,
        avg_wait REAL,
        max_wait REAL,
        p95_wait REAL,
        total_delivered INTEGER,
        throughput_per_min INTEGER,
        total_energy REAL,
        tick_energy REAL,
        floor_waiting_json TEXT,
        lifts_json TEXT,
        FOREIGN KEY (run_id) REFERENCES simulation_runs(id)
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS training_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER,
        episode INTEGER,
        total_reward_a REAL,
        total_reward_b REAL,
        avg_wait REAL,
        total_delivered INTEGER,
        total_energy REAL,
        loss REAL,
        timestamp TEXT
    )''')

    # Migrate: add avg_journey_time column if this is an existing DB
    try:
        c.execute('ALTER TABLE tick_stats ADD COLUMN avg_journey_time REAL DEFAULT 0')
        conn.commit()
    except Exception:
        pass  # Column already exists
    # Migrate: store accurate in-memory run-level summary (avoids single-tick distortion)
    try:
        c.execute('ALTER TABLE simulation_runs ADD COLUMN run_summary_json TEXT')
        conn.commit()
    except Exception:
        pass  # Column already exists
    conn.commit()
    conn.close()


def create_run(name: str, algorithm: str, sim_start_hour: float = 9.0,
               speed_multiplier: float = 1.0, custom_arrival_rate: float = None,
               config: dict = None) -> int:
    conn = get_db()
    c = conn.cursor()
    c.execute('''INSERT INTO simulation_runs
        (name, algorithm, start_time, sim_start_hour, speed_multiplier, custom_arrival_rate, config_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)''',
        (name, algorithm, datetime.now().isoformat(), sim_start_hour,
         speed_multiplier, custom_arrival_rate,
         json.dumps(config) if config else None))
    conn.commit()
    run_id = c.lastrowid
    conn.close()
    return run_id


def save_tick(run_id: int, info: dict):
    conn = get_db()
    c = conn.cursor()
    c.execute('''INSERT INTO tick_stats
        (run_id, tick, total_waiting, avg_wait, max_wait, p95_wait,
         total_delivered, throughput_per_min, total_energy, tick_energy,
         avg_journey_time, floor_waiting_json, lifts_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (run_id, info['tick'], info['total_waiting'], info['avg_wait'],
         info['max_wait'], info['p95_wait'], info['total_delivered'],
         info['throughput_per_min'], info['total_energy'], info['tick_energy'],
         info.get('avg_journey_time', 0),
         json.dumps(info['floor_waiting']),
         json.dumps(info['bank_a_lifts'] + info['bank_b_lifts'])))
    conn.commit()
    conn.close()


def finish_run(run_id: int, ticks: int):
    conn = get_db()
    c = conn.cursor()
    c.execute('''UPDATE simulation_runs SET end_time=?, sim_duration_ticks=?, status='completed'
                 WHERE id=?''', (datetime.now().isoformat(), ticks, run_id))
    conn.commit()
    conn.close()


def save_run_summary(run_id: int, summary: dict):
    """Persist accurate in-memory aggregate stats alongside the run record."""
    conn = get_db()
    conn.execute('UPDATE simulation_runs SET run_summary_json=? WHERE id=?',
                 (json.dumps(summary), run_id))
    conn.commit()
    conn.close()


def get_runs():
    conn = get_db()
    rows = conn.execute('''
        SELECT r.*,
               ROUND(AVG(t.avg_wait), 1) AS mean_avg_wait,
               MAX(t.total_delivered) AS final_delivered
        FROM simulation_runs r
        LEFT JOIN tick_stats t ON t.run_id = r.id
        GROUP BY r.id
        ORDER BY r.id DESC
    ''').fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_run_stats(run_id: int, sample_every: int = 1):
    conn = get_db()
    rows = conn.execute(
        'SELECT * FROM tick_stats WHERE run_id=? AND tick % ? = 0 ORDER BY tick',
        (run_id, sample_every)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def save_training_log(run_id, episode, reward_a, reward_b, avg_wait, delivered, energy, loss):
    conn = get_db()
    conn.execute('''INSERT INTO training_logs
        (run_id, episode, total_reward_a, total_reward_b, avg_wait, total_delivered, total_energy, loss, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (run_id, episode, reward_a, reward_b, avg_wait, delivered, energy, loss,
         datetime.now().isoformat()))
    conn.commit()
    conn.close()


def get_training_logs(run_id: int):
    conn = get_db()
    rows = conn.execute('SELECT * FROM training_logs WHERE run_id=? ORDER BY episode', (run_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_run_summary(run_id: int) -> dict:
    conn = get_db()
    run = conn.execute('SELECT * FROM simulation_runs WHERE id=?', (run_id,)).fetchone()
    if not run:
        conn.close()
        return {}
    stats = conn.execute('''
        SELECT
            MAX(total_delivered)                    AS total_delivered,
            ROUND(AVG(avg_wait), 2)                 AS mean_avg_wait,
            MAX(max_wait)                           AS peak_max_wait,
            ROUND(AVG(p95_wait), 2)                 AS mean_p95_wait,
            MAX(total_energy)                       AS total_energy,
            ROUND(AVG(throughput_per_min), 2)       AS avg_throughput,
            MAX(total_waiting)                      AS peak_waiting,
            ROUND(AVG(avg_journey_time), 2)         AS mean_avg_journey_time,
            COUNT(*)                                AS sample_count
        FROM tick_stats WHERE run_id=?
    ''', (run_id,)).fetchone()
    conn.close()
    result = dict(run)
    if stats:
        result.update(dict(stats))
    # Prefer the accurate in-memory summary saved at run end over tick_stats
    # aggregates (which can be distorted when only one tick row was saved).
    if result.get('run_summary_json'):
        try:
            result.update(json.loads(result['run_summary_json']))
        except Exception:
            pass
    if result.get('config_json'):
        try:
            result['config'] = json.loads(result['config_json'])
        except Exception:
            result['config'] = {}
    else:
        result['config'] = {}
    return result


def clear_db():
    conn = get_db()
    conn.execute('DELETE FROM tick_stats')
    conn.execute('DELETE FROM training_logs')
    conn.execute('DELETE FROM simulation_runs')
    conn.commit()
    conn.close()


init_db()
