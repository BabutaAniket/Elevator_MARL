from flask import Flask, render_template, jsonify, request
import torch
from simulation import SimulationRunner
from database import get_runs, get_run_stats, get_training_logs, get_run_summary, clear_db
from config import FLOOR_NAMES

# Allow PyTorch to use multiple cores for training backprop; single thread
# was throttling PPO updates during Auto-Train.
torch.set_num_threads(4)

app = Flask(__name__)
runner = SimulationRunner()


@app.route('/')
def index():
    return render_template('index.html', floors=FLOOR_NAMES)


@app.route('/api/start', methods=['POST'])
def start_sim():
    data = request.get_json()
    name = data.get('name', 'Simulation')
    algorithm = data.get('algorithm', 'scan')
    duration = int(data.get('duration', 60))
    start_hour = float(data.get('start_hour', 9.0))
    speed = int(data.get('speed', 10))
    custom_rate = data.get('custom_arrival_rate')
    if custom_rate is not None:
        custom_rate = float(custom_rate)
    train_ppo = data.get('train_ppo', False)

    run_id = runner.start(name, algorithm, duration, start_hour, speed, custom_rate, train_ppo)
    return jsonify({'run_id': run_id, 'status': 'started'})


@app.route('/api/stop', methods=['POST'])
def stop_sim():
    runner.stop()
    return jsonify({'status': 'stopped'})


@app.route('/api/pause', methods=['POST'])
def pause_sim():
    runner.pause()
    return jsonify({'status': 'paused'})


@app.route('/api/resume', methods=['POST'])
def resume_sim():
    runner.resume()
    return jsonify({'status': 'resumed'})


@app.route('/api/state')
def get_state():
    return jsonify(runner.get_live_state())


@app.route('/api/burst', methods=['POST'])
def add_burst():
    data = request.get_json()
    floor = data.get('floor', 'G')
    count = int(data.get('count', 20))
    direction = data.get('direction', 'random')
    runner.add_burst(floor, count, direction)
    return jsonify({'status': 'burst_added', 'floor': floor, 'count': count})


@app.route('/api/runs')
def list_runs():
    return jsonify(get_runs())


@app.route('/api/runs/<int:run_id>/stats')
def run_stats(run_id):
    sample = int(request.args.get('sample', 10))
    return jsonify(get_run_stats(run_id, sample))


@app.route('/api/runs/<int:run_id>/training')
def training_logs(run_id):
    return jsonify(get_training_logs(run_id))


@app.route('/results/<int:run_id>')
def results_page(run_id):
    summary = get_run_summary(run_id)
    return render_template('results.html', summary=summary, run_id=run_id)


@app.route('/api/runs/<int:run_id>/summary')
def run_summary_api(run_id):
    return jsonify(get_run_summary(run_id))


@app.route('/api/db/clear', methods=['POST'])
def clear_database():
    clear_db()
    return jsonify({'status': 'cleared'})


@app.route('/api/auto_train', methods=['POST'])
def start_auto_train():
    data = request.get_json()
    n_runs = int(data.get('n_runs', 50))
    config = {
        'algorithm':  data.get('algorithm', 'ppo'),
        'duration':   int(data.get('duration', 60)),
        'start_hour': float(data.get('start_hour', 8.0)),
        'speed':      int(data.get('speed', 5000)),
        'base_name':  data.get('base_name', 'AutoTrain'),
    }
    result = runner.start_auto_train(config, n_runs)
    return jsonify(result)


@app.route('/api/auto_train/status')
def auto_train_status():
    return jsonify(runner.get_auto_train_status())


@app.route('/api/auto_train/summary')
def auto_train_summary():
    return jsonify(runner.auto_train_summary)


@app.route('/api/auto_train/stop', methods=['POST'])
def stop_auto_train():
    runner.stop_auto_train()
    return jsonify({'status': 'stopped'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000
