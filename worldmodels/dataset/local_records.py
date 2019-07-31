import os

home = os.environ['HOME']
results_dir = os.path.join(home, 'world-models-experiments')

def list_local_records(record_dir, incl):
    record_dir = os.path.join(results_dir, record_dir)
    files = os.listdir(record_dir)
    return sorted([f for f in files if incl in f])
