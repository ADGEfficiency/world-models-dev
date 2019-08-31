import os


results_dir = os.path.join(os.environ['HOME'], 'world-models-experiments')

env_params = {
    'num_actions': 3
}

# input = latent_size + num_actions
# output = latent_size

memory_results_dir = os.path.join(results_dir, 'memory-training')
memory_params = {
    'input_dim': 35,
    'output_dim': 32,
    'num_timesteps': 999,
    'batch_size': 100,
    'epochs': 20,
    'lstm_nodes': 256,
    'num_mix': 5,
    'load_model': True,
    'results_dir': memory_results_dir
}

vae_results_dir = os.path.join(results_dir, 'vae-training')
vae_params = {
    'latent_dim': 32,
    'learning_rate': 0.0001,
    'load_model': True,
    'results_dir': vae_results_dir
}
