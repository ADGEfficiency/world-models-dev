import numpy as np


from world_models import memory_params, Memory

model = Memory(**memory_params)


# def sample_sequence
temperature = 1.0

#  length of actions determines length of sequence
actions = np.random.rand(10, 3).astype(np.float32)

#  latent only
previous_x = np.zeros((1, 1, model.output_dim)).astype(np.float32)

#  [h_state, c_state]
h_state, c_state = model.lstm.get_zero_hidden_state(inputs=previous_x)

sequence = np.zeros((len(actions), model.output_dim))

#  only collect hidden state (not cell state)
initial_states = np.zeros((len(actions), model.lstm.nodes))
final_states = np.zeros((len(actions), model.lstm.nodes))

for step, action in enumerate(actions):
    action = action.reshape(1, 1, -1)

    x = np.concatenate((previous_x, action), axis=2)
    assert x.shape == (1, 1, 35)

    initial_states[step] = h_state

    next_x, h_state, c_state = model(
        x=x,
        state=[h_state, c_state],
        temperature=temperature
    )

    x = next_x

    sequence[step] = next_x
    final_states[step] = h_state

