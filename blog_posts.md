## Thoughts

Reward function structure - is stay alive as long as posible for both???

Each part should go heavy on the maths - part 1 = autoeconder, part 2 = gaussian mixtures and lstms

cloud lessons
- putting up to s3 rather than just detach volume and connect to gpu instance
- leaving ssd on for ages
- not training directly from s3 (this is probably only useful for prediction)

changed from paper
- sigma at 0.5 in cmaes
- second round
- batch size in vae



## Part 1 - Vision

Epochs for VAE training - 1 or 10

### Testing

Discovered during test suite development on the first test!  For episodes where the horizon was greater than max length, the episode would return done = False for the last transition.  This was fixed by changing the donev ariable based on the max length check

```python
if step >= max_length:
	done = True
```

But this then requires a check of max length again

```python
	step += 1
	if step >= max_length:
		done = True

	transition = {
			'observation': observation,
			'action': action,
			'reward': reward,
			'next_observation': next_observation,
			'done': done
	}

	for key, data in transition.items():
		results[key].append(data)

	if step >= max_length:
		return results

	observation = next_observation
```
