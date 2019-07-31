## Environment

Simple = pong, complex = energy grid

https://claudioa.itch.io/power-the-grid

## Setup
```
brew install cmake boost boost-python sdl2 swig wget
git clone https://github.com/openai/gym
cd gym
pip install -e '.[atari]'
```

Parallelizum (to avoid in progress in another thread when fork called)
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

https://worldmodels.github.io/

http://blog.otoro.net/2017/11/12/evolving-stable-strategies/

http://blog.otoro.net/2017/10/29/visual-evolution-strategies/

## Lit review

https://github.com/ctallec/world-models

Car racing action space
- steering (-1.0 to 1.0)
- acceleration (0 to 1.0)
- brakes (0 to 1.0

## [WM Experiments Github](https://github.com/hardmaru/WorldModelsExperiments)

> the experiments work on gym 0.9.x and does NOT work on gym 0.10.x

`pip install gym==0.9.4`
pip3 install box2d box2d-kengz
pip install Pillow

###  extract.bash & extract.py

generate a random agent

collect observations, append to list

encode our observation to get z (model.encode_obs))
use z to get action (which is just a random action) (model.get_action)



