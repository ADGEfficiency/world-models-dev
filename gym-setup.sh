sudo apt-get update
sudo apt-get install python3.6
sudo apt-get install python3-pip
sudo apt install xvfb
sudo apt-get install swig
sudo apt-get install python-opengl
sudo apt-get install fontconfig
git clone https://github.com/pybox2d/pybox2d
cd /home/ubuntu/mono/world-models/pybox2d
sudo python3 setup.py install
cd /home/ubuntu/mono/world-models
sudo apt install awscli
sudo pip3 install -r requirements.txt
sudo python3 setup.py develop
