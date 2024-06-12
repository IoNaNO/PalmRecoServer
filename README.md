# Palmprint Recongnition Backend
## How to start the server
**This code can be only run on Linux.**
1. Install the required packages
```bash
conda create -n palm python=3.9
conda activate palm
pip install -r requirements-dev.txt # for development
```
To install `edcc` package, you can refer to [this link](https://github.com/IoNaNO/EDCC-Palmprint-Recognition.git).

2. Start the server
```bash
python server.py
```
## Use docker to start the server
1. Build the docker image
```bash
sudo docker build -t palm:lastest .
```
2. Run the docker container
```bash
sudo docker run -d -p 5000:5000 palm:lastest
```
