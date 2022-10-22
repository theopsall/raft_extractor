# Raft Extractor

Raft Extractor: Predict Movement and extract intermediate layer outpur as feature vectore with Optical Flow

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) 
![PyTorch](https://img.shields.io/badge/pytorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white) 
![OpenCV](https://img.shields.io/badge/opencv-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)

## Description
Raft Extractor use the raft optical flow model from torch, to predict the movement into the given video. It also provides the extraction of any intermediate layer output
in order to used as feature vector, by analyzing any video in term of seconds and passing the video pair of continuous frames.  Current video analysis implementation
use the opencv library to read the video and extract the frames. The frames are then passed to the raft model to predict the movement and extract the intermediate layer output.
In order to reduce the size of the input video to the mode, the first frame from every second in the video is used as input in the raft_extractor. 
Current implementation of the raft_extractor can use the First, the Last or All the frames from any video.
 



## Installation 
```bash
pip install -r requirements.txt
```
## Usage 

```python

```

## Command-Line-Execution 

```bash
raft_extractor -
```

---

## License
This project is licensed under the MIT License - see the [LICENSE.md](./LICENSE) file for details

## Acknowledgments

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage) project template.
