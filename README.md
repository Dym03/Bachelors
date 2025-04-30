# Traffic sign detection using CNN

## This repository is a part of a bachelor thesis, below will be stated how to install and run. Also short description of each script will be given.

### Installation
```bash
    git clone https://github.com/Dym03/Bachelors.git
    python -m venv env
    source env/bin/activate
    pip install -r requirements.txt
``` 

### Running scripts

#### All of the scripts are now written so that they should be run from the root directory, not from the src directory.
For testing this script might be most usefull, the model name is specified inside the src/video.py file the video can be specified from the cmd as an argument.
```bash
    source env/bin/activate
    python src/video.py {VIDEO_PATH}
```

## Project tree

* [src](./src)
  * [analyze_dataset.py](./src/analyze_dataset.py) Makes quick analyses of a dataset and makes a bar plot out of it.
  * [evaluate.py](./src/evaluate) Evaluation script with which results in the work were presented.
  * [generate_dataset.py](./src/generate_dataset.py) Generates dataset from background images and signs.
  * [generate_yaml.py](./src/generate_yaml.py) Creates a yaml for a dataset that is required by YOLO from the Ultralytics library.
  * [json2yolo.py](./src/json2yolo.py) Convert json annotations of GTSDB to YOLO.
  * [mapping_dicts.py](./src/mapping_dicts.py) Stores mapping dicts for Mapillary and GTSDB datasets.
  * [predict.py](./src/predict.py) Simple predict script for Faster RCNN for images.
  * [traffic_sign_dataset.py](./src/traffic_sign_dataset.py) Dataset class for loading the datset data.
  * [train_yolo.py](./src/train_yolo.py) Training script for yolo
  * [train.py](./src/train.py) Training script for Faster RCNN model
  * [validate_yolo.py](./src/train.py) Validation script for yolo models, for comparison to my script
  * [video.py](./src/train.py) Script used to test the YOLO models on real data
* [models](./models)
  * [yolo_models](./models/yolo_models/) Directory with yolo models trained
  * [Faster_RCNN_best.pt](./models/Faster_RCNN_best.pt) Best Faster RCNN model trained
* [data](./data)
  * [signs](./data/signs) Directory with created png signs

* [README.md](./README.md)