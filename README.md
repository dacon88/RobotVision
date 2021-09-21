# RobotVision

**Author**: Davide Congiu

**Project Valid for Machine Learning course 2020/2021**

RobotVision implements classifier based on fine-tuning of AlexNet crafted on iCubWorld dataset.

ICubWorld dataset: https://robotology.github.io/iCubWorld/#icubworld-1-modal

For the project development pytorch and ML course notebooks have been used as inspiration:
* https://github.com/unica-ml/ml/blob/master/notebooks/lab05.ipynb
* https://github.com/unica-ml/ml/blob/master/notebooks/lab06.ipynb

* https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

Many thanks to professor Battista Biggio for the support

**Installation**

clone repo: git clone https://github.com/dacon88/RobotVision.git

install required packages:
* _pytorch_
* _pip3 install -r requirements.txt_


**Usage**

run first training script:
* _python3 path_to_repo/RobotVision/training.py_

Then run inference script:
* _python3 path_to_repo/RobotVision/inference.py_
