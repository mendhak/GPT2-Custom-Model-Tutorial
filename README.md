## Setup

[Set up TensorFlow](https://www.tensorflow.org/install/pip#linux) by following the Step-by-step instructions, not the quick ones. Include steps to test the NVIDIA GPU and the fix for Ubuntu 22.04. 


Repeated instructions:

```
conda env create -f environment.conda.yml python=3.9
conda activate tf
pip install -r requirements.txt

```


## Disk Usage

A lot of files get downloaded and consume disk space. 

Transformers cache: /home/mendhak/.cache/huggingface/  
Anaconda: /home/mendhak/anaconda3  
Miniconda: /home/mendhak/miniconda3  

