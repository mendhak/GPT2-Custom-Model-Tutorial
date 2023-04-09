Repository to support [a tutorial](https://towardsdatascience.com/train-gpt-2-in-your-own-language-fc6ad4d60171) on using GPT-2 to train your own model. Tested on Ubuntu 22.04 with NVidia RTX 2080 Ti. 

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

Transformers cache: ~/.cache/huggingface/  
Anaconda: ~/anaconda3  
Miniconda: ~/miniconda3  

