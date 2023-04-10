Repository to support [a tutorial](https://towardsdatascience.com/train-gpt-2-in-your-own-language-fc6ad4d60171) on using GPT-2 to train your own model. Tested on Ubuntu 22.04 with NVidia RTX 2080 Ti. 

## Setup

[Set up TensorFlow](https://www.tensorflow.org/install/pip#linux) by following the Step-by-step instructions, not the quick ones. Include steps to test the NVIDIA GPU and the fix for Ubuntu 22.04. 


Repeated instructions:

```
conda env create -f environment.conda.yml python=3.9
conda activate tf
pip install -r requirements.txt
```

## Running

Following along with the tutorial, I"ve set up the various scripts to sort of correspond with the sections, although several of the sections kind of overlap with each other. 

### Download from Wikipedia

The first script downloads a massive bz2 from Wikipedia with millions of articles. It's about 20GB+ and takes ages to download. 

```
python 1.wiki.download.py
```

### Tokenise

Next script braeks down all the text into tokens. 

```
python 2.tokenise.py
```

### Create the model

This takes ages. Many more ages than the Wikipedia download. It attempts to load all the files into a single string, and doing that for millions of articles results in OOM crashes, so I've batched the model creation into chunks. As this runs it stores the model data into `./model_custom`.

```
python 3.initialise.model.py
```

### Use the model to predict

This uses the model saved in `./model_custom` to produce an output based on the value in the input `text` variable. 




## Disk Usage

A lot of files get downloaded and consume disk space. 

Transformers cache: ~/.cache/huggingface/  
Anaconda: ~/anaconda3  
Miniconda: ~/miniconda3  



## Other

https://keras.io/examples/generative/text_generation_gpt/

https://keras.io/examples/generative/text_generation_with_miniature_gpt/