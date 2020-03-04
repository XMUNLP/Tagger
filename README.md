# Tagger

This is the source code for the paper "[Deep Semantic Role Labeling with Self-Attention](https://arxiv.org/abs/1712.01586)".

## Contents

* [Basics](#basics)
  * [Notice](#notice)
  * [Prerequisites](#prerequisites)
* [Walkthrough](#walkthrough)
  * [Data](#data)
  * [Training](#training)
  * [Decoding](#decoding)
* [Benchmarks](#benchmarks)
* [Pretrained Models](#pretrained-models)
* [License](#license)
* [Citation](#citation)
* [Contact](#contact)

## Basics

### Notice

The original code used in the paper is implemented using TensorFlow 1.0, which is obsolete now. We have re-implemented our methods using PyTorch, which is based on [THUMT](https://github.com/THUNLP-MT/THUMT). The differences are as follows:

* We only implement DeepAtt-FFN model
* Model ensemble are currently not available

Please check the git history to use TensorFlow implementation.

### Prerequisites

* Python 3
* PyTorch
* TensorFlow-2.0 (CPU version)
* GloVe embeddings and `srlconll` scripts

## Walkthrough

### Data

#### Training Data

We follow the same procedures described in the [deep_srl](https://github.com/luheng/deep_srl) repository to convert the CoNLL datasets.
The GloVe embeddings and `srlconll` scripts can also be found in that link.

If you followed these procedures, you can find that the processed data has the following format:
```
2 My cats love hats . ||| B-A0 I-A0 B-V B-A1 O
```

*The CoNLL datasets are not publicly available. We cannot provide these datasets.*

#### Vocabulary

You can use the `build_vocab.py` script to generate vocabularies. The command is described as follows:

```[bash]
python tagger/scripts/build_vocab.py --limit LIMIT --lower TRAIN_FILE OUTPUT_DIR
```

where `LIMIT` specifies the vocabulary size. This command will create two vocabularies named `vocab.txt` and `label.txt` in the `OUTPUT_DIR`.

### Training

Once you finished the procedures described above, you can start the training stage.

#### Preparing the validation script

An external validation script is required to enable the validation functionality.
Here's the validation script we used to train an FFN model on the CoNLL-2005 dataset.
Please make sure that the validation script can run properly.

```[bash]
#!/usr/bin/env bash
SRLPATH=/PATH/TO/SRLCONLL
TAGGERPATH=/PATH/TO/TAGGER
DATAPATH=/PATH/TO/DATA
EMBPATH=/PATH/TO/GLOVE_EMBEDDING
DEVICE=0

export PYTHONPATH=$TAGGERPATH:$PYTHONPATH
export PERL5LIB="$SRLPATH/lib:$PERL5LIB"
export PATH="$SRLPATH/bin:$PATH"

python $TAGGERPATH/tagger/bin/predictor.py \
  --input $DATAPATH/conll05.devel.txt \
  --checkpoint train \
  --model deepatt \
  --vocab $DATAPATH/deep_srl/word_dict $DATAPATH/deep_srl/label_dict \
  --parameters=device=$DEVICE,embedding=$EMBPATH/glove.6B.100d.txt \
  --output tmp.txt

python $TAGGERPATH/tagger/scripts/convert_to_conll.py tmp.txt $DATAPATH/conll05.devel.props.gold.txt output
perl $SRLPATH/bin/srl-eval.pl $DATAPATH/conll05.devel.props.* output
```

#### Training command

The command below is what we used to train a model on the CoNLL-2005 dataset. The content of `run.sh` is described in the above section.

```[bash]
#!/usr/bin/env bash
SRLPATH=/PATH/TO/SRLCONLL
TAGGERPATH=/PATH/TO/TAGGER
DATAPATH=/PATH/TO/DATA
EMBPATH=/PATH/TO/GLOVE_EMBEDDING
DEVICE=[0]

export PYTHONPATH=$TAGGERPATH:$PYTHONPATH
export PERL5LIB="$SRLPATH/lib:$PERL5LIB"
export PATH="$SRLPATH/bin:$PATH"

python $TAGGERPATH/tagger/bin/trainer.py \
  --model deepatt \
  --input $DATAPATH/conll05.train.txt \
  --output train \
  --vocabulary $DATAPATH/deep_srl/word_dict $DATAPATH/deep_srl/label_dict \
  --parameters="save_summary=false,feature_size=100,hidden_size=200,filter_size=800,"`
               `"residual_dropout=0.2,num_hidden_layers=10,attention_dropout=0.1,"`
               `"relu_dropout=0.1,batch_size=4096,optimizer=adadelta,initializer=orthogonal,"`
               `"initializer_gain=1.0,train_steps=600000,"`
               `"learning_rate_schedule=piecewise_constant_decay,"`
               `"learning_rate_values=[1.0,0.5,0.25,],"`
               `"learning_rate_boundaries=[400000,50000],device_list=$DEVICE,"`
               `"clip_grad_norm=1.0,embedding=$EMBPATH/glove.6B.100d.txt,script=run.sh"
```

### Decoding

The following is the command used to generate outputs:

```[bash]
#!/usr/bin/env bash
SRLPATH=/PATH/TO/SRLCONLL
TAGGERPATH=/PATH/TO/TAGGER
DATAPATH=/PATH/TO/DATA
EMBPATH=/PATH/TO/GLOVE_EMBEDDING
DEVICE=0

python $TAGGERPATH/tagger/bin/predictor.py \
  --input $DATAPATH/conll05.test.wsj.txt \
  --checkpoint train/best \
  --model deepatt \
  --vocab $DATAPATH/deep_srl/word_dict $DATAPATH/deep_srl/label_dict \
  --parameters=device=$DEVICE,embedding=$EMBPATH/glove.6B.100d.txt \
  --output tmp.txt

```

## Benchmarks

We've performed 4 runs on CoNLL-05 datasets. The results are shown below.

|  Runs  | Dev-P | Dev-R | Dev-F1 | WSJ-P | WSJ-R | WSJ-F1 | BROWN-P | BROWN-R | BROWN-F1 |
| :----: | :---: | :---: | :----: | :---: | :---: | :----: | :-----: | :-----: | :------: |
| Paper  |  82.6 | 83.6  |  83.1  |  84.5 |  85.2 |  84.8  |   73.5  |  74.6   |   74.1   |
| Run0   |  82.9 | 83.7  |  83.3  |  84.6 |  85.0 |  84.8  |   73.5  |  74.0   |   73.8   |
| Run1   |  82.3 | 83.4  |  82.9  |  84.4 |  85.3 |  84.8  |   72.5  |  73.9   |   73.2   |
| Run2   |  82.7 | 83.6  |  83.2  |  84.8 |  85.4 |  85.1  |   73.2  |  73.9   |   73.6   |
| Run3   |  82.3 | 83.6  |  82.9  |  84.3 |  84.9 |  84.6  |   72.3  |  73.6   |   72.9   |

## Pretrained Models

The pretrained models of TensorFlow implementation can be downloaded at [Google Drive](https://drive.google.com/open?id=1jvBlpOmqGdZEqnFrdWJkH1xHsGU2OjiP).

## LICENSE

BSD

## Citation

If you use our codes, please cite our paper:

```
@inproceedings{tan2018deep,
  title = {Deep Semantic Role Labeling with Self-Attention},
  author = {Tan, Zhixing and Wang, Mingxuan and Xie, Jun and Chen, Yidong and Shi, Xiaodong},
  booktitle = {AAAI Conference on Artificial Intelligence},
  year = {2018}
}
```

## Contact

This code is written by Zhixing Tan. If you have any problems, feel free to send an <a href="mailto:playinf@stu.xmu.edu.cn">email</a>.
