# Tagger
This is the code for the paper "Deep Semantic Role Labeling with Self-Attention".

## Usage
### Prerequisites
* python2
* A newer version of TensorFlow
* GloVe embeddings and srlconll scripts

### Data
We follow the same procedure to convert CoNLL data as describe in the [deep_srl](https://github.com/luheng/deep_srl) repository.
The GloVe embeddings and srlconll scripts can also be found in that link.

### Vocabulary
You can use the vocabularies in the resource directory to replicate our results. If you want to use build a new vocabulary, you can use the build_vocab.py.
```
python ~/scripts/tensorflow/tagger/scripts/build_vocab.py --limit LIMIT --lower TRAIN_FILE OUTPUT_DIR
```
where LIMIT specifies the vocabulary size. This command will create two vocabularies named vocab.txt and label.txt in the OUTPUT_DIR.

### Convert data format
If you follow the above procedure, the processed data is a plain text with the following format:
```
2 My cats love hats . ||| B-A0 I-A0 B-V B-A1 O
```
The plain text should converted to tf.Record format first using input_convert.py. Using the following command:
```
python scripts/input_converter.py --input_path TRAIN_FILE --output_name NAME --output_dir OUTPUT_DIR 
                                  --vocab WORD_DICT LABEL_DICT --num_shards NUM_SHARDS --shuffle --lower
```
The above command will create NUM_SHARDS files named "NAME-\*-of-\*" in the OUTPUT_DIR.


### Training and Validating
Once you finished these procedures above, you can start training to replicate our results. The following contents describe how to train and validate a model.
* Preparing the validation script
An external validation script is required to enable the validation functionality. Here's the validation script we use to train an FFN model on the CoNLL-2005 dataset.
```
python tagger/main.py predict --data_path conll05.devel.txt --model_dir train --model_name deepatt --vocab_path word_dict label_dict --device_list 0 --decoding_params="decode_batch_size=512" --model_params="num_hidden_layers=10,feature_size=100,hidden_size=200,filter_size=800"
python scripts/convert_to_conll.py conll05.devel.txt.deepatt.decodes conll05.devel.props.gold.txt output
sh ~/data/srl/run_eval.sh ~/data/srl/conll05/conll05.devel.props.* output
```
* Training command
The command below is what we used to train an model on the CoNLL-2005 dataset.
```
python tagger/main.py train
    --data_path TRAIN_PATH --model_dir train --model_name deepatt 
    --vocab_path word_dict label_dict --emb_path glove.6B.100d.txt 
    --model_params=feature_size=100,hidden_size=200,filter_size=800,residual_dropout=0.2,
                   num_hidden_layers=10,attention_dropout=0.1,relu_dropout=0.1 
    --training_params=batch_size=4096,eval_batch_size=1024,optimizer=Adadelta,initializer=orthogonal,
                      use_global_initializer=false,initializer_gain=1.0,train_steps=600000,
                      learning_rate_decay=piecewise_constant,learning_rate_values=[1.0,0.5,0.25],
                      learning_rate_boundaries=[400000,500000],device_list=[0],clip_grad_norm=1.0 
    --validation_params=script=run.sh
```


### Decoding
```
python tagger/main.py predict 
    --data_path ~/data/srl/conll05/conll05.test.wsj.txt 
    --model_dir train/best --model_name deepatt 
    --vocab_path word_dict label_dict 
    --device_list 0 
    --decoding_params="decode_batch_size=512" 
    --model_params="num_hidden_layers=10,feature_size=100,hidden_size=200,filter_size=800" 
    --emb_path embedding/glove.6B.100d.txt
```

### Model Ensemble
```
python ~/scripts/tensorflow/tagger/main.py ensemble 
    --data_path conll05/conll05.devel.txt 
    --checkpoints ../run1/model.ckpt-588644 
    --output_name output 
    --vocab_path word_dict1 word_dict2 label_dict 
    --model_params=feature_size=100,hidden_size=200,filter_size=800,num_hidden_layers=10 
    --device_list 0 
    --model_name deepatt
```

## Contact
This code is written by Zhixing Tan. If you have any problems, feel free to send an <a href="mailto:playinf@stu.xmu.edu.cn">email</a>.

## LICENSE
 BSD
