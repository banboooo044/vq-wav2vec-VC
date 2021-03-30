# Vocoder-free Any-to-Many Voice Conversion

```text
.
┠ README.md
┣ core ┳ train.py
       ┠ convert.py
       ┠ model.py
       ┠ dataset.py
       ┠ evaluation.py
       ┣ utils.py
       ┣ config
       ┣ checkpoints
       ┣ tensorboard
       ┣ outputs
       ┣ datasets ━ cmu_arctic ┳ cmu_us_rms_arctic
                               ┣ cmu_us_slt_arctic
                               ┃
                               :
                               ┠ train.json
                               ┠ val.json
                               ┠ test.json

```


## 1. Setup

* Create empty directories (or symbolic links) named "checkpoints", "tensorboard", "outputs" in "core" directory. 
* Download dataset (ex. CMU_ARCTIC) and place it in the "datasets" directory. 
    url: http://www.festvox.org/cmu_arctic/

* Download pre-trained vq-wav2vec model and place it in a directory i.e. checkpoints/enc/vq-wav2vec.pt
    url: https://github.com/pytorch/fairseq/tree/master/examples/wav2vec

* Create json config files for training(train.json), validation(val.json), testing(test.json) in the directory of specific dataset. 

## 2. Training

Example

```bash
python3 train.py enc_checkpoint=checkpoints/enc/vq-wav2vec.pt tensorboard_dir=tensorboard/test checkpoint_dir=checkpoints/test
```

## 3. Testing (Conversion)

Example

```bash
python3 convert.py enc_checkpoint=checkpoints/enc/vq-wav2vec.pt tensorboard_dir=tensorboard/test checkpoint_dir=checkpoints/test
```
