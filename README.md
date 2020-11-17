# deep-flamenco
Deep learning architectures for flamenco singer identification using motivic patterns.

## Requirements

1. Pytorch
2. Torchaudio
3. Numpy

Steps to run the models:

1. Select the model in the file main.py
2. Adjust in the dataset the folder of your data
3. python main.py

### Data

Motivic patterns extracted from Corpus COFLA can be found in this repository: https://github.com/aitor-mir/flamenco-motifs 

### Research

If you intend to use this software for research, please cite the following article:

```
@inproceedings{Alvarez2020,
  author={Aitor Arronte Alvarez and Elsayed Sabry Abdelaal Issa},
  title={{Learning Intonation Pattern Embeddings for Arabic Dialect Identification}},
  year=2020,
  booktitle={Proc. Interspeech 2020},
  pages={472--476},
  doi={10.21437/Interspeech.2020-2906},
  url={http://dx.doi.org/10.21437/Interspeech.2020-2906}
}
```
