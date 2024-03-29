# deep-flamenco
Deep learning architectures for flamenco singer identification using motivic patterns.

## Requirements

1. Pytorch
2. Torchaudio
3. Numpy

Steps to run the models:

1. Select the model in the file main.py
2. Adjust in the dataset the folder of your data
3. Run the following command:
```
python main.py
```
### Data

Motivic patterns extracted from Corpus COFLA can be found in this repository: https://github.com/aitor-mir/flamenco-motifs 

### Research

If you intend to use this software for research, please cite the following articles:

```
@article{alvarez2021motivic,
  title={Motivic Pattern Classification of Music Audio Signals Combining Residual and LSTM Networks.},
  author={Alvarez, Aitor Arronte and G{\'o}mez, Francisco},
  journal={International Journal of Interactive Multimedia \& Artificial Intelligence},
  volume={6},
  number={6},
  year={2021}
}

@article{arronte2020singer,
  title={Singer Identification Using Convolutional Acoustic Motif Embeddings},
  author={Arronte Alvarez, Aitor and Gomez-Martin, Francisco},
  journal={arXiv e-prints},
  pages={arXiv--2008},
  year={2020}
}
```
