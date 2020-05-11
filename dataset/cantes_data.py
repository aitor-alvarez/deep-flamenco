import os
import torchaudio
from torch.utils.data import Dataset
from torchaudio.datasets.utils import walk_files


def load_audio_file(path, file):
    audio_file = os.path.join(path, file)
    label = file[0]
    waveform, sample_rate = torchaudio.load(audio_file)
    return waveform, sample_rate, label


class CantesData(Dataset):
    _ext_audio = ".wav"

    def __init__(self):
        self.path = './data/'
        walker = walk_files(
            self.path, suffix=self._ext_audio, prefix=False, remove_suffix=False
        )
        self._walker = list(walker)

    def __getitem__(self, n):
        fileid = self._walker[n]
        waveform, sample_rate, label = load_audio_file(self.path, fileid)

        return waveform, label

    def __len__(self):
        return len(self._walker)


