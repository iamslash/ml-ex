import os
import urllib.request

import torch
import torchaudio
from torch.utils.data import Dataset


def speech_commands_collator(batch):
    inputs, labels = zip(*batch)
    inputs = torch.stack(inputs)
    labels = torch.tensor(labels, dtype=torch.long)
    return {"input_values": inputs, "labels": labels}


def load_audio(audio_path, target_length=16000):
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.shape[1] > target_length:
        waveform = waveform[:, :target_length]
    else:
        waveform = torch.cat([waveform, torch.zeros(1, target_length - waveform.shape[1])], dim=1)
    return waveform


def download_progress_hook(blocks_downloaded, block_size, total_size):
    percentage = 100 * blocks_downloaded * block_size / total_size
    print(f"Downloaded {percentage:.1f}% ({blocks_downloaded * block_size}/{total_size})")


class SpeechCommandsDataset(Dataset):
    def __init__(self, root_dir, subset='train', transform=None):
        self.root_dir = root_dir
        self.subset = subset
        self.transform = transform

        # Define the list of classes and their corresponding index
        self.classes = ['bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'go', 'happy', 'house', 'left',
                        'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three',
                        'tree', 'two', 'up', 'wow', 'yes', 'zero']
        self.class_to_index = {self.classes[i]: i for i in range(len(self.classes))}

        # Create a list of audio files and their corresponding class label
        self.audio_files = []
        audio_dir = root_dir
        if not os.path.exists(audio_dir):
            os.makedirs(audio_dir)
            if subset == 'train':
                url = 'http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
                filename = 'speech_commands_v0.02.tar.gz'
            else:
                url = 'https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
                filename = 'speech_commands_test_set_v0.02.tar.gz'
            filepath = os.path.join(root_dir, filename)
            urllib.request.urlretrieve(url, filepath, reporthook=download_progress_hook)
            os.system(f'tar -xzf {filepath} -C {root_dir}')
            os.remove(filepath)
        if subset == 'train':
            with open(os.path.join(root_dir, 'validation_list.txt')) as f:
                lines = f.readlines()
                for line in lines:
                    label = line.strip().split('/')[0].lower()
                    path = line.strip()
                    if label in self.class_to_index:
                        self.audio_files.append((os.path.join(audio_dir, path), self.class_to_index[label]))
        else:
            with open(os.path.join(root_dir, 'testing_list.txt' if subset == 'test' else 'validation_list.txt')) as f:
                lines = f.readlines()
                for line in lines:
                    label = line.strip().split('/')[0].lower()
                    path = line.strip()
                    if label in self.class_to_index:
                        self.audio_files.append((os.path.join(audio_dir, path), self.class_to_index[label]))

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        # Load the audio file and its corresponding class label
        audio_path, label = self.audio_files[idx]
        audio_tensor = load_audio(audio_path)  # you will
        return audio_tensor, label
