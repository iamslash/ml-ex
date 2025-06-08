import argparse
import torch
import torchaudio
from torchaudio.models import Wav2Vec2Model

from train import AudioClassifier
from transformers import Wav2Vec2Processor
from data import load_audio


def load_model(model_path, num_classes):
    model_name = "facebook/wav2vec2-base-960h"
    wav2vec2_model = Wav2Vec2Model.from_pretrained(model_name)
    model = AudioClassifier(wav2vec2_model, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def predict(model, processor, audio_path):
    audio_tensor = load_audio(audio_path)
    input_values = processor(audio_tensor, return_tensors="pt", padding=True, truncation=True).input_values
    with torch.no_grad():
        logits = model(input_values)
    predicted_class_idx = torch.argmax(logits, dim=1).item()
    return predicted_class_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model.')
    parser.add_argument('--audio_path', type=str, required=True, help='Path to the input audio file.')
    args = parser.parse_args()

    num_classes = 30
    model = load_model(args.model_path, num_classes)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    classes = ['bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'go', 'happy', 'house', 'left',
               'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three',
               'tree', 'two', 'up', 'wow', 'yes', 'zero']

    predicted_class_idx = predict(model, processor, args.audio_path)
    predicted_class = classes[predicted_class_idx]
    print(f"Predicted class: {predicted_class}")


if __name__ == "__main__":
    main()
