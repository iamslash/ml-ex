import argparse
import os

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset, Dataset
from transformers import Wav2Vec2Model, Trainer, TrainingArguments, TrainerCallback, Wav2Vec2Config

from data import SpeechCommandsDataset, speech_commands_collator


class AudioClassifier(nn.Module):
    def __init__(self, wav2vec2_model, num_classes):
        super(AudioClassifier, self).__init__()
        self.wav2vec2_model = wav2vec2_model
        self.pre_classifier = nn.Linear(wav2vec2_model.config.hidden_size, wav2vec2_model.config.hidden_size)
        self.dropout = nn.Dropout(wav2vec2_model.config.final_dropout)
        self.classifier = nn.Linear(wav2vec2_model.config.hidden_size, num_classes)

    def forward(self, input_values, labels=None):
        input_values = input_values.squeeze(1).squeeze(1)
        features = self.wav2vec2_model(input_values).last_hidden_state
        features = features.mean(dim=1)
        features = self.pre_classifier(features)
        features = nn.ReLU()(features)
        features = self.dropout(features)
        logits = self.classifier(features)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.classifier.out_features), labels.view(-1))
            return loss, logits
        else:
            return logits

    def save_pretrained(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(output_dir, "best_model.pt"))


class SaveBestModelCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.best_loss = float('inf')

    def on_evaluate(self, args, state, control, model=None, logs=None, **kwargs):
        if logs is None:
            logs = {}

        eval_loss = logs.get("eval_loss", None)
        print(f"on_evaluate, eval_loss: {eval_loss}")
        if eval_loss is None or eval_loss < self.best_loss:
            self.best_loss = eval_loss
            model.save_pretrained(self.output_dir)
            print(f"Model saved at {self.output_dir} with eval_loss: {eval_loss}")


class CustomSubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--train_subset', type=int, default=None, help='Limit the training dataset size.')
    parser.add_argument('--output_dir', type=str, default='./.results', help='Directory to save the trained model.')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the pretrained model file.')
    args = parser.parse_args()

    model_name = "facebook/wav2vec2-base-960h"
    config = Wav2Vec2Config.from_pretrained(model_name)
    config.architectures = ["Wav2Vec2Model"]
    wav2vec2_model = Wav2Vec2Model.from_pretrained(model_name, config=config)
    model = AudioClassifier(wav2vec2_model, num_classes=30)

    # Load the pretrained model if provided
    if args.model_path is not None:
        model.load_state_dict(torch.load(args.model_path))

    train_dataset = SpeechCommandsDataset(root_dir='.data', subset='train')
    valid_dataset = SpeechCommandsDataset(root_dir='.data', subset='test')

    if args.train_subset is not None:
        train_dataset = CustomSubset(train_dataset, indices=list(range(args.train_subset)))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=32,
        logging_dir="./.logs",
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    optimizer = AdamW(model.parameters(), lr=5e-5)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=speech_commands_collator,
        callbacks=[SaveBestModelCallback(args.output_dir)],
        optimizers=(optimizer, None),
    )

    trainer.train()


if __name__ == "__main__":
    main()
