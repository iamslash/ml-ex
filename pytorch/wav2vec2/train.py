import argparse

from torch import nn
from torch.utils.data import DataLoader, Subset
from transformers import Wav2Vec2Model, Trainer, TrainingArguments

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--train_subset', type=int, default=None, help='Limit the training dataset size.')
    args = parser.parse_args()

    model_name = "facebook/wav2vec2-base-960h"
    # processor = Wav2Vec2Processor.from_pretrained(model_name)
    wav2vec2_model = Wav2Vec2Model.from_pretrained(model_name)
    model = AudioClassifier(wav2vec2_model, num_classes=30)

    train_dataset = SpeechCommandsDataset(root_dir='data', subset='train')

    if args.train_subset is not None:
        train_dataset = Subset(train_dataset, indices=list(range(args.train_subset)))

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=32,
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=speech_commands_collator,
    )

    trainer.train()


if __name__ == "__main__":
    main()
