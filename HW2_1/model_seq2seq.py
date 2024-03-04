import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import os

# needs two args
if len(sys.argv) == 3:
    feat_dir = sys.argv[1]
    output_file = sys.argv[2]

    print(f"Argument 1: {feat_dir}, Argument 2: {output_file}")
else:
    print("Please pass in two arguments.")
    sys.exit(1)



# loading jsons
def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# clean and create tokens
def clean_caption(caption):
    tokens = caption.lower().split()
    return tokens

# build vocab
class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<BOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold
        
    def get_vocab_size(self):
        return len(self.stoi)

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4

        for sentence in sentence_list:
            for word in sentence:
                frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def save_vocabulary(self, file_path):
        with open(file_path, 'w') as file:
            json.dump({'itos': self.itos, 'stoi': self.stoi}, file, indent=4)

    def numericalize(self, text):
        tokenized_text = clean_caption(text)
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]

def load_vocabulary(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    vocab = Vocabulary(freq_threshold=0)
    vocab.itos = {int(k): v for k, v in data['itos'].items()}
    vocab.stoi = data['stoi']
    return vocab

class VideoCaptionDataset(Dataset):
    def __init__(self, features_dir, annotations, vocab, max_length, transform=None):
        self.features_dir = features_dir
        self.annotations = annotations
        self.vocab = vocab
        self.max_length = max_length
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        caption_data = self.annotations[index]
        caption_list = caption_data['caption']
        img_id = caption_data['id'] 
        img_path = f"{self.features_dir}/{img_id}.npy"
        video_features = np.load(img_path)
        if index == 0:
            print(f"Example video features shape: {video_features.shape}")
        
        #picks a random caption
        caption = np.random.choice(caption_list)
        numericalized_caption = [self.vocab.stoi["<BOS>"]] + self.vocab.numericalize(caption) + [self.vocab.stoi["<EOS>"]]
        padded_caption = numericalized_caption + [self.vocab.stoi["<PAD>"]] * (self.max_length - len(numericalized_caption))
        
        if self.transform:
            video_features = self.transform(video_features)

        return torch.tensor(video_features, dtype=torch.float), torch.tensor(padded_caption, dtype=torch.long)

#generates a caption
def generate_caption(model, video_features, vocab, max_length):
    #eval mode
    model.eval()

    with torch.no_grad():
        input_ids = torch.tensor([vocab.stoi["<BOS>"]], device=video_features.device)
        
        generated_caption_indices = []
        
        for _ in range(max_length):
            outputs = model(video_features, input_ids)
            next_word_id = outputs[0, -1].argmax(0)
            generated_caption_indices.append(next_word_id.item())
            input_ids = torch.cat([input_ids, next_word_id.unsqueeze(0)])
            
            # break if eos is generated
            if next_word_id == vocab.stoi["<EOS>"]:
                break
        
        #convert words
        generated_caption = ' '.join([vocab.itos[idx] for idx in generated_caption_indices if idx in vocab.itos])
        
    return generated_caption


# training files
#features_dir = './MLDS_hw2_1_data/training_data/feat'
#training_data = load_json('./MLDS_hw2_1_data/training_label.json')

# flatten for vocab
#all_captions = [caption for item in training_data for caption in item['caption']]

# calling clean caption
#tokenized_captions = [clean_caption(caption) for caption in all_captions]

# build vocabulary
#vocab = Vocabulary(freq_threshold=3)
#vocab.build_vocabulary(tokenized_captions)
#vocab.save_vocabulary('vocabulary.json')
vocab = load_vocabulary('vocabulary.json')
#caption_example = training_data[0]['caption'][0]
#tokens_example = clean_caption(caption_example)
#numericalized_example = vocab.numericalize(caption_example)


# max length for all padding
#max_caption_length = max(len(clean_caption(c)) for c in all_captions) + 2
max_caption_length = 42

# create dataset
#dataset = VideoCaptionDataset(features_dir, training_data, vocab, max_caption_length)

# dataloader
#data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
data_loader = []

class S2VTModel(nn.Module):
    def __init__(self, feature_dim, hidden_dim, vocab_size, max_seq_length=41):
        super(S2VTModel, self).__init__()
        self.encoder_lstm = nn.LSTM(feature_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.5)
        
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.5)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.max_seq_length = max_seq_length

    def forward(self, video_features, captions=None):
        # encoder
        _, (hidden, _) = self.encoder_lstm(video_features)

        # decode
        outputs = []
        step_input = torch.zeros(video_features.size(0), 1, hidden.size(2)).to(video_features.device)
        for _ in range(self.max_seq_length):
            output, (hidden, _) = self.decoder_lstm(step_input, (hidden, hidden))
            step_input = output
            output = self.fc(output.squeeze(1))
            outputs.append(output)
        outputs = torch.stack(outputs, 1)

        return outputs
    
#vocab_size = vocab.get_vocab_size()
test_vocab_size = vocab.get_vocab_size()
# predetermined
feature_dim = 4096
hidden_dim = 500

model = S2VTModel(feature_dim, hidden_dim, test_vocab_size)

#use cuda if avail
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])
#scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

num_epochs = 30
def train():
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (video_features, captions) in enumerate(data_loader):
            video_features, captions = video_features.to(device), captions.to(device)
        
            outputs = model(video_features, captions[:, :-1])
            loss = criterion(outputs.view(-1, outputs.size(-1)), captions[:, 1:].reshape(-1))
            
            # see into epochs and progress
            if batch_idx == 0 and (epoch % 5 == 0):
                print("Epoch:", epoch + 1)
                logits = outputs[0, :, :] 
                predicted_indices = logits.argmax(dim=1) 
                predicted_tokens = [vocab.itos[idx.item()] for idx in predicted_indices]
                
                ground_truth_indices = captions[0, 1:]
                ground_truth_tokens = [vocab.itos[idx.item()] for idx in ground_truth_indices]
                
                print("Predicted:", predicted_tokens)
                print("Ground Truth:", ground_truth_tokens)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            #scheduler.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(data_loader)}')

        torch.save(model.state_dict(),"s2vt3_model.pth")


class VideoCaptionDatasetTest(Dataset):
    def __init__(self, features_dir):
        self.features_dir = features_dir
        self.file_names = [f for f in os.listdir(features_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        file_name = self.file_names[index]
        file_path = os.path.join(self.features_dir, file_name)
        video_features = np.load(file_path)

        return torch.tensor(video_features, dtype=torch.float), file_name

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = S2VTModel(feature_dim, hidden_dim, test_vocab_size)
model.load_state_dict(torch.load("s2vt3_model.pth", map_location=device))
model = model.to(device)
#eval mode
model.eval()

test_dataset = VideoCaptionDatasetTest(feat_dir)
test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
max_length = 20

with open(output_file, 'w') as file:
    with torch.no_grad():
        for i, (test_video_features, file_name) in enumerate(test_data_loader):
            test_video_features = test_video_features.to(device)
            generated_caption = generate_caption(model, test_video_features, vocab, max_length)
            cleaned_file_name = file_name[0]
            file.write(f"{cleaned_file_name}, {generated_caption}\n")