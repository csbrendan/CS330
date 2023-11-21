from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
from torch.utils.data import DataLoader, Dataset
import torch

# Load the dataset
class MyDebiasedDataset(Dataset):
    def __init__(self, file_path):
        # Load and preprocess the data
        self.examples = []
        self.labels = []  # Add labels to store subgroup information
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                sentence, label = line.strip().split('\t')  # Assuming tab-separated values
                self.examples.append(sentence)
                self.labels.append(label)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx], self.labels[idx]


# Create dataset and dataloader
file_path = '/Users/brendanmurphy/Desktop/CS330 META/PROJECT/project_code/train.txt'
dataset = MyDebiasedDataset(file_path)    #train_small.txt is 50 sentences, train.txt is 3680 this is COUNTERFACTUAL DATA AUGMENTATION
dataloader = DataLoader(dataset, batch_size=8, shuffle=True) #batch_size=32

# Initialize the model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')    # this is gpt2 small 117M params
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', padding_side='left')  


# Freeze all layers first
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last few layers (adjust the number of layers as needed)
for layer in model.transformer.h[-2:]:
    for param in layer.parameters():
        param.requires_grad = True

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training configuration
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
epochs = 1 #3

# Set the pad token to be the same as the EOS token
tokenizer.pad_token = tokenizer.eos_token

# Fine-tuning loop
# During training, calculate subgroup losses
model.train()

# ... [previous code] ...

# Initialize weights for each subgroup (assuming labels are strings like '1', '2', etc.)
subgroup_weights = {label: 1.0 for label in set(dataset.labels)}

for epoch in range(epochs):
    # Reset subgroup losses for the new epoch
    subgroup_losses = {label: [] for label in subgroup_weights}

    for batch, labels in dataloader:
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss

        # Calculate and store loss for each subgroup in the batch
        for label, l in zip(labels, loss):
            subgroup_losses[label].append(l.item())

        # Adjust loss based on subgroup weights
        adjusted_loss = sum(subgroup_weights[label] * l for label, l in zip(labels, loss)) / len(loss)

        # Backpropagation using adjusted loss
        adjusted_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # After each epoch, update the weights based on the average loss of each subgroup
    max_avg_loss = max(sum(losses) / len(losses) for losses in subgroup_losses.values())

    for label, losses in subgroup_losses.items():
        avg_loss = sum(losses) / len(losses)
        subgroup_weights[label] = max_avg_loss / avg_loss  # Weight inversely proportional to average loss

model.save_pretrained('/Users/brendanmurphy/Desktop/CS330 META/PROJECT/project_code/saved_dro_models')
