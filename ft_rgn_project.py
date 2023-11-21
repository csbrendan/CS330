from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
from torch.utils.data import DataLoader, Dataset
import torch


# Load the dataset
class MyDebiasedDataset(Dataset):
    def __init__(self, file_path):
        # Load and preprocess the data
        with open(file_path, 'r', encoding='utf-8') as file:
            self.examples = [line.strip() for line in file.readlines()]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

# Create dataset and dataloader
file_path = '/Users/brendanmurphy/Desktop/CS330 META/PROJECT/project_code/train.txt'
dataset = MyDebiasedDataset(file_path)    #train_small.txt is 50 sentences, train.txt is 3680 this is COUNTERFACTUAL DATA AUGMENTATION
dataloader = DataLoader(dataset, batch_size=8, shuffle=True) #batch_size=32

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', padding_side='left')
model.train()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
epochs = 1
tokenizer.pad_token = tokenizer.eos_token

# Dictionary to store RGN values for each layer
layer_rgn = {}

# Fine-tuning loop
with open('rgn_values.txt', 'w') as file:
    for epoch in range(epochs):
        for batch in dataloader:
            inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            loss.backward()
            # Calculate RGN for each layer
            for name, param in model.named_parameters():
                if 'weight' in name and param.requires_grad:  # Consider only trainable weights
                    grad_norm = torch.norm(param.grad)
                    param_norm = torch.norm(param)
                    rgn = grad_norm / param_norm if param_norm != 0 else 0

                    if name not in layer_rgn:
                        layer_rgn[name] = []
                    layer_rgn[name].append(rgn.item())

                    file.write(f"Layer {name}, RGN: {rgn:.4f}\n")
                    print(f"Layer {name}, RGN: {rgn:.4f}")
            optimizer.step()
            optimizer.zero_grad()
            print(f"Epoch: {epoch}, Loss: {loss.item()}")

# Save the model
model.save_pretrained('/Users/brendanmurphy/Desktop/CS330 META/PROJECT/project_code/saved_rgn_models')


# Calculate the average RGN for each layer
average_rgn = {layer: np.mean(rgns) for layer, rgns in layer_rgn.items()}

# Sort layers by RGN in descending order
sorted_layers = sorted(average_rgn.items(), key=lambda x: x[1], reverse=True)

# Select top 10-20% of layers
percentage = 0.15  # for 10%, adjust as needed
num_layers_to_select = int(len(sorted_layers) * percentage)
layers_to_finetune = [layer for layer, _ in sorted_layers[:num_layers_to_select]]

print("Layers to fine-tune based on top RGN values:", layers_to_finetune)

layers_to_ft = ['transformer.h.0.ln_1.weight', 'transformer.h.1.ln_2.weight', 'transformer.h.2.ln_2.weight', 'transformer.h.1.ln_1.weight', 'transformer.h.2.ln_1.weight', 'transformer.h.0.ln_2.weight', 'transformer.h.6.ln_2.weight', 'transformer.h.4.ln_2.weight', 'transformer.h.5.ln_2.weight', 'transformer.h.7.ln_2.weight', 'transformer.h.8.ln_2.weight']
#['transformer.h.0.ln_1.weight', 'transformer.h.1.ln_2.weight', 'transformer.h.2.ln_2.weight', 'transformer.h.1.ln_1.weight', 'transformer.h.2.ln_1.weight', 'transformer.h.0.ln_2.weight', 'transformer.h.6.ln_2.weight', 'transformer.h.4.ln_2.weight', 'transformer.h.5.ln_2.weight', 'transformer.h.7.ln_2.weight', 'transformer.h.8.ln_2.weight']

#THRESHOLD METHOD
# Calculate average RGN for each layer
#average_rgn = {k: np.mean(v) for k, v in layer_rgn.items()}
# Determine a threshold for RGN (this is a sample threshold, adjust based on your analysis)
#rgn_threshold = 0.01  # Example threshold
# Identify layers exceeding the threshold
#layers_to_finetune = [layer for layer, avg_rgn in average_rgn.items() if avg_rgn > rgn_threshold]
#print("Layers to fine-tune based on RGN threshold:", layers_to_finetune)






