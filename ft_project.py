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
file_path = '/Users/brendanmurphy/Desktop/CS330 META/PROJECT/project_code/train_gender.txt'
dataset = MyDebiasedDataset(file_path)    #train_small.txt is 50 sentences, train.txt is 3680 this is COUNTERFACTUAL DATA AUGMENTATION
dataloader = DataLoader(dataset, batch_size=8, shuffle=True) #batch_size=32

# Initialize the model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')    # this is gpt2 small 117M params
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', padding_side='left')  

'''
# Freeze all layers first
for param in model.parameters():
    param.requires_grad = False

'''
# Unfreeze the last few layers (adjust the number of layers as needed)
#for layer in model.transformer.h[-2:]:
#    for param in layer.parameters():
#        param.requires_grad = True

'''
# RGN: List of layers to fine-tune
#layers_to_finetune = ['transformer.h.0.ln_1.weight', 'transformer.h.1.ln_2.weight', 'transformer.h.2.ln_2.weight', 'transformer.h.1.ln_1.weight', 'transformer.h.2.ln_1.weight', 'transformer.h.0.ln_2.weight', 'transformer.h.6.ln_2.weight', 'transformer.h.4.ln_2.weight', 'transformer.h.5.ln_2.weight', 'transformer.h.7.ln_2.weight', 'transformer.h.8.ln_2.weight']
#gender_layers_to_finetune = ['transformer.h.0.ln_1.weight', 'transformer.h.1.ln_2.weight', 'transformer.h.2.ln_2.weight', 'transformer.h.1.ln_1.weight', 'transformer.h.2.ln_1.weight', 'transformer.h.0.ln_2.weight', 'transformer.h.10.ln_2.weight', 'transformer.h.4.ln_2.weight', 'transformer.h.8.ln_2.weight', 'transformer.h.3.ln_2.weight', 'transformer.h.6.ln_2.weight']
#gender ds2 all ['transformer.h.0.ln_1.weight', 'transformer.h.1.ln_2.weight', 'transformer.h.2.ln_2.weight', 'transformer.h.1.ln_1.weight', 'transformer.h.2.ln_1.weight', 'transformer.h.0.ln_2.weight', 'transformer.h.10.ln_2.weight', 'transformer.h.8.ln_2.weight', 'transformer.h.4.ln_2.weight', 'transformer.h.6.ln_2.weight', 'transformer.h.3.ln_2.weight', 'transformer.h.7.ln_2.weight', 'transformer.h.9.ln_2.weight', 'transformer.h.5.ln_2.weight', 'transformer.h.11.ln_2.weight', 'transformer.h.3.ln_1.weight', 'transformer.wte.weight', 'transformer.h.4.ln_1.weight']

# I will go further and only use those layers from middle to end, leaving the first half layers in tact:
#gender_layers_to_finetune = ['transformer.h.10.ln_2.weight', 'transformer.h.8.ln_2.weight', 'transformer.h.6.ln_2.weight', 'transformer.h.7.ln_2.weight', 'transformer.h.9.ln_2.weight', 'transformer.h.5.ln_2.weight', 'transformer.h.11.ln_2.weight']

# RGN for NM all ['transformer.h.0.ln_1.weight', 'transformer.h.1.ln_2.weight', 'transformer.h.1.ln_1.weight', 'transformer.h.2.ln_2.weight', 'transformer.h.0.ln_2.weight', 'transformer.h.2.ln_1.weight', 'transformer.h.10.ln_2.weight', 'transformer.h.4.ln_2.weight', 'transformer.h.8.ln_2.weight', 'transformer.h.6.ln_2.weight', 'transformer.h.9.ln_2.weight', 'transformer.h.11.ln_2.weight', 'transformer.h.7.ln_2.weight', 'transformer.h.5.ln_2.weight', 'transformer.h.3.ln_2.weight', 'transformer.h.3.ln_1.weight', 'transformer.h.0.attn.c_proj.weight', 'transformer.wte.weight']
gender_layers_to_finetune = ['transformer.h.10.ln_2.weight', 'transformer.h.8.ln_2.weight', 'transformer.h.6.ln_2.weight', 'transformer.h.9.ln_2.weight', 'transformer.h.11.ln_2.weight', 'transformer.h.7.ln_2.weight', 'transformer.h.5.ln_2.weight']

# RGN for GPT2-large (and > height 16)
gender_layers_to_finetune = ['transformer.h.17.attn.c_attn.weight', 'transformer.h.18.attn.c_attn.weight', 'transformer.h.19.attn.c_attn.weight', 'transformer.h.22.attn.c_attn.weight', 'transformer.h.17.attn.c_proj.weight', 
'transformer.h.18.attn.c_proj.weight', 'transformer.h.20.attn.c_attn.weight', 'transformer.h.20.attn.c_proj.weight', 'transformer.h.19.attn.c_proj.weight', 'transformer.h.21.attn.c_attn.weight', 'transformer.h.21.attn.c_proj.weight', 
'transformer.h.23.attn.c_attn.weight', 'transformer.h.24.attn.c_attn.weight', 'transformer.h.35.mlp.c_fc.weight', 'transformer.h.22.attn.c_proj.weight', 'transformer.h.23.attn.c_proj.weight', 'transformer.h.25.attn.c_attn.weight', 'transformer.h.24.attn.c_proj.weight', 'transformer.h.26.attn.c_attn.weight']


# RGN: Unfreeze selected layers
for name, param in model.named_parameters():
    if name in gender_layers_to_finetune:
        param.requires_grad = True
'''

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training configuration
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
epochs = 4

# Set the pad token to be the same as the EOS token
tokenizer.pad_token = tokenizer.eos_token

# Fine-tuning loop
model.train()
for epoch in range(epochs):
    for batch in dataloader:
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        #loss = model(**inputs, labels=inputs["input_ids"]).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Epoch: {epoch}, Loss: {loss.item()}")

        # Decode and print the input and predicted output
        #input_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        #predictions = model.generate(inputs["input_ids"], max_length=50, pad_token_id=tokenizer.eos_token_id, attention_mask=inputs["attention_mask"])
        #predicted_text = tokenizer.decode(predictions[0], skip_special_tokens=True)

# Save the model
model.save_pretrained('/Users/brendanmurphy/Desktop/CS330 META/PROJECT/project_code/saved_gpt2_small_all_layers_4epochs')
