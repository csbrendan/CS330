from typing import List, Tuple, Iterable
import argparse
import torch
import transformers
import torch.nn as nn

try:
    import utils
except ModuleNotFoundError:
    from . import utils
import copy
import numpy as np
import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import itertools

try:
    from icl_project import get_icl_prompts, do_sample, get_performance_metric
except ModuleNotFoundError:
    from .icl_project import get_icl_prompts, do_sample, get_performance_metric
import tqdm
import random


#python ft.py --task ft --model med --mode first,last,middle, --dataset xsum,babi --k 0,1,8,128
#python ft.py --task ft --device cuda --model med --mode lora4,lora16 --dataset xsum,babi --k 0,1,8,128 

#debug mode


parser = argparse.ArgumentParser()
parser.add_argument("--task", default="ft")
parser.add_argument("--model", default="med")
parser.add_argument("--dataset", default="crows")
parser.add_argument("--k", default="24")
parser.add_argument("--mode", default="middle")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--repeats", default=1, type=int)
parser.add_argument("--device", default="cpu")
parser.add_argument("--plot_name", default="plot.png")
args = parser.parse_args()

'''
parser = argparse.ArgumentParser()
parser.add_argument("--task")
parser.add_argument("--model")
parser.add_argument("--dataset")
parser.add_argument("--k")
parser.add_argument("--mode", default="all")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--repeats", default=1, type=int)
parser.add_argument("--device", default="cuda")
parser.add_argument("--plot_name", default="plot.png")
args = parser.parse_args()
'''


if os.environ.get("FORCE_DEVICE", False):
    DEVICE = torch.device(os.environ["FORCE_DEVICE"])
else:
    DEVICE = torch.device(args.device)

print("Fine-tuning using device: ", DEVICE)


class LoRALayerWrapper(nn.Module):
    def __init__(self, base_module: nn.Module, lora_rank: int):
        super().__init__()

        self.base_module = base_module

        ###
        ### Set up your LoRA-augmented layer here.
        ### You should initialize your parameters so that the residual matrix AB^T is zero,
        ###     but be careful how you do this (i.e., make sure you eventually get
        ###     non-zero gradients to both matrices during fine-tuning)!
        ### For randomly initializing the parameters, use torch.randn.
        ### Note: you should use nn.Parameter to wrap your parameters so that they are registered as
        ### learnable.
        ### Initialization hint: what do the gradients look like after 1 and 2 steps of fine-tuning
        ###     if you initialize both A and B to zero? What about if just one is zero?
        ###

        ## YOUR CODE HERE, complete for Q2.2b
        #assert False, "Complete this for Q2.2b"

        # monday 10/23 lecture, cannot init both A and B to be zero otw no grads get in.
        # so init one of them to 0 and other to gaussian noise. this way there product is 0, but you are init with frozen weights.
        dim1, dim2 = base_module.weight.shape

        #self.lora_A = nn.Parameter(torch.randn(dim1, lora_rank))
        #self.lora_B = nn.Parameter(torch.zeros(lora_rank, dim2))
        self.lora_A = nn.Parameter(torch.randn(dim1, lora_rank))   #we will do A @ B.T in fwd pass
        self.lora_B = nn.Parameter(torch.zeros(dim2, lora_rank))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_module(x)  # The output of the pre-trained module.
        #base_out.shape is [1,20,3072]
        ### Perform the forward pass of your LoRA-augmented layer here.
        ### Note: you don't need to ever explicitly construct the matrix AB^T.
        ### Hint: matrix multiplication is associative.

        ## YOUR CODE HERE, complete for Q2.2b
        #assert False, "Complete this for Q2.2b"
        '''
        Finish the LoRALayerWrapper in ft.py, which wraps a pre-trained linear layer with LoRA parameters. 
        You can extract the shape of the pre-trained weight matrix from the base module.weight.shape tuple. 
        You dont need to worry about biases here, just the low-rank weight matrix residual.
        '''

        LoRA_matrix_residual = x @ self.lora_A @ self.lora_B.T  
        
        return base_out + LoRA_matrix_residual     

    

def parameters_to_fine_tune(model: nn.Module, mode: str) -> Iterable[nn.Parameter]:
    """
    Select the parameters in `model` that should be fine-tuned in mode `mode`.

    For every mode except "all", the model is going to be GPT-2 (transformers.GPT2LMHeadModel).
    We encourage you to print the model architecture (e.g. by placing a PDB breakpoint and doing
    `print(model)`), and identifying where the model layers are located.

    Note: this function only returns the list of parameters to fine tune. It doesn't change the
    `requires_grad` component of any parameters, and you should not touch that in this assignment!

    Args:
      model: the model we're fine-tuning
      mode: the fine-tuning mode we're using; may be 'all', 'last', 'first',
        'middle', or 'loraN' (where N is an integer)

    Returns:
      A list of nn.Parameters of `model` that should be fine-tuned in the given
        fine-tuning mode.
    """

    '''
    select the correct subset of parameters for each version listed above in parameters to fine tune(). 
    Keep in mind you should be returning an iterable of nn.Parameter here, not nn.Module.
    Hint: to understand how to get parameters for specific layers, we encourage you to print the model architecture. 
    You can place a Python debugging breakpoint in the code with import pdb; pdb.set trace(), 
    and can run the command in subpart 3 to get the function to run with the right parameters.
    '''

    # print model architecture
    #import pdb 
    #pdb.set_trace()

    #parameters_to_fine_tune: List[nn.Parameter] = None
    parameters_to_fine_tune: List[nn.Parameter] = []
    if mode == "all":
        # Every learnable parameter from `model` should be fine-tuned.
        # Complete this for Q0.1
        #assert False, "Complete this for Q0.1"
        '''
        Implement the logic for fine-tuning, including selecting the params that will be FT'd 
        '''
        parameters_to_fine_tune = list(model.parameters())

    elif mode == "last":
        # Only fine tune the last 2 transformer blocks
        # Complete this for Q2.1
        # assert False, "Complete this for Q2.1"
        num_tx_blocks = len(model.transformer.h) 
        begin = num_tx_blocks - 2
        end = num_tx_blocks
        for block in range(begin, end):
            parameters_to_fine_tune.extend(model.transformer.h[block].parameters())

    elif mode == "first":
        # Only fine tune the first 2 transformer blocks
        # Complete this for Q2.1
        # assert False, "Complete this for Q2.1"
        begin = 0
        end = 2
        for block in range(begin, end):
            parameters_to_fine_tune.extend(model.transformer.h[block].parameters())
        
    elif mode == "middle":
        # Only fine tune middle 2 transformer blocks
        # Complete this for Q2.1
        #assert False, "Complete this for Q2.1"
        num_tx_blocks = len(model.transformer.h) 

        if num_tx_blocks % 2 == 0: # even # blocks case
            begin = num_tx_blocks // 2 - 1
            end = num_tx_blocks // 2 + 1
        else:    #odd blocks case
            begin = num_tx_blocks // 2 - 1
            end = begin + 2 

        for block in range(begin, end):
            parameters_to_fine_tune.extend(model.transformer.h[block].parameters())

    elif mode.startswith("lora"):
        # Only fine tune the rank decomposition matrices A and B from the LoRA layers.
        #Hint: consider using the .modules() fn from nn.Module and check for modules that are instances of LoRALayerWrapper
        # Complete this for Q2.2c
        #assert False, "Complete this for Q2.2c"
        
        #gpt2 ft
        #for m in model.transformer.h:
        #    m.mlp.c_fc = LoRALayerWrapper(m.mlp.c_fc, int(mode[4:]))
        #    m.mlp.c_proj = LoRALayerWrapper(m.mlp.c_proj, int(mode[4:]))
        #    m.attn.c_attn = LoRALayerWrapper(m.attn.c_attn, int(mode[4:]))
        #parameters_to_fine_tune: List[nn.Parameter] = []
        for module in model.modules():
            if isinstance(module, LoRALayerWrapper):
                #parameters_to_fine_tune += [module.lora_A, module.lora_B]
                parameters_to_fine_tune.append(module.lora_A)
                parameters_to_fine_tune.append(module.lora_B)

    else:
        raise ValueError(f"Unrecognized fine-tuning mode {mode}")

    return parameters_to_fine_tune


def get_loss(unnormalized_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes the cross-entropy loss for either sequence classification or generation.

    For generation, you'll need to deal with the fact that different sequences within
      the batch are different lengths, and the targets tensor includes some mask
      values (-100). The average loss is the *average loss over all non-masked timesteps*.
      You'll also need to handle the fact that the prediction for what token t will be is
      made after seeing only t - 1 tokens; that is, there is an off-by-one shift needed
      between the logits and targets.

    Args:
      unnormalized_logits: a 2D [batch_size, n_classes] (for classification) or 3D
        [batch_size, sequence_length, vocab_size] (for generation) tensor
        of *UNNORMALIZED* logits
      targets: a 1D [batch_size] (for classification) or 2D [batch_size, sequence_length]
        (for generation) tensor of target indices. For the generation case, may contain
        -100 in some positions, meaning that the loss for this timestep should be ignored.

    Returns:
      A zero-dim tensor (scalar) representing the average cross-entropy loss over all batch
        elements (and sequence timesteps, if applicable)
    """
    import torch.nn.functional as F

    loss: torch.Tensor = None
    if unnormalized_logits.dim() == 2:
        # This is the classification case.
        # Complete this for Q0.1
        #assert False, "Complete this for Q0.1"
        '''
        compute the loss in get_loss()  only under if logits.dim() == 2: 
        '''
        loss = F.cross_entropy(unnormalized_logits, targets)   

    elif unnormalized_logits.dim() == 3:
    
        # This is the generation case.
        # Remember that the target tensor may contain -100 values, which should be masked out
        # and that an off-by-one shift is needed between the logits and targets.
        # Complete this for Q2.2d

        # chop off first
        shifty_targets = targets[..., 1:].contiguous()  

        # shift logits back 1 time step
        shifty_logits = unnormalized_logits[..., :-1, :].contiguous()   

        # we dont want to calc loss for padded tokens, so mask out and ensure we have valid token for calc
        mask = (shifty_targets != -100)  
        if mask.sum() == 0:
            raise ValueError("no valid tokens found for loss calc... check target sequences.")

        masked_logits = shifty_logits[mask]
        masked_targets = shifty_targets[mask]

        # calc loss on non padded tokeys
        masked_out_loss = F.cross_entropy(masked_logits, masked_targets, reduction='sum')

        # calc for avg over sequence
        loss = masked_out_loss / mask.sum()

    else:
        raise ValueError(
            f"Logits should either be 2-dim (for classification) or 3-dim (for generation); got {unnormalized_logits.dim()}"
        )

    assert (
        loss is not None and loss.dim() == 0
    ), "Loss should be a scalar tensor. It should be the mean loss over the batch"
    return loss


def get_acc(unnormalized_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes the exact match accuracy for either sequence classification or generation. i.e.,
      the fraction of predictions for which the most likely class/token equals the target.

    For generation, you'll need to deal with the fact that different sequences within
      the batch are different lengths, and the targets tensor includes some mask
      values (-100). The average accuracy is the *average accuracy over all non-masked timesteps*.
      You'll also need to handle the fact that the prediction for what token t will be is
      made after seeing only t - 1 tokens; that is, there is an off-by-one shift needed
      between the logits and targets.

    Args:
      unnormalized_logits: a 2D [batch_size, n_classes] (for classification) or 3D
        [batch_size, sequence_length, vocab_size] (for generation) tensor of logits
      targets: a 1D [batch_size] (for classification) or 2D [batch_size, sequence_length]
        (for generation) tensor of target indices. For the generation case, may contain
        -100 in some positions, meaning that the loss for this timestep should be ignored.

    Returns:
      A *scalar* representing the average exact-match accuracy over all non-masked batch
        elements (and sequence timesteps, if applicable)
    """
    accuracy: torch.Tensor = None
    if unnormalized_logits.dim() == 2:
        # This is the classification case.
        # Complete this for Q0.1
        #assert False, "Complete this for Q0.1"
    
        predictions = unnormalized_logits.argmax(dim=1)
        accuracy = torch.mean(torch.eq(predictions, targets).float())

    elif unnormalized_logits.dim() == 3:
        # This is the generation case.
        # Complete this for Q2.2d
        # assert False, "Complete this for Q2.2d"
        '''
            For generation, you'll need to deal with the fact that different sequences within
            the batch are different lengths, and the targets tensor includes some mask
            values (-100). The average accuracy is the *average accuracy over all non-masked timesteps*.
            You'll also need to handle the fact that the prediction for what token t will be is
            made after seeing only t - 1 tokens; that is, there is an off-by-one shift needed
            between the logits and targets.        

            We are predicting the next token for each time step, so the logits for time step t should be matched with 
            the word at time step t+1 in the ground truth sequence. 
            That is why you need to perform a shifting operation while calculating the accuracy. - Ansh
        '''
        # shift logits / targets for acc calc
        shifted_logits = unnormalized_logits[..., :-1, :].contiguous()
        shifted_targets = targets[..., 1:].contiguous()

        # mask out -100 for acc calc & ensure we have valid tokens
        mask = (shifted_targets != -100)  
        if mask.sum() == 0:
            raise ValueError("no valid tokens found for get_acc() calc... check target sequences...")

        # calc preds (retrieve most likely prediction from last dim)
        predictions = torch.argmax(shifted_logits, -1)

        # calc acc (create a bool tensor of correct preds)
        #first index predictions and targets using the mask, and then compute acc in normal way...
        masked_predictions = predictions[mask]
        masked_shifted_targets = shifted_targets[mask]
        correct = (masked_predictions == masked_shifted_targets)
        #if any(correct):
        #    print("Found a correct prediction! :) " + str(correct))
        #else:
        #    print("didnt find any correct predictions")

        accuracy = correct.float().mean() 
        #print("my accuracy " + str(accuracy.item()))
        
        # decode predictions
        #for batch_idx in range(predictions.size(0)):  
        #    pred_ids = predictions[batch_idx].detach().cpu().numpy() 
        #    pred_tokens = pred_ids[mask[batch_idx].cpu().numpy()]  
        #    decoded_prediction = tokenizer.decode(pred_tokens.tolist())  
        #    print(f"Decoded prediction: {decoded_prediction}")
            
    else:
        raise ValueError(
            f"Logits should either be 2-dim (for classification) or 3-dim (for generation); got {unnormalized_logits.dim()}"
        )
    # accuracy is tensor(0.)
    assert (
        accuracy is not None and accuracy.dim() == 0
    ), "Accuracy should be a scalar tensor. It should be the mean accuracy over the batch"
    return accuracy.item()


def ft_bert(model, tok, x, y, mode, batch_size=8):
    model = copy.deepcopy(model)

    if mode.startswith("lora"):
        for m in model.transformer.h:
            m.mlp.c_fc = LoRALayerWrapper(m.mlp.c_fc, int(mode[4:]))
            m.mlp.c_proj = LoRALayerWrapper(m.mlp.c_proj, int(mode[4:]))

    model.to(DEVICE)

    optimizer = torch.optim.Adam(parameters_to_fine_tune(model, mode), lr=1e-4)
    all_x = tok(
        x, return_tensors="pt", padding=True, truncation=True, max_length=100
    ).to(DEVICE)
    all_y = torch.tensor(y, device=DEVICE)
    pbar = tqdm.tqdm(range(1000))
    for step in pbar:
        batch = np.random.randint(0, len(x), batch_size)
        x_ = tok(
            [x[i] for i in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=100,
        ).to(DEVICE)
        y_ = torch.tensor([y[i] for i in batch], device=DEVICE)
        logits = model(**x_).logits
        loss = get_loss(logits, y_)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if args.debug:
            break

        if step % 10 == 0:
            with torch.inference_mode():
                total_acc = get_acc(model(**all_x).logits, all_y)
            pbar.set_description(f"Fine-tuning acc: {total_acc:.04f}")
            if total_acc > 0.75:
                break
    return model


def tokenize_gpt2_batch(
    tokenizer: transformers.GPT2Tokenizer, x: List[str], y: List[str]
):
    """
    Implement the tokenization step for a batch of examples for GPT-2.

    Args:
        tokenizer: a GPT2Tokenizer that you can call and receive a dictionary of:
          - input_ids: a list (or tensor) of token ids
          - attention_mask: a list (or tensor) of 1s and 0s indicating which tokens
              are padding (if you requested padding and tensors from the tokenizer)
        x: a list of strings, each of which is the input for a single example
        y: a list of strings, each of which is a *target* for a single example

    Returns:
        A dictionary with the following keys:
            - input_ids: a tensor of shape [batch_size, sequence_length]
                containing the token ids
            - attention_mask: a tensor of shape [batch_size, sequence_length]
                containing 1s and 0s indicating which tokens are padding
            - labels: a tensor of shape [batch_size, sequence_length] containing
                the target token ids, with -100 for non-target tokens (i.e., the
                tokens in the input part of each example or padding tokens)
        where sequence_length is determined by the (x, y) pair whose tokenized
        length is the longest in the batch. The other sequences should be padded to
        this length (you can get the tokenizer to handle this padding!).

    Example:
        >>> x = ['Who is the singer for the band Queen?', 'What is the capital of France?']
        >>> y = ['Freddie Mercury', 'Paris']
        >>> tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
        >>> tokenizer_dict = tokenizer([x_ + y_ for x_, y_ in zip(x, y)], return_tensors='pt', padding=True) 
        #[x_ + y_ for x_, y_ in zip(x, y)] -> results in this -> ['Who is the singer for the band Queen? Freddie Mercury', 'What is the capital of France? Paris']

        >>> tokenizer_dict['input_ids']
        tensor([[ 8241,   318,   262, 14015,   329,   262,  4097,  7542,    30, 30847, 11979, 21673],
                [ 2061,   318,   262,  3139,   286,  4881,    30, 40313, 50256, 50256, 50256, 50256]])      #50256 is padding or eos (end of sentence)
        >>> tokenizer_dict['attention_mask']
        tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]])
        >>> tokenizer(x)['input_ids']
        [[8241, 318, 262, 14015, 329, 262, 4097, 7542, 30],
         [2061, 318, 262, 3139, 286, 4881, 30]]
        >>> tokenizer(y)['input_ids']
        [[30847, 11979, 21673],
         [40313]]

        In this case, our labels should look like:
        [[-100, -100, -100, -100, -100, -100, -100, -100,   -100,  30847, 11979, 21673],
         [-100, -100, -100, -100, -100, -100, -100,  40313, -100, -100,  -100,  -100]]
        Note we've replaced padding tokens and the input prefix for each example
            with -100, leaving only the tokens in y.

        Other note: you can add new keys (such as 'labels') to the dictionary
            returned by the tokenizer without creating a new dictionary.
    """
    combined_sequences = None
    # YOUR CODE HERE, complete for Q2.2e
    
    combined_sequences = [x_ + y_ for x_, y_ in zip(x, y)]
    tokenized_dict = tokenizer(combined_sequences, return_tensors='pt', padding=True).to(DEVICE)

    # set all labels to -100, then insert the tokenized labels
    labels = torch.full_like(tokenized_dict['input_ids'], -100)
    input_ids = tokenized_dict['input_ids']
    batch_size, seq_len = input_ids.shape

    #for i in range(batch_size):
    #    y_length = len(y[i])
    #    labels[i, :seq_len - y_length] = -100
    #    labels[i, seq_len - y_length:] = input_ids[i, seq_len - y_length:]
    for i in range(batch_size):
        this_examples_labels_tokens = tokenizer(y[i], return_tensors='pt').input_ids.to(DEVICE).squeeze(0)
        start_pos = tokenizer(x[i])['input_ids']
        start_index = len(start_pos)
        labels[i, start_index :start_index + this_examples_labels_tokens.size(0)] = this_examples_labels_tokens

    # replace padding 50256
    pad_token_id = tokenizer.pad_token_id
    labels[labels == pad_token_id] = -100

    tokenized_dict['labels'] = labels

    # decode sequences and their labels
    for i, input_ids in enumerate(tokenized_dict['input_ids']):
        print(f"Decoded sequence {i+1}: {tokenizer.decode(input_ids)}")    
        print(f"Decoded labels {i+1}: {tokenizer.decode(labels[i][labels[i] != -100] )}") #, skip_special_tokens=True)}")


    return tokenized_dict


def add_prefixes(
    x: List[str], y: List[str], dataset: str
) -> Tuple[List[str], List[str]]:
    input_prefix = "" if utils.is_qa_dataset(dataset) else ""
    label_prefix = " In the" if utils.is_qa_dataset(dataset) else " TL;DR:"
    label_suffix = "." if utils.is_qa_dataset(dataset) else ""

    x = [input_prefix + x_.replace("\n", " ") + label_prefix for x_ in x]
    y = [" " + y_.replace("\n", " ") + label_suffix for y_ in y]

    return x, y


def ft_gpt2(model, tok, x, y, mode, dataset, batch_size=8, grad_accum=8):
    x, y = add_prefixes(x, y, dataset)

    #print("x from initial ft_gpt2() " + ', '.join(x))
    #print("y from initial ft_gpt2() " + ', '.join(y))

    model = copy.deepcopy(model)

    if mode.startswith("lora"):
        for m in model.transformer.h:
            m.mlp.c_fc = LoRALayerWrapper(m.mlp.c_fc, int(mode[4:]))
            m.mlp.c_proj = LoRALayerWrapper(m.mlp.c_proj, int(mode[4:]))
            m.attn.c_attn = LoRALayerWrapper(m.attn.c_attn, int(mode[4:]))

    model.to(DEVICE)

    optimizer = torch.optim.Adam(parameters_to_fine_tune(model, mode), lr=2e-5)
    all_both = tokenize_gpt2_batch(tok, x, y)
    max_n = len(x) * 10
    pbar = tqdm.tqdm(range(max_n))
    idxs = []
    for step in pbar:
        model.train()

        if len(idxs) < batch_size // grad_accum: #only enter this condition the first iter, then idxs i len 7
            idxs = list(range(len(x)))   # idxs is 0-7, len 8
            random.shuffle(idxs)
        batch_idxs = idxs[: batch_size // grad_accum]   # selects from beginning UP TO (not including) the index: floor division 8/8
        idxs = idxs[batch_size // grad_accum :] # this selects the REST of idxs: from result of div to the end. if idxs[1,4,5,6..]  batch_idxs['1'] since 8/8 = 1. 

        # Outline:
        # YOUR CODE HERE, complete for Q2.2f
        # 1. Sample a random minibatch of examples of size batch_size // grad_accum using the batch_idxs variable
        mini_batch_x = [x[i] for i in batch_idxs]  
        mini_batch_y = [y[i] for i in batch_idxs]  

        # 2. Tokenize the batch using the tokenize_gpt2_batch function you implemented
        tokenized_batch = tokenize_gpt2_batch(tok, mini_batch_x, mini_batch_y)
        #tokenized_batch = {k: v.to(DEVICE) for k, v in tokenized_batch.items()} # need .to(DEVICE) here ?

        # 3. Run the model on the batch, get the logits, and compute the loss using the get_loss function you implemented
        # *NOTE 1* Pass `use_cache=False` when you call model() to avoid a huggingface warning
        # *NOTE 2* You MUST compute the loss using your get_loss function applied to the model_output.logits.
        # Don't use the loss attribute of the model output for training (you will not get credit for this).
        model_output = model(**tokenized_batch, use_cache=False)  # use_cache=False avoids warning

        # if mem issue try, but temp need full model output for loss comparison: logits = model(**tokenized_batch, use_cache=False).logits 
        loss = get_loss(model_output.logits, tokenized_batch['labels'])  

        # However, you can use the loss attribute of the model output to test your get_loss function (they should match).
        #if hasattr(model_output, 'loss'):
        #    builtin_loss = model_output.loss
        #    print(f"my custom loss: {loss.item()}, model built-in loss: {builtin_loss.item()}")
        #    print(f"my scaled custom loss: {loss.item()/grad_accum}, model built-in loss: {builtin_loss.item()}")
        # 4. Backpropagate the loss (divided by the grad_accum parameter)
        loss = loss / grad_accum #average the loss over the training examples
        loss.backward()

        # 5. Take a step of the optimizer and zero the model gradients ***only every grad_accum steps***
        # Be careful that you don't take a step after the very first backward pass (i.e., when step == 0)
        # Note: the ** operator will unpack a dictionary into keyword arguments to a function (such as your model)
        if (step + 1) % grad_accum == 0 and step != 0:  # take step only after grad_accum steps and not at step 0
            optimizer.step()
            optimizer.zero_grad()

        # END YOUR CODE

        #code below is for occasional (5*8=40 steps) evaluation of performance during training ... 

        if step % (grad_accum * 5) == 0:
            with torch.inference_mode():
                model.eval()
                accs = []
                for idx in range(len(list(all_both.values())[0])):
                    d = {k: v[idx : idx + 1] for k, v in all_both.items()}
                    acc = get_acc(model(**d).logits, d["labels"])
                    accs.append(acc)
                total_acc = sum(accs) / len(accs)
                pbar.set_description(f"Fine-tuning acc: {total_acc:.04f}")

            if total_acc >= utils.early_stop_thresold(dataset):
                print("Early stopping!")
                break
    return model


def eval(model, tok, val_data):
    x = tok(
        val_data["x"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=100,
    ).to(DEVICE)
    y = torch.tensor(val_data["y"], device=DEVICE)
    with torch.inference_mode():
        logits = model(**x).logits
    return get_acc(logits, y)


def run_ft(
    models: List[str],
    datasets: List[str],
    ks: List[int],
    modes: List[str],
    n_val: int = 125,
):
    results = {}
    for dataset in datasets:
        utils.fix_random_seeds()
        if args.debug:
            n_val = 1
        train, val = utils.get_dataset(dataset, max(ks), n_val=n_val)
        for model_name, mode in itertools.product(models, modes):
            utils.fix_random_seeds()
            if dataset == "amazon":
                model, tokenizer = utils.get_model_and_tokenizer(
                    model_name,
                    transformers.AutoModelForSequenceClassification,
                    num_labels=5,
                )
            else:
                model, tokenizer = utils.get_model_and_tokenizer(
                    model_name, transformers.AutoModelForCausalLM
                )
            stop_tokens = utils.stop_tokens(tokenizer)

            for k in ks:
                print(
                    f"Fine-tuning {model_name} on {dataset} with k={k} and mode={mode}"
                )
                utils.fix_random_seeds()
                for repeat in range(args.repeats):
                    if repeat > 0:
                        print(f"Beginning repeat #{repeat}")
                    if dataset == "amazon":
                        fine_tuned = ft_bert(
                            model,
                            tokenizer,
                            train["x"][: k * 5],
                            train["y"][: k * 5],
                            mode,
                        )
                        val_acc = eval(fine_tuned, tokenizer, val)
                        results["_".join([model_name, dataset, str(k), mode])] = val_acc
                    else:
                        if k > 0:
                            fine_tuned = ft_gpt2(
                                model,
                                tokenizer,
                                train["x"][:k],
                                train["simple_y"][:k],
                                mode,
                                dataset,
                            )
                        else:
                            fine_tuned = copy.deepcopy(model)
                            fine_tuned.to(DEVICE)

                        fine_tuned.eval()
                        targets = []
                        predictions = []
                        pbar = tqdm.tqdm(list(range(min(n_val, len(val["x"])))))

                        for row in pbar:
                            test_input = val["x"][row]
                            targets.append(val["y"][row])
                            max_tokens = utils.max_sampled_tokens_for_dataset(dataset)
                            prompt_mode = (
                                "babi" if utils.is_qa_dataset(dataset) else "tldr"
                            )
                            prompt = get_icl_prompts(
                                [], [], test_input, prompt_mode=prompt_mode
                            )
                            input_ids = tokenizer(
                                prompt, return_tensors="pt"
                            ).input_ids.to(DEVICE)
                            sampled_tokens = do_sample(
                                fine_tuned, input_ids, stop_tokens, max_tokens
                            )
                            decoded = tokenizer.decode(sampled_tokens).strip()
                            predictions.append(decoded)
                            metric = get_performance_metric(
                                predictions, targets, utils.metric_for_dataset(dataset)
                            )
                            pbar.set_description(f"Eval: {metric:.04f}")
                        results["_".join([model_name, dataset, str(k), mode])] = metric

                    print(results)
                    question = "ft"
                    if not os.path.exists(f"{utils.RESULTS_DIR}/{question}"):
                        os.makedirs(f"{utils.RESULTS_DIR}/{question}")

                    for k_, v in results.items():
                        print(
                            f"Writing results to: {utils.RESULTS_DIR}/{question}/{k_}.json"
                        )
                        with open(
                            f"{utils.RESULTS_DIR}/{question}/{k_}.json", "w"
                        ) as f:
                            json.dump({"metric": v}, f)
                    results = {}


def plot_ft(models, datasets, ks, modes, output_path: str):
    data = defaultdict(lambda: defaultdict(list))
    question = "ft"

    x_vals = set()
    for dataset in datasets:
        for model, mode in itertools.product(models, modes):
            for k in ks:
                fn = "_".join([model, dataset, str(k), mode])
                id_ = "_".join([model, dataset, mode])
                with open(f"{utils.RESULTS_DIR}/{question}/{fn}.json", "r") as f:
                    score = json.load(f)["metric"]
                    data[id_]["x"].append(k)
                    x_vals.add(k)
                    data[id_]["y"].append(score)

        for k, v in data.items():
            plt.plot(v["x"], v["y"], label=k)

    if max(x_vals) > 4:
        plt.xscale("symlog")
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_ticks(sorted(x_vals))
    plt.legend()
    plt.title(" & ".join(datasets))
    plt.ylabel("/".join([utils.metric_for_dataset(dataset) for dataset in datasets]))
    plt.xlabel("Number of support examples")
    # plt.show()
    plt.savefig(output_path, bbox_inches="tight")

'''
def run():

    ks = [int(k) for k in args.k.split(",")]
    if args.task == "ft":
        run_ft(args.model.split(","), args.dataset.split(","), ks, args.mode.split(","))
    elif args.task == "plot":
        plot_ft(
            args.model.split(","),
            args.dataset.split(","),
            ks,
            args.mode.split(","),
            args.plot_name,
        )


if __name__ == "__main__":
    run()
'''
    
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
dataset = MyDebiasedDataset('train_neutralized.txt')    #train_small.txt is 50 sentences, train.txt is 3680 this is COUNTERFACTUAL DATA AUGMENTATION
dataloader = DataLoader(dataset, batch_size=8, shuffle=True) #batch_size=32

# Initialize the model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')    # this is gpt2 small 117M params
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', padding_side='left')  
#model = GPT2LMHeadModel.from_pretrained('gpt2-medium')   # med is 345M params, gpt2-large is 774M
#tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium', padding_side='left')

# Freeze all layers first
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last few layers (adjust the number of layers as needed)
for layer in model.transformer.h[-2:]:
    for param in layer.parameters():
        param.requires_grad = True

# Unfreeze the middle layers (layers 5 and 6 in gpt2-small)
#for layer_index in [4, 5]:
#    for param in model.transformer.h[layer_index].parameters():
#        param.requires_grad = True


# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training configuration
#optimizer = AdamW(model.parameters(), lr=5e-5)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
epochs = 1 #3

# Set the pad token to be the same as the EOS token
tokenizer.pad_token = tokenizer.eos_token

# Fine-tuning loop
model.train()
for epoch in range(epochs):
    for batch in dataloader:
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
        #inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Epoch: {epoch}, Loss: {loss.item()}")


        # Decode and print the input and predicted output
        input_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        #predictions = model.generate(inputs["input_ids"], max_length=512)
        predictions = model.generate(inputs["input_ids"], max_length=50, pad_token_id=tokenizer.eos_token_id, attention_mask=inputs["attention_mask"])
        predicted_text = tokenizer.decode(predictions[0], skip_special_tokens=True)


        #print(f"Input: {input_text}")
        #print(f"Predicted: {predicted_text}")
        #print(f"Loss: {loss.item()}\n")

# Save the model
model.save_pretrained('saved_models')
# after saving i can load the tokenizer and model
#tokenizer = GPT2Tokenizer.from_pretrained('saved_models')
#model = GPT2LMHeadModel.from_pretrained('saved_models')
