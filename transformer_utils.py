# 1) TOKENIZERS

import re #python module to support functions like match, search, split
import tiktoken #library to implement BPE tokenizer


# 1.1) Forming simple word-level and character-level vocabularies

def make_word_level_vocabulary(raw_text):
    """ Generates a word-level vocabulary (dictionary with keys = words, vals = integers) 
        based on a raw input text (str)."""
    
    #split the text into words including spaces
    splitted = re.split(r'([,.?_!"()\']|-|\s)', raw_text)
    
    #remove spaces
    splitted_nospace = [item.strip() for item in splitted if item.strip()] 
    
    #determine unique "words" using set() method and sort them
    unique_words = sorted(list(set(splitted_nospace))) 
    
    #create a dictionary where each unique word is assigned to an integer
    vocab = {token:integer for integer,token in enumerate(unique_words)} 
    
    # append one more integer for the "unknown" key
    vocab["<|unk|>"] = len(unique_words) 
    
    return vocab

def make_char_level_vocabulary(raw_text):
    """ Generates a character-level vocabulary (dictionary with keys = characters, vals = integers) 
        based on a raw input text (str)."""
    
    #list unique chars in a sorted fashion 
    unique_chars = sorted(list(set(raw_text)))
    
    #create a dictionary where each unique word is assigned to an integer
    vocab = {token:integer for integer,token in enumerate(unique_chars)} 
    
    # append one more integer for the "unknown" key
    vocab["<|unk|>"] = len(vocab) 
    
    return vocab

# 1.2) Word-level, character-level, and BPE tokenizers

class WordTokenizer: 
    """Encodes words to integers and decodes integers to words based 
    on a simple world-level vocabulary"""
    
    def __init__(self, vocab):
        #vocab
        self.str_to_int = vocab
        
        #reverse vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        """text --> integers"""
        
        #split the text into words including spaces
        splitted = re.split(r'([,.?_!"()\']|--|\s)', text)
        
        #remove spaces
        splitted_nospace = [item.strip() for item in splitted if item.strip()]
        
        #replace unknown words by <|unk|> tokens
        splitted_nospace_w_unk = [item if item in self.str_to_int else "<|unk|>" for item in splitted_nospace] 

        #assign an int to each word using the vocabulary
        ids = [self.str_to_int[s] for s in  splitted_nospace_w_unk]
        
        return ids
        
    def decode(self, ids):
        """integers --> text"""
        
        #convert ids to str and join them with a space in-between
        text = ' '.join([self.int_to_str[i] for i in ids])

        #remove spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        
        return text


class CharTokenizer:
    """encode chars to integers and decode integers to chars based 
    on a simple char-level vocabulary"""
    
    def __init__(self, vocab):       
        #vocab
        self.str_to_int = vocab
        
        #reverse vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
        
    def encode(self, text):
        #replace unknown chars by <|unk|> tokens
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in text] 
                        
        #assign an int to each char using the vocabulary
        ids = [self.str_to_int[s] for s in preprocessed]
        
        return ids
        
    def decode(self, ids):
        #convert chars to ids and merge (space has a specific id = 1)
        text = "".join([self.int_to_str[i] for i in ids])
        
        return text


class BPETokenizer():
    """Sub-word tokenizer used in GPT2&3. BPE can represent words that were not 
    in the training data by combining known sub-words."""
    
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding('gpt2')
    
    def encode(self, text):
        return self.tokenizer.encode(text)
        
    def decode(self, ids):
        return self.tokenizer.decode(ids)


# 2) DATASETS

from torch.utils.data import Dataset, DataLoader
import pandas as pd


def print_two_batches(dataloader):
    "for a quick check on dataloaders"
    i = 0
    for (x, y) in dataloader:
        print("x:\n", x)
        print("y:\n", y, "\n")
        i+=1
        if i>1:
            break


# 2.1) Text generation dataset

class TextGenDataset(Dataset):     
    """Takes a training text and creates input/target ID pairs, which are accessible by index.
    Created dataset object can be passed into a DataLoader for batching and shuffling in PyTorch"""
    
    def __init__(self, text_data, tokenizer, Nmax, stride=1):

        self.input_ids = []
        self.target_ids = []
        
        # Tokenize the entire text
        tokenized_text = tokenizer.encode(text_data)

        # Create input/target ID pairs by slicing the text into overlapping sequences of length Nmax
        # stride: step size between consecutive starting indices
        for i in range(0, len(tokenized_text) - Nmax, stride):
            input_seq = tokenized_text[i:i + Nmax]
            target_seq = tokenized_text[i + 1: i + Nmax + 1]
            self.input_ids.append(torch.tensor(input_seq))
            self.target_ids.append(torch.tensor(target_seq))
        
    def __len__(self):
        #__len__ method: returns the total number of items in the dataset
        return len(self.input_ids)

    def __getitem__(self, idx): 
        # __getitem__ method: retrieves an individual item from the dataset by index
        return self.input_ids[idx], self.target_ids[idx]


# 2.2) Text classification dataset

class TextClassDataset(Dataset):
    
    def __init__(self, text_csvdata, tokenizer, Nmax, pad_token_id=50256):
        """Tokenize the texts to classify and truncate or pad them to the predefined max length"""
        
        self.data = pd.read_csv(text_csvdata)
        self.headers = self.data.columns.tolist()      

        # Pre-tokenize texts
        encoded_text = [tokenizer.encode(text) for text in self.data[self.headers[0]]]
        
        # Truncate sequences if they are longer than Nmax
        truncated_text = [text[:Nmax]for text in encoded_text]

        # Pad sequences to Nmax
        # Default id of 50256 corresponds to '<|endoftext|>' of GPT2 BPE tokenizer
        self.final_text = [text + [pad_token_id] * (Nmax - len(text)) for text in truncated_text]

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        encoded = self.final_text[idx]
        label = self.data.iloc[idx][self.headers[1]]
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long)


# 2.3) Image classification dataset

# Just use one of the classes from torchvision.datasets. Those work without problem accross versions, unlike torchtext classes.

# 3) MODELS

#  3.1) Embeddings

import torch
import torch.nn as nn


class TextEmbeddings(nn.Module): 
    """Used in both transformer encoder and transformer decoder"""
    
    def __init__(self, d, Nmax, len_vocab, dpo):
        super().__init__()
        
        """
        d: embedding model size
        Nmax: max sequence length
        len_vocab: vocabulary length
        dpo: dropout ratio
        """
        
        self.tok_emb_table = nn.Embedding(len_vocab, d) #look-up table for word embeddings
        self.pos_emb_table = nn.Embedding(Nmax, d) #look-up table for position embeddings
        self.drop_emb = nn.Dropout(dpo) #Dropout post embeddings, before transformer layers

    def forward(self, indices): 
        """indices: encoded input tensor (containing the indices for each token) """
        B, N = indices.shape
        
        #for each token in the input tensor, find out the emb vector of length d from the vocab LUT
        tok_embeds = self.tok_emb_table(indices) #token embeddings specific to input
        
        #for each position in the input tensor,look up for pos emb vector of length d from pos LUT
        pos_embeds = self.pos_emb_table(torch.arange(N, device=indices.device)) #pos embed specific to input
        
        x = tok_embeds + pos_embeds  # (B, N, d)
        x = self.drop_emb(x) # (B, N, d)
        return x 


class Patchify(nn.Module): 
    """Takes an image and converts to patches represented with 
    embedded vectors of length d: [B,Cin,W,H] --> [B, num_patches, d]"""

    def __init__(self, H, P, Cin, d):
        super().__init__()
        
        """
        H: image height = width
        P: patch height = width
        Cin: num input channels
        d: embedding vector to represent each patch
            = num_output channels = number of filters
        """
        
        # Calculate the number of patches 
        self.num_patches = (H//P)**2 
        
        # Projection layer to convert the image into patches
        self.proj = nn.Conv2d(Cin, d, kernel_size=P, stride=P)
        
    def forward(self, x): # x: image, [B,Cin,W,H]

        x = self.proj(x) #[B, d, num_patches_per_row, num_patches_per_column]
        x = x.flatten(2) #[B, d, num_patches]
        x = x.transpose(1, 2) #[B, num_patches, d]
        
        return x


class ImageEmbeddings(nn.Module):
    """
    Combine the patch embeddings with the class token and position embeddings.
    """

    def __init__(self, H, P, Cin, d, dpo):
        super().__init__()
        
        self.patch_embeddings = Patchify(H, P, Cin, d)

        # One Learnable [CLS] token per image to embed the class info for each sequence of patches
        self.cls_token = nn.Parameter(torch.randn(1, 1, d))

        # Position embedding for each patch + one for CLS
        self.position_embeddings = nn.Parameter(torch.randn(1, self.patch_embeddings.num_patches + 1, d)) 
        # (B=1, num_patches+1, d)
        
        self.dropout = nn.Dropout(dpo) #hidden_dropout_prob


    def forward(self, x):
        x = self.patch_embeddings(x) # (B, Cin, W, H) -->  (B, num_patches, d)
        B, _, _ = x.size() #retrieve the batch size
        
        # Expand the [CLS] token to the batch size: (1, 1, d) -> (B, 1, d)
        # -1s allow to keep the associated dimensions the same
        cls_tokens = self.cls_token.expand(B, -1, -1) #one CLS token per image
        
        # Prepend the learnable [CLS] token to the input patches, enabling to embed the class info for the sequence
        x = torch.cat((cls_tokens, x), dim=1) #(B, num_patches, d) --> (B, num_patches+1, d)
       
        # add pos embed by broadcasting (same embedding for each image in the batch)
        x = x + self.position_embeddings #(B, num_patches+1, d) + (1, num_patches+1, d) --> (B, num_patches+1, d)
        x = self.dropout(x) #(B, num_patches+1, d)
        return x 


# 3.2) Transformer blocks

class MHA(nn.Module): #for encoder, no masking
    def __init__(self, d, nh, dpo): #No Nmax
        
        """
        nh: number of heads
        """
        
        super().__init__()
        
        assert d % nh == 0
        self.nh = nh
        self.dh = d // nh  

        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)
        self.Wo = nn.Linear(d, d)  
        
        self.dropout = nn.Dropout(dpo)
        #no register_buffer

    def forward(self, x):
        B, N, d = x.shape

        # Form QKV before splitting data into heads
        Q = self.Wq(x) # (B, N, d)
        K = self.Wk(x)  
        V = self.Wv(x)

        # Split QKV into heads: d -> nh x dh
        Q = Q.view(B, N, self.nh, self.dh) #(B, N, nh, dh)
        K = K.view(B, N, self.nh, self.dh)
        V = V.view(B, N, self.nh, self.dh)
        
        # Transpose
        Q = Q.transpose(1, 2) #(B, nh, N, dh)
        K = K.transpose(1, 2) 
        V = V.transpose(1, 2)

        # Calculate QKT = attention scores for each head
        S = Q @ K.transpose(2, 3) #(B, nh, N, N) 

        # Softmax along the rows, no mask applied to S
        P = torch.softmax(S / self.dh**0.5, dim=-1) #(B, nh, N, N)
        P = self.dropout(P)

        # Calculate the output of each head
        PV = (P @ V) #(B, nh, N, dh)
        
        # Concat along columns = transpose & reshape --> col dim = d = nh * dh
        PV = PV.transpose(1, 2).reshape(B, N, d) #(B, N, d)
        
        # Wo projection
        self_attn_out = self.Wo(PV) #(B, N, d)

        return self_attn_out


class MaskedMHA(nn.Module):
    def __init__(self, d, Nmax, nh, dpo):        
        super().__init__()
        
        assert d % nh == 0
        self.nh = nh
        self.dh = d // nh  

        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)
        self.Wo = nn.Linear(d, d)  
        
        self.dropout = nn.Dropout(dpo)
        
        #To ensure the intermediate tensors are automatically moved to GPU along with the model,
        #an upper triangle tensor of size Nmax by Nmax is registered as a buffer named 'mask'. 
        #The resulting buffer tensor, self.mask, is accessible as an attribute of the module.
        self.register_buffer('mask', torch.triu(torch.ones(Nmax, Nmax), diagonal=1))


    def forward(self, x):
        B, N, d = x.shape

        # Form QKV before splitting data into heads
        Q = self.Wq(x) # (B, N, d)
        K = self.Wk(x)  
        V = self.Wv(x)

        # Split QKV into heads: d -> nh x dh
        Q = Q.view(B, N, self.nh, self.dh) #(B, N, nh, dh)
        K = K.view(B, N, self.nh, self.dh)
        V = V.view(B, N, self.nh, self.dh)
        
        # Transpose
        Q = Q.transpose(1, 2) #(B, nh, N, dh)
        K = K.transpose(1, 2) 
        V = V.transpose(1, 2)

        # Calculate QKT = attention scores for each head
        S = Q @ K.transpose(2, 3) #(B, nh, N, N) 
        
        # The mask that had been registered as a buffer is truncated 
        # to the number of current input tokens and converted to boolean
        mask_bool = self.mask.bool()[:N, :N] #(N,N)

        # Use the boolean mask to 'hide' the upper right triangle in attention scores
        S.masked_fill_(mask_bool, -torch.inf) #(B, nh, N, N) 

        # Calculate the probs matrix (attention weights) using the masked S
        # Softmax along the rows
        P = torch.softmax(S / self.dh**0.5, dim=-1) #(B, nh, N, N)
        P = self.dropout(P)

        # Calculate the output of each head
        PV = (P @ V) #(B, nh, N, dh)
        
        # Concat along columns = transpose & reshape --> col dim = d = nh * dh
        PV = PV.transpose(1, 2).reshape(B, N, d) #(B, N, d)
        
        # Wo projection
        self_attn_out = self.Wo(PV) #(B, N, d)

        return self_attn_out


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FFN(nn.Module):
    def __init__(self, d):
        super().__init__()
        
        self.linear1 = nn.Linear(d, 4*d) 
        self.gelu = GELU()
        self.linear2 = nn.Linear(4*d, d) # not followed by an activation function 
        
    def forward(self, x):
        x = self.linear1(x) #(B, N, 4d)
        x = self.gelu(x)
        x = self.linear2(x) #(B, N, d)
        return x


class LayerNorm(nn.Module):
    def __init__(self, d):
        
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(d))
        self.shift = nn.Parameter(torch.zeros(d))
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True) #row-wise
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        
        #each column of norm_x scales and shifted by a dedicated parameter
        #(d)*(N,d)+(d)
        norm_x = self.scale * norm_x + self.shift
        
        return norm_x


class TransformerBlockEncoder(nn.Module):
    
    def __init__(self, d, nh, dpo): #no Nmax as argument
        super().__init__()
        self.mha = MHA(d, nh, dpo) #no masking in attention
        self.ffn = FFN(d)
        self.norm_premha = LayerNorm(d)
        self.norm_preffn = LayerNorm(d)
        self.drop_post = nn.Dropout(dpo)

    def forward(self, x):        
        x1 = x #(B,N,d)
        x = self.norm_premha(x)
        x = self.mha(x)   #(B,N,d)
        x = self.drop_post(x)
        x = x + x1  
        
        x1 = x #(B,N,d)
        x = self.norm_preffn(x)
        x = self.ffn(x) #(B,N,d)
        x = self.drop_post(x)
        x = x + x1 

        return x


class TransformerBlockDecoder(nn.Module):
    
    def __init__(self, d, Nmax, nh, dpo): #Nmax also argument
        super().__init__()
        self.mha = MaskedMHA(d, Nmax, nh, dpo) #Masked attention
        self.ffn = FFN(d)
        self.norm_premha = LayerNorm(d)
        self.norm_preffn = LayerNorm(d)
        self.drop_post = nn.Dropout(dpo)

    def forward(self, x):        
        x1 = x #(B,N,d)
        x = self.norm_premha(x)
        x = self.mha(x)   #(B,N,d)
        x = self.drop_post(x)
        x = x + x1  
        
        x1 = x #(B,N,d)
        x = self.norm_preffn(x)
        x = self.ffn(x) #(B,N,d)
        x = self.drop_post(x)
        x = x + x1 

        return x


# 3.3) Final classes combining embeddings, transformer blocks, and output heads

class TransformerEncoder(nn.Module): 
    def __init__(self, d, Nmax, nh, L, len_vocab, dpo, num_classes):
        super().__init__()

        self.embedding = TextEmbeddings(d, Nmax, len_vocab, dpo)

        self.trf_blocks = nn.Sequential(*[TransformerBlockEncoder(d, nh, dpo) for _ in range(L)])

        self.final_norm = LayerNorm(d)

        self.out_head = nn.Linear(d, num_classes, bias=False) #(d,len_vocab in decoder)

    def forward(self, x): 
        x = self.embedding(x)
        x = self.trf_blocks(x) # (B, N, d)
        x = self.final_norm(x) # (B, N, d)
        
        # Use the last token's output for classification
        logits = self.out_head(x)[:,-1,:] # (B, N, num_classes) --> (B, num_classes)
        return logits


class TransformerDecoder(nn.Module):
    def __init__(self, d, Nmax, nh, L, len_vocab, dpo):
        super().__init__()
        
        self.embedding = TextEmbeddings(d, Nmax, len_vocab, dpo)

        self.trf_blocks = nn.Sequential(*[TransformerBlockDecoder(d, Nmax, nh, dpo) for _ in range(L)])

        self.final_norm = LayerNorm(d)
        
        self.out_head = nn.Linear(d, len_vocab, bias=False) #(d,num_classes in decoder)

    def forward(self, x):
        x = self.embedding(x)
        x = self.trf_blocks(x) # (B, N, d)
        x = self.final_norm(x) # (B, N, d)
        logits = self.out_head(x) # (B, N, len_vocab)
        return logits


class VisionTransformer(nn.Module):
    """
    The ViT model for classification.
    """

    def __init__(self, H, P, Cin, d, nh, L,  dpo, num_classes):
        super().__init__()
        
        self.embedding = ImageEmbeddings(H, P, Cin, d, dpo)
        
        # Use the standard transformer encoder
        self.trf_blocks = nn.Sequential(*[TransformerBlockEncoder(d, nh, dpo) for _ in range(L)])
        
        # No final_norm
        
        self.out_head = nn.Linear(d, num_classes) #same as standard encoder

    def forward(self, x):
        x = self.embedding(x)
        x = self.trf_blocks(x) # (B, N, d)
        
        # Use the [CLS] token's output for classification
        logits = self.out_head(x[:, 0, :]) # (B, N, num_classes) --> (B, num_classes)
        return logits


# 4) TRAIN / EVALUATE / GENERATE: Single set for all transformers/tasks

# 4.1) Supporting functions

from tqdm.notebook import trange, tqdm
import matplotlib.pyplot as plt
import random
import time


def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True) # arg of max in each of row = digit w highest prob 
    correct = top_pred.eq(y.view_as(top_pred)).sum() 
    # y.view_as(top_pred) = y.view(top_pred.shape)
    # top_pred.eq(y.view_as(top_pred)) = torch.eq(top_pred, y.view_as(top_pred))
    acc = correct / y.shape[0] # no need for correct.float()
    return acc

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def plot_losses(train_losses, val_losses, train_acc, val_acc, plot, val=True):
    num_epochs = len(train_losses) 
    epochs = torch.arange(1, num_epochs+1, 1)
    plt.grid()
    #plt.figure(figsize=(14, 4))
    
    if plot == 'loss':
        #plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, label="tr")
        if val: 
            plt.plot(epochs, val_losses, label="val")
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.show();
    
    elif plot == 'acc':
        #plt.subplot(1, 2, 2)
        plt.plot(epochs, train_acc, label="tr")
        if val:
            plt.plot(epochs, val_acc, label="val")
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.show();


# 4.2) Main functions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, data, batch, optimizer, criterion):

    epoch_loss = 0
    epoch_acc = 0

    model.train() # put our model into `train` mode
    
    dataloader = DataLoader(data, batch_size=batch, shuffle=True, drop_last=True)

    for (x, y) in tqdm(dataloader, desc="Training", leave=False):

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad() # clear the gradients calculated from the last batch

        y_pred = model(x)
        
        # flatten y_pred and y for decoder application, skip for encoder
        if len(y_pred.shape) == 3: #(B, Nmax, len_vocab)
            y_pred = y_pred.flatten(0, 1) # (B, Nmax, len_vocab) --> (B x Nmax,len_vocab)
            y = y.flatten() #[B,Nmax] --> (B x Nmax)
            
        loss = criterion(y_pred, y).to(device)

        acc = calculate_accuracy(y_pred, y) 

        loss.backward() # calculate the gradients of each parameter wrt loss

        optimizer.step() # update weights

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)

def evaluate(model, data, batch, criterion):

    epoch_loss = 0
    epoch_acc = 0

    model.eval() # put our model into evaluation mode

    dataloader = DataLoader(data, batch_size=batch, shuffle=False, drop_last=False)

    with torch.no_grad(): # gradients are not calculated for whatever is inside the with block

        for (x, y) in tqdm(dataloader, desc="Evaluating", leave=False):

            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)
            
            # flatten y_pred and y for decoder application, skip for encoder
            if len(y_pred.shape) == 3: #(B, Nmax, len_vocab)
                y_pred = y_pred.flatten(0, 1) # (B, Nmax, len_vocab) --> (B x Nmax,len_vocab)
                y = y.flatten() #[B,Nmax] --> (B x Nmax)

            loss = criterion(y_pred, y).to(device)

            acc = calculate_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)


def trainval_epochs(model, datatr, dataval, batch, epochs, optimizer, criterion, plot, val=True):
  
    best_valid_loss = float('inf')
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in trange(epochs): # trange(n) is shorthand for tqdm.tqdm(range(n))

        start_time = time.monotonic()

        train_loss, train_acc = train(model, datatr, batch, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, dataval, batch, criterion)
                
        train_losses.append(train_loss)
        val_losses.append(valid_loss)
        train_accs.append(train_acc)
        val_accs.append(valid_acc)
        
        if valid_loss < best_valid_loss: # go with loss, not the accuracy
            best_valid_loss = valid_loss
            trained_params = model.state_dict()

        end_time = time.monotonic()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
        
    plot_losses(train_losses, val_losses, train_accs, val_accs, plot, val)

    return trained_params # as an ordered dictionary

def generate(eval_prompt, tokenizer, model, Nmax, Nmax_out):

    torch.manual_seed(123) 
    dpo = 0
    
    # Convert the input text to an encoded list of values
    encoded = tokenizer.encode(eval_prompt) 
    
    # list--> encoded input tensor
    idx = torch.tensor(encoded).unsqueeze(0).to(device)
    
    #put the model in eval mode --> disable dropout
    model.eval()

    #Loop Nmaxout times
    for _ in range(Nmax_out): 
        
        #limit the encoded tensor size (input to model for fwd pass) to Nmax
        idx_cond = idx[:, -Nmax:]  # (B,Nmax)
        
        #turn-off grad since fwd pass only
        with torch.no_grad(): 
            
            #fwd pass to get the predictions -->(B, N or Nmax, len_vocab)
            logits = model(idx_cond)

        # Take only the last generated logit 
        logits = logits[:, -1, :] #(B, len_vocab)

        # Optional softmax if the next token is selected by argmax (since softmax monotonic)
        probs = torch.softmax(logits, dim=-1) # (B, len_vocab)
                                              
        # Find the highest value prob or sample depending on the probs
        idx_next = torch.argmax(probs, dim=-1, keepdim=True)  # (B,1) 
        #idx_next = torch.multinomial(probs, num_samples=1) # go with softmax probability rather than argmax

        # Append to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)   #(B, N+1)
       
    #Convert the finalized tensor to output sequence
    output_seq = tokenizer.decode(idx.squeeze(0).tolist()) 

    return output_seq.rstrip("\n") #without the new lines at the end
