Minimal implementation of transformer-encoder for text classification, transformer-decoder for text generation, and ViT for image classification (for image generation using diffusion transformer, see . 

[The .py file]([(https://github.com/u84819482/Nano-transformer/blob/main/transformer_utils.py)]) contains codes for: 
- Word, character, and BPE tokenizers as well as for vocabulary generation,
- Text generation and text classification datasets,
- Text and image embeddings,
- Encoder, decoder, and ViT models, with shared modules as much as possible,
- Training and evaluation, common for all three tasks.

.ipynb files minimally illustrate the training and evaluation of models by using toy datasets and small models. However [the code in .py file]([(https://github.com/u84819482/Nano-transformer/blob/main/transformer_utils.py)]) should allow training large models on serious datasets as well.
