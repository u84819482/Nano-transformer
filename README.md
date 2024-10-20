Minimal implementation of [transformer-encoder for text classification](https://nbviewer.org/github/akrvrdr/NanoTransformer/blob/main/transformer_encoder_text_classification.ipynb), [transformer-decoder for text generation](https://nbviewer.org/github/akrvrdr/NanoTransformer/blob/main/transformer_decoder_text%20_generation.ipynb), and [ViT for image classification](https://nbviewer.org/github/akrvrdr/NanoTransformer/blob/main/vision_transformer_image_classification.ipynb) (diffusion transformer for image generation is [in this repo](https://github.com/akrvrdr/NanoDiffusion).)

[The .py file](https://github.com/akrvrdr/NanoTransformer/blob/main/transformer_utils.py) contains codes for: 
- Word, character, BPE tokenizers and vocabulary generation,
- Text generation and text classification dataset formation,
- Text and image embeddings,
- Encoder, decoder, and ViT models, with modules shared as much as possible,
- Training and evaluation, common for all three tasks.

.ipynb files minimally illustrate the training and evaluation of models by using [toy datasets](https://github.com/akrvrdr/NanoTransformer/tree/main/training%20data) (including MNIST for ViT) and light-weight transformers. However, the code in .py file should allow training scaled-up models on large datasets as well.

References:
- https://github.com/rasbt/LLMs-from-scratch
- https://github.com/karpathy/nanoGPT
- https://github.com/tintn/vision-transformer-from-scratch
