# DSG-proposal
# Focus-CLIP
### Region-Aware Vision–Language Alignment Using Mask-Augmented Inputs

*Author:* Mahi Garg
*Domain:* Vision–Language Modelling (VLM), Computer Vision

## Project Overview

Standard CLIP models possess a "global" bias, meaning they analyze an entire image simultaneously. In cluttered scenes, this often leads to the mixing of visual information from multiple objects, making it difficult to align text with a specific part of an image.

Focus-CLIP addresses this limitation by transforming CLIP from a passive observer into an active, user-controllable model. By modifying the visual encoder to accept an alpha channel (binary mask), the system explicitly directs the model's attention to specific regions.

## Key Features

* *Region Guidance:* Adds a fourth input channel (RGB + Mask) to focus attention on specific objects within the scene.
* *Architecture Modification:* Modifies the standard ViT-B/16 patch embeddings to accommodate mask-augmented inputs.
* *Efficient Training:* Utilizes Low-Rank Adaptation (LoRA) to fine-tune only ~1-2% of parameters, making training feasible on standard GPUs (e.g., Colab T4).
* *Better Alignment:* Implements region-aware contrastive loss to pull correct region-text pairs closer together in the embedding space.

## How It Works

The system moves from data preparation to inference through the following pipeline:

1.  *Data Preparation:* The project uses the Visual Genome dataset. Images are preprocessed into 4-channel inputs [R, G, B, Mask], where the mask is generated from bounding boxes.
2.  *The Architecture Tweak:* The pretrained CLIP visual encoder is modified to accept this 4-channel input structure.
3.  *Fine-Tuning:* The majority of the model (text encoder and deep vision layers) is frozen. Only the modified embedding layer and the LoRA adapters are trained.
4.  *Result:* The model produces embeddings that represent the masked region specifically, rather than the global context of the image.

## Tech Stack

* *Core:* Python, PyTorch
* *Model:* HuggingFace Transformers (CLIP ViT-B/16)
* *Training:* CUDA, PEFT/LoRA
* *Data Processing:* PIL, Torchvision

## Development Roadmap

This project is executed over a 4-week "Bring Your Own Project" (BYOP) timeline.

* [ ] *Week 1:* Dataset familiarization (Visual Genome) and baseline CLIP testing.
* [ ] *Week 2:* Modifying the Model Architecture (3 to 4 channels) and building the data pipeline.
* [ ] *Week 3:* Training loop implementation (Region-Contrastive Loss) and LoRA integration.
* [ ] *Week 4:* Final training convergence, evaluation, and comparative analysis.

## References

This project is inspired by recent advancements in controllable vision-language models.

* *Sun et al. (CVPR 2024):* Alpha-CLIP: A CLIP Model Focusing on Wherever You Want.
* *Radford et al. (2021):* Learning Transferable Visual Models From Natural Language Supervision.

## Author

*Mahi Garg*
IIT Roorkee
