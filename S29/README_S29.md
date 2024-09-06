## S29
**Objective: Fine-tuning Microsoft-Phi2 using QLoRA (Quantized Low-Rank Adapters) for Question Answering (QA).**

QLoRA was introduced by Tim Dettmers on May 23, 2023. It combines the techniques of 4-bit quantization with LoRA (Low-Rank Adaptation), enabling efficient fine-tuning of large language models (LLMs). Here's how it works:

1. **Quantization (4-bit precision)**:
   All weights of the large language model (LLM) are converted into a 4-bit format. This drastically reduces the memory footprint while preserving performance. A 4-bit integer can represent 16 distinct values (from 0 to 15), while a typical 32-bit float provides a much larger range. By carefully mapping these smaller values to their corresponding 32-bit floating-point representations, we can maintain reasonable accuracy during training and inference.

2. **LoRA and Adapters**:
   We utilize LoRA, a method to fine-tune LLMs with low-rank matrix adapters. Instead of training all model parameters, only the newly added low-rank adapters are fine-tuned. These adapters remain in 32-bit precision to preserve training fidelity, ensuring that only a small part of the model is trained while the core model remains frozen in its quantized 4-bit form. The adapter parameters are trained separately using 32-bit precision for optimal performance.

3. **Dequantization during inference and backpropagation**:
   During inference and backpropagation, the quantized 4-bit model is dequantized temporarily, so that full precision is used where necessary for gradient updates. The model’s core weights remain frozen in the 4-bit quantized format, but the adapter parameters are updated with 32-bit precision during backpropagation, ensuring the fine-tuning process does not suffer from low-precision issues.

For a more in-depth look, please check out the code comments and explanations on the Hugging Face space:

[Hugging Face App for ERAV2 S29 QLoRA on Microsoft-Phi2](https://huggingface.co/spaces/Vasudevakrishna/ERAV2_S29_QLora_Phi2)
