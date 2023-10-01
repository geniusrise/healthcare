import logging
import os
from typing import List, Optional, Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

log = logging.getLogger(__name__)


def quantize_transformer_model_with_autogptq(
    model_id: str,
    bits: int,
    save_dir: str,
    dataset: Union[str, List[str]] = "c4",
    device_map: Optional[str] = "auto",
    group_size: Optional[int] = 128,
    damp_percent: Optional[float] = 0.1,
    desc_act: Optional[bool] = False,
    sym: Optional[bool] = True,
    true_sequential: Optional[bool] = True,
    use_cuda_fp16: Optional[bool] = True,
    model_seqlen: Optional[int] = None,
    block_name_to_quantize: Optional[str] = None,
    module_name_preceding_first_block: Optional[List[str]] = None,
    batch_size: Optional[int] = 1,
    pad_token_id: Optional[int] = None,
    disable_exllama: Optional[bool] = False,
) -> AutoModelForCausalLM:
    """
    Quantizes a Hugging Face Transformer model using AutoGPTQ.

    Parameters:
    - model_id (str): The identifier of the pre-trained model.
    - bits (int): The number of bits to quantize to (2, 3, 4, 8).
    - dataset (Union[str, List[str]]): The dataset used for quantization.
    - save_dir (str): Directory to save the quantized model.
    - device_map (Optional[str]): Device map for model placement. Default is "auto".
    - group_size (Optional[int]): The group size for quantization. Default is 128.
    - damp_percent (Optional[float]): Dampening percent. Default is 0.1.
    - desc_act (Optional[bool]): Whether to quantize columns in order of decreasing activation size. Default is False.
    - sym (Optional[bool]): Whether to use symmetric quantization. Default is True.
    - true_sequential (Optional[bool]): Whether to perform sequential quantization. Default is True.
    - use_cuda_fp16 (Optional[bool]): Whether to use optimized CUDA kernel for fp16. Default is False.
    - model_seqlen (Optional[int]): The maximum sequence length that the model can take.
    - block_name_to_quantize (Optional[str]): The transformers block name to quantize.
    - module_name_preceding_first_block (Optional[List[str]]): The layers preceding the first Transformer block.
    - batch_size (Optional[int]): The batch size used when processing the dataset. Default is 1.
    - pad_token_id (Optional[int]): The pad token id.
    - disable_exllama (Optional[bool]): Whether to use exllama backend. Default is False.

    Returns:
    - AutoModelForCausalLM: The quantized model.
    """
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Create GPTQConfig
    gptq_config = GPTQConfig(
        bits=bits,
        dataset=dataset,
        tokenizer=tokenizer,
        group_size=group_size,
        damp_percent=damp_percent,
        desc_act=desc_act,
        sym=sym,
        true_sequential=true_sequential,
        use_cuda_fp16=use_cuda_fp16,
        model_seqlen=model_seqlen,
        block_name_to_quantize=block_name_to_quantize,
        module_name_preceding_first_block=module_name_preceding_first_block,
        batch_size=batch_size,
        pad_token_id=pad_token_id,
        disable_exllama=disable_exllama,
    )

    # Load and quantize the model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16 if use_cuda_fp16 else torch.float32,
        quantization_config=gptq_config,
    )

    # Save the quantized model
    model.save_pretrained(save_dir)

    log.info(f"Quantized model saved to {save_dir}")

    return model


def load_huggingface_model(
    model_name: str,
    use_cuda: bool = False,
    precision: str = "float16",
    quantize: bool = False,
    quantize_bits: int = 8,
    quantize_dataset: str = "c4",
    quantize_save_dir: str = "./quantized_model",
    max_memory={0: "24GB"},
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Loads a Hugging Face model and tokenizer optimized for inference.

    Parameters:
    - model_name (str): The name of the model to load.
    - use_cuda (bool): Whether to use CUDA for GPU acceleration. Default is False.
    - precision (str): The bit precision for model and tokenizer. Options are 'float32', 'float16', 'bfloat16'. Default is 'float16'.
    - quantize (bool): Whether to quantize the model. Default is False.
    - quantize_bits (int): Bit-width for quantization. Only 4 and 8 are supported. Default is 8.
    - quantize_dataset (str): The dataset used for quantization. Default is "c4".
    - quantize_save_dir (str): Directory to save the quantized model. Default is "./quantized_model".
    - max_memory (str): Maximum GPU memory to be allocated.

    Returns:
    Tuple[AutoModelForCausalLM, AutoTokenizer]: The loaded model and tokenizer.

    Usage:
    ```python
    model, tokenizer = load_huggingface_model("gpt-2", use_cuda=True, precision='float32', quantize=True, quantize_bits=8)
    ```

    Note:
    - If `quantize` is set to False, the model will not be quantized regardless of the `quantize_bits` value.
    """
    log.info(f"Loading Hugging Face model: {model_name}")

    # Determine the torch dtype based on precision
    if precision == "float16":
        torch_dtype = torch.float16
    elif precision == "float32":
        torch_dtype = torch.float32
    elif precision == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        raise ValueError("Unsupported precision. Choose from 'float32', 'float16', 'bfloat16'.")

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype=torch_dtype)
    quantized_model_path = os.path.join(quantize_save_dir, f"{model_name}_quantized.pth")

    # Check if a quantized model exists and should be loaded
    if quantize and os.path.exists(quantized_model_path):
        log.info(f"Loading saved quantized model from {quantize_save_dir}")
        model = AutoModelForCausalLM.from_pretrained(
            quantize_save_dir,
            torch_dtype=torch_dtype,
            torchscript=True,
            max_memory=max_memory,
            device_map="auto",
        )
    else:
        # Quantize the model if requested and save it
        if quantize:
            model = quantize_transformer_model_with_autogptq(
                model_id=model_name,
                bits=quantize_bits,
                save_dir=quantize_save_dir,
                use_cuda_fp16=True if "float16" in precision else False,
            )
            log.info(f"Model quantized to {quantize_bits}-bits.")
        else:
            log.info(f"Loading model from {model_name}")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                torchscript=True,
                max_memory=max_memory,
                device_map="auto",
            )

    # Set to evaluation mode for inference
    model.eval()

    # Check if CUDA should be used
    if use_cuda and torch.cuda.is_available():
        log.info("Using CUDA for Hugging Face model.")
        model.to("cuda:0")

    log.debug("Hugging Face model and tokenizer loaded successfully.")
    return model, tokenizer
