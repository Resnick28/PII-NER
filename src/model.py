import logging
from typing import Optional, Union

import torch
import torch.nn as nn

from transformers import (
    AutoConfig, 
    AutoModelForTokenClassification, 
    PreTrainedModel
)

# Import label mappings to ensure the model config matches the dataset
from labels import LABEL2ID, ID2LABEL

# # Configure module-level logger
logger = logging.getLogger(__name__)

'''Quantization Code'''

def optimize_for_inference(model: nn.Module) -> nn.Module:
    """
    Applies Dynamic Quantization to the model for low-latency CPU inference.

    This reduces the precision of Linear layer weights from Float32 to Int8, 
    typically reducing model size by 2-4x and speeding up CPU inference by 2-3x 
    with negligible accuracy loss.

    Args:
        model (nn.Module): The trained PyTorch model (Float32).

    Returns:
        nn.Module: The quantized model (Int8).
    """
    # logger.info("Applying Dynamic Quantization (Float32 -> Int8)...")
    
    # We only quantize Linear layers (nn.Linear). 
    # Quantizing other layers (like LayerNorm or Embeddings) usually yields 
    # diminishing returns or stability issues in dynamic quantization.
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {nn.Linear}, 
        dtype=torch.qint8
    )
    
    return quantized_model


''' Baseline Model '''

# def create_model(model_name: str):
#     model = AutoModelForTokenClassification.from_pretrained(
#         model_name,
#         num_labels=len(LABEL2ID),
#         id2label=ID2LABEL,
#         label2id=LABEL2ID,
#     )
#     return model


''' New Model '''

"""
Core module for initializing and optimizing the Token Classification model.

This module handles:
1. Model Initialization: Loading architectures with specific configuration overrides
   (label mappings, dropout regularization).
2. Inference Optimization: Applying dynamic quantization to meet strict latency 
   requirements (<20ms).
"""

def create_model(
    model_name: str, 
    dropout_rate: float = 0.2
) -> PreTrainedModel:
    """
    Initializes a Transformer-based Token Classification model with custom configuration.

    The configuration is explicitly loaded to inject label mappings and tune regularization
    parameters (dropout) which are critical for preventing overfitting on small, noisy datasets.

    Args:
        model_name (str): The HuggingFace model identifier (e.g., 'distilroberta-base').
        dropout_rate (float): The dropout probability. Defaults to 0.2 (higher than standard 0.1)
                              to improve generalization on synthetic/noisy data.

    Returns:
        PreTrainedModel: The initialized model ready for training or evaluation.

    Raises:
        OSError: If the model name is invalid or network issues occur.
    """
    logger.info(f"Initializing model: {model_name} | Dropout: {dropout_rate}")

    try:
        # 1. Load Configuration First
        # We explicitly load the config to inject label mappings. This ensures that
        # saved models retain the knowledge of which ID maps to which Label (e.g. 0 -> "O").
        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=len(LABEL2ID),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            finetuning_task="ner"
        )

        # 2. Tune Hyperparameters (Regularization)
        # For small datasets (~1k examples), standard models memorize patterns too quickly.
        # We inject higher dropout into the config to force robust feature learning.
        _apply_dropout_config(config, dropout_rate)

        # 3. Instantiate the Model
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            config=config
        )
        
        return model

    except Exception as e:
        logger.error(f"Failed to create model '{model_name}': {e}")
        raise



def _apply_dropout_config(config, rate: float) -> None:
    """Helper to safely inject dropout rates into various transformer configs."""
    
    # Standard BERT/RoBERTa attributes
    if hasattr(config, "dropout"):
        config.dropout = rate
    
    if hasattr(config, "attention_probs_dropout_prob"):
        config.attention_probs_dropout_prob = rate
    
    # Specific to BERT
    if hasattr(config, "hidden_dropout_prob"):
        config.hidden_dropout_prob = rate
        
    # Specific to DistilBERT
    if hasattr(config, "seq_classif_dropout"):
        config.seq_classif_dropout = rate
