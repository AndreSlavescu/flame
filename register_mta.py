"""
This module implements multi-token attention mechanisms and provides a registry for attention functions.
It includes both softmax and sparsemax variants of multi-token attention, along with helper functions
for registering and retrieving attention implementations.
"""

from einops import rearrange
from typing import Optional
import torch
import math
from liger_kernel.transformers.functional import liger_multi_token_attention
from torch.nn import functional as F

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeats the key and value states for multi-query attention.
    
    Args:
        hidden_states (torch.Tensor): Input tensor of shape (batch, num_key_value_heads, slen, head_dim)
        n_rep (int): Number of times to repeat the states
        
    Returns:
        torch.Tensor: Repeated tensor of shape (batch, num_key_value_heads * n_rep, slen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def multi_token_attn(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    *args,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
    kernel_size: int = 3,
    stride: int = 1,
    padding: Optional[int] = None,
    dilation: int = 1,
    groups: int = 1,
    bias: bool = True,
    sparse: bool = False,
    **kwargs
) -> torch.Tensor:
    """
    Implements multi-token attention mechanism.

    Paper: https://arxiv.org/abs/2504.00927
    
    Args:
        module (torch.nn.Module): Module instance to store attention weights
        query (torch.Tensor): Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
        key (torch.Tensor): Key tensor of shape (batch_size, num_heads, seq_len, head_dim)
        value (torch.Tensor): Value tensor of shape (batch_size, num_heads, seq_len, head_dim)
        attention_mask (Optional[torch.Tensor]): Attention mask tensor
        scale (Optional[float]): Scaling factor for attention scores
        cu_seqlens (Optional[torch.LongTensor]): Cumulative sequence lengths for packed sequences
        head_first (bool): If True, expects input tensors in (batch, head, time, dim) format
        kernel_size (int): Size of the convolutional kernel
        stride (int): Stride of the convolution
        padding (Optional[int]): Padding size for the convolution
        dilation (int): Dilation factor for the convolution
        groups (int): Number of groups for grouped convolution
        bias (bool): Whether to include bias in convolution
        sparse (bool): Whether to use sparse attention
        
    Returns:
        tuple[torch.Tensor, torch.Tensor]: Output tensor and attention weights
    """
    head_dim = query.shape[-1]
    batch_size = query.shape[0]
    
    num_query_heads = query.shape[1]
    num_key_value_heads = key.shape[1]
    seq_len = query.shape[2]

    if num_query_heads != num_key_value_heads:
        key = repeat_kv(key, num_query_heads // num_key_value_heads)
        value = repeat_kv(value, num_query_heads // num_key_value_heads)

    if scale is None:
        scale = 1.0 / (head_dim ** 0.5)
    
    if not head_first:
        query, key, value = map(lambda x: rearrange(x, 'b t h d -> b h t d'), (query, key, value))
    
    query_len = query.shape[-2]
    key_len = key.shape[-2]
    
    wei = torch.matmul(query, key.transpose(2, 3))
    wei = wei * scale
    
    mask = torch.tril(torch.ones(key_len, key_len, device=query.device))
    wei = wei.masked_fill(mask[key_len-query_len:key_len, :key_len] == 0, float('-inf'))
    
    scores = wei.view(batch_size * num_query_heads, 1, query_len, key_len)
    
    if padding is None:
        padding = kernel_size // 2
    
    cache_key = f"_mta_weight_{kernel_size}_{groups}_{bias}_{sparse}"
    if not hasattr(module, cache_key) or getattr(module, cache_key) is None:
        out_channels = 1
        in_channels = 1
        weight = torch.empty(
            out_channels, in_channels // groups, kernel_size, kernel_size,
            device=scores.device, dtype=scores.dtype, requires_grad=True
        )
        torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        setattr(module, cache_key, weight)
        
        if bias:
            bias_key = f"_mta_bias_{kernel_size}_{groups}_{sparse}"
            bias_tensor = torch.empty(out_channels, device=scores.device, dtype=scores.dtype, requires_grad=True)
            torch.nn.init.zeros_(bias_tensor)
            setattr(module, bias_key, bias_tensor)
        else:
            bias_key = f"_mta_bias_{kernel_size}_{groups}_{sparse}"
            setattr(module, bias_key, None)
    
    weight = getattr(module, cache_key)
    bias_key = f"_mta_bias_{kernel_size}_{groups}_{sparse}"
    bias_tensor = getattr(module, bias_key)
    
    attention_output = liger_multi_token_attention(
        scores=scores,
        weight=weight,
        bias=bias_tensor,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        sparse=sparse,
    )
    
    attention_weights = attention_output.view(batch_size, num_query_heads, query_len, key_len)
    
    o = torch.matmul(attention_weights, value)
    
    if not head_first:
        o = rearrange(o, 'b h t d -> b t h d')
    
    return o, attention_weights

def softmax_multi_token_attention(*args, **kwargs):
    """
    Wrapper for multi-token attention using softmax normalization.
    
    Args:
        *args: Positional arguments passed to multi_token_attn
        **kwargs: Keyword arguments passed to multi_token_attn
        
    Returns:
        tuple[torch.Tensor, torch.Tensor]: Output tensor and attention weights
    """
    kwargs['sparse'] = False
    return multi_token_attn(*args, **kwargs)

def sparsemax_multi_token_attention(*args, **kwargs):
    """
    Wrapper for multi-token attention using sparsemax normalization.
    
    Args:
        *args: Positional arguments passed to multi_token_attn
        **kwargs: Keyword arguments passed to multi_token_attn
        
    Returns:
        tuple[torch.Tensor, torch.Tensor]: Output tensor and attention weights
    """
    kwargs['sparse'] = True
    return multi_token_attn(*args, **kwargs)

class AttentionRegistry:
    """
    Registry class for managing different attention implementations.
    Provides functionality to register, retrieve and list available attention functions.
    """
    _registry = {}
    
    @classmethod
    def register(cls, name: str, attention_fn):
        """
        Register an attention function with the given name.
        
        Args:
            name (str): Name to register the attention function under
            attention_fn: The attention function to register
        """
        cls._registry[name] = attention_fn
        print(f"Registered attention function: {name}")
    
    @classmethod
    def get(cls, name: str):
        """
        Retrieve a registered attention function by name.
        
        Args:
            name (str): Name of the attention function to retrieve
            
        Returns:
            The registered attention function
            
        Raises:
            ValueError: If the requested attention function is not found
        """
        if name not in cls._registry:
            raise ValueError(f"Attention function '{name}' not found in registry. Available: {list(cls._registry.keys())}")
        return cls._registry[name]
    
    @classmethod
    def list_available(cls):
        """
        List all registered attention function names.
        
        Returns:
            list[str]: List of registered attention function names
        """
        return list(cls._registry.keys())

AttentionRegistry.register("softmax_multi_token", softmax_multi_token_attention)
AttentionRegistry.register("sparsemax_multi_token", sparsemax_multi_token_attention)

def register_attention(name: str, attention_fn):
    """
    Register a new attention function in the registry.
    
    Args:
        name (str): Name to register the attention function under
        attention_fn: The attention function to register
    """
    AttentionRegistry.register(name, attention_fn)

def get_attention(name: str):
    """
    Retrieve a registered attention function by name.
    
    Args:
        name (str): Name of the attention function to retrieve
        
    Returns:
        The registered attention function
    """
    return AttentionRegistry.get(name)
