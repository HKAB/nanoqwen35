import pytest
import torch
from nanoqwen35.checkpoint_manager import load_pretrained_hf
from nanoqwen35.engine import Engine

def test_load_pretrained_and_inference():
    pretrained_dir = "/home/truongnp5/Desktop/qwen35/Qwen3.5-0.8B"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model and tokenizer
    model, tokenizer, meta = load_pretrained_hf(pretrained_dir, device=device, phase="eval")
    
    # Check if they are loaded successfully
    assert model is not None
    assert tokenizer is not None
    
    # Create engine
    engine = Engine(model, tokenizer)
    
    # Run inference
    prompt = "The capital of France is"
    
    # Need to properly prepend bos token
    bos_id = tokenizer.get_bos_token_id()
    if bos_id is not None:
        tokens = tokenizer.encode(prompt, prepend=bos_id)
    else:
        tokens = tokenizer.encode(prompt)
        
    sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=10, temperature=0.0)
    output_text = tokenizer.decode(sample[0])
    
    print(f"Prompt: {prompt}")
    print(f"Output: {output_text}")
    
    # Very basic check to see if the inference runs and returns text
    assert len(output_text) > len(prompt)
    assert "Paris" in output_text or output_text != prompt
