import torch
import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.training.agent import DQN

def test_multi_head():
    input_dim = 13
    output_dim = 5
    model = DQN(input_dim, output_dim)
    
    # Test single input
    x = torch.randn(1, input_dim)
    domain_idx = 2
    out = model(x, domain_idx)
    print(f"Single output shape: {out.shape}")
    
    # Test batch input with same domain
    x_batch = torch.randn(10, input_dim)
    out_batch = model(x_batch, domain_idx)
    print(f"Batch output (same domain) shape: {out_batch.shape}")
    
    # Test batch input with different domains
    domain_idxs = torch.randint(0, 4, (10, 1))
    out_multi = model(x_batch, domain_idxs)
    print(f"Batch output (multi domain) shape: {out_multi.shape}")
    
    print("Test complete.")

if __name__ == "__main__":
    test_multi_head()
