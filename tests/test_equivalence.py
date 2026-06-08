import sys
import os
import torch
import pytest

# Add the notebooks directory to path so we can import baseline classes
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../notebooks")))
import baseline as orig

# Import modularized model classes
from models.feature_encoder import FeatureEncoder as ModFeatureEncoder
from models.convgru import ConvGRUCell as ModConvGRUCell
from models.correlation import AllPairsCorrelationVolume as ModCorrVolume
from models.pi_raft import PhysicsInformedRAFT as ModPhysicsInformedRAFT

def test_feature_encoder_equivalence():
    torch.manual_seed(42)
    orig_encoder = orig.FeatureEncoder(output_dim=128)
    
    torch.manual_seed(42)
    mod_encoder = ModFeatureEncoder(output_dim=128)
    
    # Verify weights are initialized identically
    for (name_o, param_o), (name_m, param_m) in zip(orig_encoder.named_parameters(), mod_encoder.named_parameters()):
        assert name_o == name_m
        assert torch.allclose(param_o, param_m), f"Weights for {name_o} mismatch."

    dummy_input = torch.randn(2, 1, 256, 256)
    
    orig_encoder.eval()
    mod_encoder.eval()
    
    with torch.no_grad():
        out_o = orig_encoder(dummy_input)
        out_m = mod_encoder(dummy_input)
        
    assert out_o.shape == out_m.shape, f"Shape mismatch: {out_o.shape} vs {out_m.shape}"
    assert torch.allclose(out_o, out_m), "Output activations mismatch."

def test_convgru_equivalence():
    torch.manual_seed(42)
    orig_gru = orig.ConvGRUCell(hidden_dim=128, input_dim=51)
    
    torch.manual_seed(42)
    mod_gru = ModConvGRUCell(hidden_dim=128, input_dim=51)
    
    for (name_o, param_o), (name_m, param_m) in zip(orig_gru.named_parameters(), mod_gru.named_parameters()):
        assert name_o == name_m
        assert torch.allclose(param_o, param_m)

    dummy_h = torch.randn(2, 128, 32, 32)
    dummy_x = torch.randn(2, 51, 32, 32)
    
    orig_gru.eval()
    mod_gru.eval()
    
    with torch.no_grad():
        out_o = orig_gru(dummy_h, dummy_x)
        out_m = mod_gru(dummy_h, dummy_x)
        
    assert out_o.shape == out_m.shape
    assert torch.allclose(out_o, out_m)

def test_full_model_equivalence():
    torch.manual_seed(42)
    orig_model = orig.PhysicsInformedRAFT(hidden_dim=128, corr_radius=3)
    
    torch.manual_seed(42)
    mod_model = ModPhysicsInformedRAFT(hidden_dim=128, corr_radius=3)
    
    # Verify state dict keys match exactly (no nested sub-modules under height_head)
    orig_keys = set(orig_model.state_dict().keys())
    mod_keys = set(mod_model.state_dict().keys())
    assert orig_keys == mod_keys, f"State dict keys mismatch: {orig_keys ^ mod_keys}"
    
    # Load weights from orig_model to mod_model to be 100% sure
    mod_model.load_state_dict(orig_model.state_dict())
    
    orig_model.eval()
    mod_model.eval()
    
    # Inputs
    images = torch.randn(1, 4, 1, 512, 512)
    dem = torch.zeros(1, 1, 512, 512)
    
    # Compute forward pass (with zero seed to ensure random correlation lookup is identical if run on same device state)
    torch.manual_seed(100)
    with torch.no_grad():
        flow_o, height_o = orig_model(images, dem, iters=4)
        
    torch.manual_seed(100)
    with torch.no_grad():
        flow_m, height_m = mod_model(images, dem, iters=4)
        
    assert flow_o.shape == flow_m.shape == (1, 3, 2, 512, 512)
    assert height_o.shape == height_m.shape == (1, 3, 1, 512, 512)
    assert torch.allclose(flow_o, flow_m), "Flow predictions mismatch."
    assert torch.allclose(height_o, height_m), "Height predictions mismatch."
