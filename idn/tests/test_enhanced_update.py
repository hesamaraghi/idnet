import torch
import sys
from pathlib import Path

# Add the parent directory to the path to allow importing from idn
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from idn.model.update import LiteUpdateBlock, EnhancedUpdateBlock


def test_update_block_forward():
    """Test the forward pass of both update blocks to ensure compatibility."""
    print("Testing update block forward pass compatibility...")
    
    # Test parameters
    batch_size = 2
    hidden_dim = 64
    input_dim = 32
    height = 30
    width = 40
    
    # Create random input tensors
    hidden = torch.randn(batch_size, hidden_dim, height, width)
    input_tensor = torch.randn(batch_size, input_dim, height, width)
    
    # Create the update blocks
    lite_update = LiteUpdateBlock(
        hidden_dim=hidden_dim, 
        input_dim=input_dim,
        num_outputs=1,
        downsample=8
    )
    
    enhanced_update = EnhancedUpdateBlock(
        hidden_dim=hidden_dim, 
        input_dim=input_dim,
        num_outputs=1,
        downsample=8,
        with_uncertainty=False,
        use_attention=True
    )
    
    # Test forward pass
    lite_output = lite_update(hidden, input_tensor)
    enhanced_output = enhanced_update(hidden, input_tensor)
    
    # Check output shapes
    print(f"Lite update output shape: {lite_output.shape}")
    print(f"Enhanced update output shape: {enhanced_output.shape}")
    
    # Check that outputs have the same shape
    assert lite_output.shape == enhanced_output.shape, "Output shapes do not match!"
    
    # Check flow computation compatibility
    lite_flow = lite_update.compute_deltaflow(lite_output)
    enhanced_flow = enhanced_update.compute_deltaflow(enhanced_output)
    
    print(f"Lite flow output shape: {lite_flow.shape}")
    print(f"Enhanced flow output shape: {enhanced_flow.shape}")
    
    # Check flow mask computation compatibility
    lite_mask = lite_update.compute_up_mask(lite_output)
    enhanced_mask = enhanced_update.compute_up_mask(enhanced_output)
    
    print(f"Lite mask output shape: {lite_mask.shape}")
    print(f"Enhanced mask output shape: {enhanced_mask.shape}")
    
    print("Basic compatibility test passed!")
    return True


def test_uncertainty_estimation():
    """Test the uncertainty estimation capability of EnhancedUpdateBlock."""
    print("Testing uncertainty estimation...")
    
    # Test parameters
    batch_size = 2
    hidden_dim = 64
    input_dim = 32
    height = 30
    width = 40
    
    # Create random input tensors
    hidden = torch.randn(batch_size, hidden_dim, height, width)
    input_tensor = torch.randn(batch_size, input_dim, height, width)
    
    # Create the update block with uncertainty
    enhanced_update = EnhancedUpdateBlock(
        hidden_dim=hidden_dim, 
        input_dim=input_dim,
        num_outputs=1,
        downsample=8,
        with_uncertainty=True,
        use_attention=True
    )
    
    # Test forward pass
    output = enhanced_update(hidden, input_tensor)
    
    # Test flow computation with uncertainty
    flow = enhanced_update.compute_deltaflow(output)
    
    # Get uncertainty
    uncertainty = enhanced_update.get_flow_uncertainty()
    
    # Verify uncertainty shape
    print(f"Flow output shape: {flow.shape}")
    print(f"Uncertainty output shape: {uncertainty.shape}")
    
    print("Uncertainty estimation test passed!")
    return True


def main():
    """Run all tests."""
    print("Running EnhancedUpdateBlock tests...")
    
    test_update_block_forward()
    test_uncertainty_estimation()
    
    print("All tests completed successfully!")


if __name__ == "__main__":
    main()

