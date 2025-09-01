"""
Lecture 1: Unit Testing Kit
Neural Network Fundamentals + MNIST

Simple, intuitive tests using single batch data to verify essential functionality.
Run this to quickly validate your implementation is working correctly.
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Test Configuration
BATCH_SIZE = 8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_test_batch():
    """Load a single batch for testing"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    return next(iter(dataloader))

def create_test_model():
    """Create the standard MNIST model"""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )

# ============================================================================
# TEST SUITE 1: BASIC FUNCTIONALITY TESTS
# ============================================================================

def test_data_loading():
    """Test 1: Verify data loading works correctly"""
    print("🧪 Test 1: Data Loading")
    
    try:
        images, labels = load_test_batch()
        
        # Shape tests
        assert images.shape == (BATCH_SIZE, 1, 28, 28), f"Expected {(BATCH_SIZE, 1, 28, 28)}, got {images.shape}"
        assert labels.shape == (BATCH_SIZE,), f"Expected {(BATCH_SIZE,)}, got {labels.shape}"
        
        # Value range tests
        assert images.min() >= -3.0 and images.max() <= 3.0, f"Images not normalized properly: min={images.min():.3f}, max={images.max():.3f}"
        assert labels.min() >= 0 and labels.max() <= 9, f"Labels outside valid range: min={labels.min()}, max={labels.max()}"
        
        # Data type tests
        assert images.dtype == torch.float32, f"Images should be float32, got {images.dtype}"
        assert labels.dtype == torch.int64, f"Labels should be int64, got {labels.dtype}"
        
        print("✅ Data loading: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Data loading: FAILED - {e}")
        return False

def test_model_architecture():
    """Test 2: Verify model architecture is correct"""
    print("\n🧪 Test 2: Model Architecture")
    
    try:
        model = create_test_model()
        images, _ = load_test_batch()
        
        # Forward pass test
        with torch.no_grad():
            output = model(images)
        
        # Shape test
        expected_shape = (BATCH_SIZE, 10)
        assert output.shape == expected_shape, f"Expected output shape {expected_shape}, got {output.shape}"
        
        # Parameter count test
        total_params = sum(p.numel() for p in model.parameters())
        expected_params = 784*128 + 128 + 128*64 + 64 + 64*10 + 10  # 109,386
        assert total_params == expected_params, f"Expected {expected_params} parameters, got {total_params}"
        
        # Layer count test
        num_layers = len([m for m in model.modules() if isinstance(m, (nn.Linear, nn.ReLU))])
        assert num_layers == 5, f"Expected 5 layers (3 Linear + 2 ReLU), got {num_layers}"
        
        print("✅ Model architecture: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Model architecture: FAILED - {e}")
        return False

def test_gradient_computation():
    """Test 3: Verify gradients are computed correctly"""
    print("\n🧪 Test 3: Gradient Computation")
    
    try:
        model = create_test_model()
        images, labels = load_test_batch()
        
        # Forward pass
        output = model(images)
        loss = nn.CrossEntropyLoss()(output, labels)
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist and are reasonable
        gradient_norms = []
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient computed for {name}"
            grad_norm = param.grad.norm().item()
            assert grad_norm > 0, f"Zero gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradients in {name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradients in {name}"
            gradient_norms.append(grad_norm)
        
        # Gradient magnitude sanity check
        avg_grad_norm = np.mean(gradient_norms)
        assert 0.001 < avg_grad_norm < 10.0, f"Unusual gradient magnitude: {avg_grad_norm:.6f}"
        
        print("✅ Gradient computation: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Gradient computation: FAILED - {e}")
        return False

# ============================================================================
# TEST SUITE 2: TRAINING FUNCTIONALITY TESTS
# ============================================================================

def test_training_step():
    """Test 4: Verify single training step works"""
    print("\n🧪 Test 4: Training Step")
    
    try:
        model = create_test_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        images, labels = load_test_batch()
        
        # Get initial parameters
        initial_params = [p.clone() for p in model.parameters()]
        
        # Training step
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        # Verify parameters changed
        params_changed = False
        for initial, current in zip(initial_params, model.parameters()):
            if not torch.equal(initial, current):
                params_changed = True
                break
        
        assert params_changed, "Parameters didn't change after training step"
        
        # Verify loss is reasonable
        assert 0.1 < loss.item() < 10.0, f"Unusual loss value: {loss.item():.4f}"
        
        print("✅ Training step: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Training step: FAILED - {e}")
        return False

def test_learning_progress():
    """Test 5: Verify model learns over multiple steps"""
    print("\n🧪 Test 5: Learning Progress")
    
    try:
        model = create_test_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        images, labels = load_test_batch()
        
        # Record initial loss
        with torch.no_grad():
            initial_output = model(images)
            initial_loss = criterion(initial_output, labels).item()
        
        # Train for several steps
        losses = []
        for _ in range(10):
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        # Verify learning (loss should generally decrease)
        final_loss = losses[-1]
        assert final_loss < initial_loss, f"No learning detected: initial={initial_loss:.4f}, final={final_loss:.4f}"
        
        # Verify loss is trending downward (at least better than worst)
        assert final_loss < max(losses), "Loss not improving over training"
        
        print("✅ Learning progress: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Learning progress: FAILED - {e}")
        return False

# ============================================================================
# TEST SUITE 3: MODEL BEHAVIOR TESTS
# ============================================================================

def test_prediction_sanity():
    """Test 6: Verify model predictions make sense"""
    print("\n🧪 Test 6: Prediction Sanity")
    
    try:
        model = create_test_model()
        images, labels = load_test_batch()
        
        # Get predictions
        with torch.no_grad():
            output = model(images)
            probabilities = torch.softmax(output, dim=1)
            predictions = output.argmax(dim=1)
        
        # Probability tests
        assert torch.allclose(probabilities.sum(dim=1), torch.ones(BATCH_SIZE)), "Probabilities don't sum to 1"
        assert (probabilities >= 0).all(), "Negative probabilities found"
        assert (probabilities <= 1).all(), "Probabilities > 1 found"
        
        # Prediction tests
        assert (predictions >= 0).all() and (predictions <= 9).all(), "Predictions outside valid range"
        
        # Confidence tests
        max_probs = probabilities.max(dim=1)[0]
        assert (max_probs > 0.1).all(), "Extremely low confidence predictions"
        
        print("✅ Prediction sanity: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Prediction sanity: FAILED - {e}")
        return False

def test_evaluation_mode():
    """Test 7: Verify model behaves consistently in eval mode"""
    print("\n🧪 Test 7: Evaluation Mode")
    
    try:
        model = create_test_model()
        images, _ = load_test_batch()
        
        # Test consistency in eval mode
        model.eval()
        
        with torch.no_grad():
            output1 = model(images)
            output2 = model(images)
        
        # Outputs should be identical in eval mode
        assert torch.allclose(output1, output2, rtol=1e-5), "Inconsistent outputs in eval mode"
        
        # Test difference between train and eval mode (should be same for this simple model)
        model.train()
        with torch.no_grad():
            output_train = model(images)
        
        model.eval()
        with torch.no_grad():
            output_eval = model(images)
        
        # For this simple model without dropout/batchnorm, should be identical
        assert torch.allclose(output_train, output_eval, rtol=1e-5), "Different outputs between train/eval modes (unexpected for this architecture)"
        
        print("✅ Evaluation mode: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Evaluation mode: FAILED - {e}")
        return False

# ============================================================================
# TEST SUITE 4: ROBUSTNESS TESTS
# ============================================================================

def test_edge_cases():
    """Test 8: Verify model handles edge cases"""
    print("\n🧪 Test 8: Edge Cases")
    
    try:
        model = create_test_model()
        
        # Test with single sample
        single_image = torch.randn(1, 1, 28, 28)
        with torch.no_grad():
            single_output = model(single_image)
        assert single_output.shape == (1, 10), "Failed on single sample"
        
        # Test with different batch sizes
        for batch_size in [1, 3, 16]:
            test_input = torch.randn(batch_size, 1, 28, 28)
            with torch.no_grad():
                output = model(test_input)
            assert output.shape == (batch_size, 10), f"Failed on batch size {batch_size}"
        
        # Test with extreme inputs
        extreme_inputs = [
            torch.zeros(BATCH_SIZE, 1, 28, 28),  # All zeros
            torch.ones(BATCH_SIZE, 1, 28, 28),   # All ones
            torch.randn(BATCH_SIZE, 1, 28, 28) * 10  # Large values
        ]
        
        for extreme_input in extreme_inputs:
            with torch.no_grad():
                output = model(extreme_input)
                assert not torch.isnan(output).any(), "NaN outputs with extreme inputs"
                assert not torch.isinf(output).any(), "Inf outputs with extreme inputs"
        
        print("✅ Edge cases: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Edge cases: FAILED - {e}")
        return False

# ============================================================================
# TEST SUITE 5: PERFORMANCE TESTS
# ============================================================================

def test_inference_speed():
    """Test 9: Verify inference speed is reasonable"""
    print("\n🧪 Test 9: Inference Speed")
    
    try:
        model = create_test_model()
        model.eval()
        images, _ = load_test_batch()
        
        # Warm up
        with torch.no_grad():
            _ = model(images)
        
        # Time multiple runs
        import time
        times = []
        for _ in range(10):
            start_time = time.time()
            with torch.no_grad():
                _ = model(images)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times) * 1000  # Convert to ms
        time_per_sample = avg_time / BATCH_SIZE
        
        # Speed requirements (generous for CPU)
        assert time_per_sample < 100, f"Too slow: {time_per_sample:.2f}ms per sample"
        
        print(f"✅ Inference speed: PASSED ({time_per_sample:.2f}ms per sample)")
        return True
        
    except Exception as e:
        print(f"❌ Inference speed: FAILED - {e}")
        return False

def test_memory_usage():
    """Test 10: Verify reasonable memory usage"""
    print("\n🧪 Test 10: Memory Usage")
    
    try:
        model = create_test_model()
        images, labels = load_test_batch()
        
        # Model size check
        model_size = sum(p.numel() * 4 for p in model.parameters()) / (1024**2)  # MB (float32 = 4 bytes)
        assert model_size < 10, f"Model too large: {model_size:.2f}MB"
        
        # Memory usage during forward pass
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            output = model(images)
            peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB
            assert peak_memory < 100, f"Excessive GPU memory usage: {peak_memory:.2f}MB"
        
        print(f"✅ Memory usage: PASSED (Model: {model_size:.2f}MB)")
        return True
        
    except Exception as e:
        print(f"❌ Memory usage: FAILED - {e}")
        return False

# ============================================================================
# VISUALIZATION AND DIAGNOSTIC FUNCTIONS
# ============================================================================

def visualize_test_batch():
    """Visualize the test batch for debugging"""
    images, labels = load_test_batch()
    
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            # Denormalize for visualization
            img = images[i].squeeze()
            img = img * 0.3081 + 0.1307  # Reverse normalization
            ax.imshow(img, cmap='gray')
            ax.set_title(f'Label: {labels[i].item()}')
            ax.axis('off')
    plt.tight_layout()
    plt.show()

def diagnose_model_predictions():
    """Diagnose model predictions on test batch"""
    model = create_test_model()
    images, true_labels = load_test_batch()
    
    # Quick training to get some reasonable predictions
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()
    
    for _ in range(50):
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, true_labels)
        loss.backward()
        optimizer.step()
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        output = model(images)
        probabilities = torch.softmax(output, dim=1)
        predicted_labels = output.argmax(dim=1)
        confidences = probabilities.max(dim=1)[0]
    
    # Show results
    print("\nPrediction Analysis:")
    print("Sample | True | Predicted | Confidence | Correct?")
    print("-" * 50)
    for i in range(min(8, len(images))):
        correct = "✓" if predicted_labels[i] == true_labels[i] else "✗"
        print(f"   {i}   |  {true_labels[i]}   |     {predicted_labels[i]}     |   {confidences[i]:.3f}    |    {correct}")

# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run all unit tests and return summary"""
    print("🚀 Starting Neural Network Unit Tests")
    print("=" * 60)
    
    tests = [
        test_data_loading,
        test_model_architecture,
        test_gradient_computation,
        test_training_step,
        test_learning_progress,
        test_prediction_sanity,
        test_evaluation_mode,
        test_edge_cases,
        test_inference_speed,
        test_memory_usage
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test.__name__}: FAILED - Unexpected error: {e}")
            failed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"🎯 TEST SUMMARY: {passed} PASSED, {failed} FAILED")
    
    if failed == 0:
        print("🎉 All tests passed! Your implementation is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Check the errors above and debug your implementation.")
        return False

def quick_smoke_test():
    """Quick smoke test for immediate feedback"""
    print("⚡ Quick Smoke Test")
    
    try:
        # Load data
        images, labels = load_test_batch()
        print(f"✓ Loaded batch: {images.shape}")
        
        # Create model
        model = create_test_model()
        print(f"✓ Created model: {sum(p.numel() for p in model.parameters())} parameters")
        
        # Forward pass
        output = model(images)
        print(f"✓ Forward pass: {output.shape}")
        
        # Backward pass
        loss = nn.CrossEntropyLoss()(output, labels)
        loss.backward()
        print(f"✓ Backward pass: loss = {loss.item():.4f}")
        
        print("🎉 Smoke test passed! Basic functionality works.")
        return True
        
    except Exception as e:
        print(f"❌ Smoke test failed: {e}")
        return False

if __name__ == "__main__":
    # Run quick smoke test first
    print("Starting with quick smoke test...\n")
    if quick_smoke_test():
        print("\nRunning comprehensive test suite...\n")
        run_all_tests()
        
        # Optional: Run diagnostics
        response = input("\nRun diagnostic visualizations? (y/n): ")
        if response.lower().startswith('y'):
            print("\nVisualizing test batch...")
            visualize_test_batch()
            
            print("\nAnalyzing model predictions...")
            diagnose_model_predictions()
    else:
        print("Fix basic issues before running full test suite.")