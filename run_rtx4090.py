
"""
Run TinyZero training on RTX 4090 for 100 iterations
"""

import torch
import time
import sys
from pathlib import Path

def main():
    # Check for GPU
    if not torch.cuda.is_available():
        print("❌ No GPU detected! Please check your CUDA installation.")
        sys.exit(1)
    
    device_name = torch.cuda.get_device_name(0)
    print(f"✅ GPU Detected: {device_name}")
    
    if "4090" not in device_name:
        print(f"⚠️ Warning: Expected RTX 4090 but found {device_name}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    # Print GPU info
    print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"🚀 Starting training for 100 iterations on RTX 4090...")
    
    # Import trainer
    from train import TinyZeroTrainer
    
    # Start timing
    start_time = time.time()
    
    # Initialize trainer with RTX 4090 config
    config_path = "configs/rtx4090_100iter.yaml"
    
    try:
        trainer = TinyZeroTrainer(config_path)
        print("✅ Trainer initialized successfully")
        
        # Run training
        trainer.train()
        
        # Calculate total time
        total_time = time.time() - start_time
        minutes = int(total_time // 60)
        seconds = int(total_time % 60)
        
        print(f"\n🎉 Training completed successfully!")
        print(f"⏱️ Total time: {minutes} minutes {seconds} seconds")
        print(f"📈 Best reward achieved: {trainer.best_reward:.4f}")
        print(f"💾 Checkpoints saved in: ./checkpoints_rtx4090/")
        print(f"📊 Logs saved in: ./logs_rtx4090/")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        raise

if __name__ == "__main__":
    main()

