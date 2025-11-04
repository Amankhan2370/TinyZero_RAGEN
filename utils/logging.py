import logging
import time
import json
from typing import Dict, Any, Optional
from pathlib import Path
import torch

class TrainingLogger:
    """Advanced training logger with multiple outputs."""
    
    def __init__(self, log_dir: str = "logs", experiment_name: str = "tinyzero"):
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.start_time = time.time()
        
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration."""
        self.log_dir.mkdir(exist_ok=True)
        
        # Create log file path with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"{self.experiment_name}_{timestamp}.log"
        
        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger("tinyzero")
        self.logger.info(f"Training logger initialized: {log_file}")
        
    def log_metrics(self, metrics: Dict[str, Any], step: int):
        """Log training metrics."""
        # Format metrics string
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Step {step:06d} | {metric_str}")
        
        # Save to JSON file for later analysis
        metrics_file = self.log_dir / "training_metrics.jsonl"
        log_entry = {
            "step": step,
            "timestamp": time.time(),
            "elapsed_time": time.time() - self.start_time,
            **metrics
        }
        
        with open(metrics_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
            
    def log_model_checkpoint(self, model: torch.nn.Module, step: int):
        """Log model checkpoint information."""
        self.logger.info(f"Checkpoint saved at step {step}")
        
    def log_experiment_config(self, config: Dict[str, Any]):
        """Log experiment configuration."""
        config_file = self.log_dir / "experiment_config.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
            
        self.logger.info(f"Experiment config saved: {config_file}")

class WandbLogger:
    """Weights & Biases logger for experiment tracking."""
    
    def __init__(self, project: str = "tinyzero", entity: Optional[str] = None, 
                 config: Optional[Dict[str, Any]] = None):
        try:
            import wandb
            self.wandb = wandb
        except ImportError:
            self.wandb = None
            return
            
        self.run = self.wandb.init(
            project=project,
            entity=entity,
            config=config,
            reinit=True
        )
        
    def log_metrics(self, metrics: Dict[str, Any], step: int):
        """Log metrics to wandb."""
        if self.wandb and self.run:
            self.wandb.log(metrics, step=step)
            
    def watch_model(self, model: torch.nn.Module):
        """Watch model for gradient tracking."""
        if self.wandb and self.run:
            self.wandb.watch(model, log="all", log_freq=100)
            
    def finish(self):
        """Finish wandb run."""
        if self.wandb and self.run:
            self.wandb.finish()

def setup_logging(log_dir: str = "logs", experiment_name: str = "tinyzero",
                  use_wandb: bool = False, wandb_config: Optional[Dict] = None):
    """Setup logging infrastructure."""
    logger = TrainingLogger(log_dir, experiment_name)
    wandb_logger = None
    
    if use_wandb:
        wandb_logger = WandbLogger(project=experiment_name, config=wandb_config)
        
    return logger, wandb_logger