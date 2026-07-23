import os
from pathlib import Path
from dotenv import load_dotenv # For loading environment variables from a .env file
import wandb
import anomalib.data
from anomalib.callbacks import ModelCheckpoint
import lightning.pytorch.callbacks 
import anomalib.metrics
# from anomalib.pipelines import Benchmark
from anomalib.engine import Engine
import anomalib.models
from anomalib.loggers import AnomalibWandbLogger
from omegaconf import OmegaConf 
from argparse import ArgumentParser
from datetime import datetime

load_dotenv() # Load environment variables from .env file
PREBUILT_DATA_MODULES = [   
    "BMAD",
    "MPDD",
    "VAD",
    "BTech",
    "Kaputt",
    "Kolektor",
    "MVTecAD",
    "MVTecAD2",
    "MVTecLOCO",
    "RealIAD",
    "Visa"]

def setup_directory(root: Path, dataset: str) -> Path:
    dataset_path = Path(os.path.join(root, "datasets", dataset))
    print(f"Current data directory: {dataset_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory {dataset_path} does not exist. Please check the path and try again.")
    return dataset_path




if __name__ == "__main__":
    parser = ArgumentParser(description="Train and evaluate an anomaly detection model using Anomalib.")
    parser.add_argument("--config", type=Path, default="configs/anomaly_quai.yaml", help="Path to the YAML configuration file.")
    parser.add_argument("--dataset", type=str, default="MVTecAD", help="Name of the dataset. If using a prebuilt datamodule, choose the category name in the configuration. Otherwise, provide a custom dataset name and ensure the data is organized in the expected folder structure.")
    parser.add_argument("--logger",action="store_true", help="Whether to use Weights & Biases for logging.Ensure API key is in .env file.")
    parser.add_argument("--mode", type= str, choices=["train","test"], default="train", help="Mode of operation: 'train' for training and evaluation, 'test' for evaluation only.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to a model checkpoint for evaluation. Required if mode is 'test'.")
    args = parser.parse_args()
    
    if args.mode == "test" and args.checkpoint is None:
        raise ValueError("Checkpoint path must be provided in test mode.")
    
    # Load configuration from YAML file
    config = OmegaConf.load(args.config)
    print(f"Loaded configuration at {args.config}")

    # Set up data directory
    data_dir = setup_directory(root = config.root, dataset=args.dataset)
    print(f"Using dataset: {args.dataset} at {data_dir}")

    #Load Data
    if args.dataset in PREBUILT_DATA_MODULES:
        datamodule = getattr(anomalib.data, args.dataset)(root=data_dir, **config.data.prebuilt)
        datamodule.setup()
        print(f"Using prebuilt datamodule for {args.dataset}")
        print(f"Number of training samples: {len(datamodule.train_dataloader().dataset)}")
        print(f"Number of validation samples: {len(datamodule.val_dataloader().dataset)}")
        print(f"Number of test samples: {len(datamodule.test_dataloader().dataset)}")
    else:
        datamodule = anomalib.data.Folder(name=args.dataset, root=data_dir, **config.data.custom)
        datamodule.setup()
        print(f"Using custom folder datamodule for {args.dataset}")
        print(f"Number of training samples: {len(datamodule.train_dataloader().dataset)}")
        print(f"Number of validation samples: {len(datamodule.val_dataloader().dataset)}")
        print(f"Number of test samples: {len(datamodule.test_dataloader().dataset)}")

    #Setup logger
    if args.logger:
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        logger = AnomalibWandbLogger(**config.wandb)
    else:
        logger = None
    
    #Setup callbacks
    if config.callbacks.enabled:
        callbacks = []
        for callback_name, callback_config in config.callbacks.items():
            #Skips the "enabled" key in the callbacks configuration, which is not a callback class.
            if callback_name == "enabled":
                continue  
            # Add date and model name to ModelCheckpoint filename
            if callback_name == "ModelCheckpoint":
                callback_config = dict(callback_config)
                date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
                callback_config["filename"] = f"{config.model.name}-{date_str}"
                # Use the custom ModelCheckpoint class from anomalib.callbacks
                callback = ModelCheckpoint(**callback_config)
                callbacks.append(callback)  
            
            else:
                # Use the standard callback classes from lightning.pytorch.callbacks for other callbacks
                callback = getattr(lightning.pytorch.callbacks, callback_name)(**callback_config)
                callbacks.append(callback)
    else:
        callbacks = None
    print(f"Using callbacks: {callbacks}")

    #setup evaluator
    val_metrics = []
    test_metrics = []
    for stage, metric_list in config.metrics.items():
        target_metrics = val_metrics if stage == "val" else test_metrics if stage == "test" else None
        if target_metrics is None:
            continue

        for metric_config in metric_list:
            metric_dict = OmegaConf.to_container(metric_config, resolve=True)
            metric_name = metric_dict.pop("name")
            metric_class = getattr(anomalib.metrics, metric_name)
            target_metrics.append(metric_class(**metric_dict))
    evaluator = anomalib.metrics.Evaluator(val_metrics=val_metrics, test_metrics=test_metrics)

    #setup model
    if args.mode == "train":
        model = getattr(anomalib.models, config.model.name)(evaluator=evaluator)
        print(f"Initialized model: {config.model.name} with evaluator: {evaluator}")
    if args.mode == "test":
        # Extract model name from checkpoint filename
        model = getattr(anomalib.models, args.checkpoint.stem.split('-')[0])(evaluator=evaluator)
        print(f"Initialized model: {args.checkpoint.stem.split('-')[0]} with evaluator: {evaluator} for testing using checkpoint: {args.checkpoint}")
       



    # Initialize the engine with the model and configuration
    engine = Engine(callbacks=callbacks,
                    logger=logger,
                    **config.engine
                )

    # Train and test the model
    if args.mode == "train":
        engine.fit(datamodule=datamodule, model=model)
        engine.test(datamodule=datamodule, model=model)

    # Evaluate the model in test mode using the provided checkpoint
    if args.mode == "test":
        engine.test(datamodule=datamodule, model=model, ckpt_path=args.checkpoint)




