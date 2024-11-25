from utils.util import set_seed, infer_and_save
set_seed(464562)

from data.dataloader import get_dataloaders
from models.unet import UNet
from utils.trainer import UNetTrainer
from utils.toCSV import convert_infer_to_csv
from config import *
import segmentation_models_pytorch as smp
import argparse

# Argument parser for WandB key
parser = argparse.ArgumentParser(description="Machine Translation Training Script")
parser.add_argument("--wandb_key", type=str, required=True, help="Wandb key for logging")
args = parser.parse_args()

def main():
    # Set name of experiment
    experiment_name = EXPERIMENT_NAME
    
    # Get dataloaders
    dataloaders = get_dataloaders(image_dir=TRAIN_IMAGE_DIR, mask_dir=TRAIN_MASK_DIR, 
                                  batch_size=BATCH_SIZE, img_size=IMG_SIZE, val_split=VAL_SPLIT, augmentations=AUGMENTATIONS)
    
    # Initialize model
    model = smp.UnetPlusPlus(
        encoder_name="resnet34",        
        encoder_weights="imagenet",     
        in_channels=3,                  
        classes=3     
    )
    
    # Initialize trainer
    trainer = UNetTrainer(model=model,
                          name=experiment_name,
                          learning_rate=LEARNING_RATE,
                          wandb_key=args.wandb_key,
                          device=DEVICE)
    
    # Train model
    trainer.train(dataloaders['train'], dataloaders['val'], n_epochs=NUM_EPOCHS)
    
    # Infer and save masks
    infer_and_save(model, TEST_IMAGE_DIR, f'results/{EXPERIMENT_NAME}/infer_image/', DEVICE)
    
    # Convert inference masks to CSV
    convert_infer_to_csv(f'results/{EXPERIMENT_NAME}/infer_image/', 'results/submission.csv')

if __name__ == '__main__':
    main()
