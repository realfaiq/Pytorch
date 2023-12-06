"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import torch
import argparse
import data_setup, engine, model_builder, utils

from torchvision import transforms

def train(num_epochs: int,
          batch_size: int,
          hidden_units: int,
          learning_rate: int,
          save_path: str):
    # Setup directories
    train_dir = "pizza_steak_sushi/train"
    test_dir = "pizza_steak_sushi/test"

    # Setup target device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create transforms
    data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
    ])

    # Create DataLoaders with help from data_setup.py
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=data_transform,
        batch_size=batch_size
    )

    # Create model with help from model_builder.py
    model = model_builder.TinyVGG(
        input_shape=3,
        hidden_units=hidden_units,
        output_shape=len(class_names)
    ).to(device)

    # Set loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=learning_rate)

    # Start training with help from engine.py
    engine.train(model=model,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                epochs=num_epochs,
                device=device)

    # Save the model with help from utils.py
    utils.save_model(model=model,
                    target_dir="models",
                    model_name=f"{save_path}.pth")

def main():
    parser = argparse.ArgumentParser(description="Please provide these hyperparameters to train the model.\n1. No. of epochs\n2. Batch_size\n3. Hidden units\n4. Learning rate\n5. Saving directory path")
    parser.add_argument('num_epochs', type=int, help='No. of Epochs')
    parser.add_argument('batch_size', type=int, help="Batch size")
    parser.add_argument('hidden_units', type=int, help="Hidden units")
    parser.add_argument("learning_rate", type=float, help="Learning rate")
    parser.add_argument("save_path", type=str, help="Model Saving directory path")

    args = parser.parse_args()

    train(args.num_epochs, args.batch_size, args.hidden_units, args.learning_rate, args.save_path)
    print("Training Completed!")

if __name__ == '__main__':
    main()




