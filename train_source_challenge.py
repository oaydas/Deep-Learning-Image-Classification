import torch

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend

import matplotlib.pyplot as plt

from train_common import (
    count_parameters,
    evaluate_epoch,
    early_stopping,
    restore_checkpoint,
    save_checkpoint,
    train_epoch,
)
from dataset import get_train_val_test_loaders
from model.source_challenge import Source
from utils import config, set_random_seed, make_training_plot


def main() -> None:
    """Train source model on multiclass data."""
    set_random_seed()
    
    # Data loaders for the source task (8 classes)
    tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
        task="source",
        batch_size=config("source.batch_size"),
    )

    # Instantiate source model
    model = Source()

    # Loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.01)

    print("Number of float-valued parameters:", count_parameters(model))

    # Restore checkpoint if exists
    print("Loading source...")
    model, start_epoch, stats = restore_checkpoint(model, config("source.checkpoint"))

    # Plot setup
    axes = make_training_plot("Source Training")

    # Evaluate model before training
    evaluate_epoch(
        axes,
        tr_loader,
        va_loader,
        te_loader,
        model,
        criterion,
        start_epoch,
        stats,
        multiclass=True,
    )

    # Early stopping parameters
    prev_val_loss = stats[0][1]
    patience = 10
    curr_patience = 0

    # Training loop
    epoch = start_epoch
    while curr_patience < patience:
        train_epoch(tr_loader, model, criterion, optimizer)

        evaluate_epoch(
            axes,
            tr_loader,
            va_loader,
            te_loader,
            model,
            criterion,
            epoch + 1,
            stats,
            multiclass=True,
        )

        save_checkpoint(model, epoch + 1, config("source.checkpoint"), stats)

        curr_patience, prev_val_loss = early_stopping(stats, curr_patience, prev_val_loss)
        epoch += 1

    print("Finished Training")
    plt.savefig(f"challenge_source_training_plot_patience={patience}.png", dpi=200)


if __name__ == "__main__":
    main()
