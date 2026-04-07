import os
import math

import torch
from torch import optim, nn
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm

from unet import Unet, Unet_Depthwise
from cloud_dataset import CloudDataset

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, eps=1e-6):
        super().__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).clamp(self.eps, 1.0 - self.eps)
        targets = targets.float().clamp(0.0, 1.0)

        dims = (1, 2, 3)
        intersection = (probs * targets).sum(dim=dims)
        denom = probs.sum(dim=dims) + targets.sum(dim=dims)

        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth + self.eps)
        loss = 1.0 - dice.mean()

        if not torch.isfinite(loss):
            return torch.tensor(1.0, device=logits.device, dtype=logits.dtype)

        return loss
    
def early_stopping(val_loss, best_val_loss, patience_counter, patience, epoch, model_save_path):
    if val_loss < best_val_loss:
        save_model(epoch, model_save_path)
        return val_loss, 0, True
    patience_counter += 1
    if patience_counter >= patience:
        return best_val_loss, patience_counter, False
    return best_val_loss, patience_counter, True

def save_model(epoch, model_save_path):
    model_dir = "./tmp_models/"
    current_filename = f"epoch_{epoch}.pth"

    os.rename(os.path.join(model_dir, current_filename), model_save_path)

    for filename in os.listdir(model_dir):
        os.remove(os.path.join(model_dir, filename))

def build_train_val_datasets(data_path, csv_path, val_ratio=0.2, seed=42):
    base = CloudDataset(data_path, csv_path=csv_path, augment=False)
    n_total = len(base)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=generator).tolist()

    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    train_base = CloudDataset(data_path, csv_path=csv_path, augment=True)
    val_base = CloudDataset(data_path, csv_path=csv_path, augment=False)

    return Subset(train_base, train_idx), Subset(val_base, val_idx)

if __name__ == "__main__":
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 20
    EPOCHS = 150
    DATA_PATH = "./dataset"
    CSV_PATH = "./dataset/training_patches_38-cloud_nonempty.csv"
    MODEL_SAVE_PATH = "unet_dw_96k.pth"
    PATIENCE = 30
    BCE_DICE_MIX = [0.9, 0.1]

    os.makedirs("./tmp_models/", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataset, val_dataset = build_train_val_datasets(DATA_PATH, CSV_PATH, val_ratio=0.2, seed=42)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    model = Unet_Depthwise(in_channels=4, num_classes=1, down_layers=2, up_layers=2, first_layer_channel=32).to(device)

    #Print the number of parameters in the model
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in the model: {num_params}")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    bce = nn.BCEWithLogitsLoss()
    dice = DiceLoss()
    loss_fn = lambda y_pred, mask: BCE_DICE_MIX[0]*bce(y_pred, mask) + BCE_DICE_MIX[1]*dice(y_pred, mask)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    min_val_loss = float("inf")
    patience_counter = 0

    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_running_loss = 0.0
        train_batches = 0
        skipped_train = 0

        for img, mask in tqdm(train_dataloader):
            img = img.float().to(device, non_blocking=True)
            mask = mask.float().to(device, non_blocking=True)

            if not torch.isfinite(img).all() or not torch.isfinite(mask).all():
                skipped_train += 1
                continue

            optimizer.zero_grad(set_to_none=True)

            y_pred = model(img)
            if not torch.isfinite(y_pred).all():
                skipped_train += 1
                continue

            loss = loss_fn(y_pred, mask)
            if not torch.isfinite(loss):
                skipped_train += 1
                continue

            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if not torch.isfinite(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                skipped_train += 1
                continue

            optimizer.step()

            train_running_loss += loss.item()
            train_batches += 1

        train_loss = train_running_loss / max(1, train_batches)

        model.eval()
        val_running_loss = 0.0
        val_running_bce = 0.0
        val_running_dice = 0.0
        val_batches = 0
        skipped_val = 0

        with torch.no_grad():
            for img, mask in tqdm(val_dataloader):
                img = img.float().to(device, non_blocking=True)
                mask = mask.float().to(device, non_blocking=True)

                if not torch.isfinite(img).all() or not torch.isfinite(mask).all():
                    skipped_val += 1
                    continue

                y_pred = model(img)
                if not torch.isfinite(y_pred).all():
                    skipped_val += 1
                    continue

                val_bce = bce(y_pred, mask)
                val_dice = dice(y_pred, mask)
                loss = BCE_DICE_MIX[0] * val_bce + BCE_DICE_MIX[1] * val_dice

                if not torch.isfinite(loss):
                    skipped_val += 1
                    continue

                val_running_loss += loss.item()
                val_running_bce += val_bce.item()
                val_running_dice += val_dice.item()
                val_batches += 1

        val_loss = val_running_loss / max(1, val_batches)
        val_bce_loss = val_running_bce / max(1, val_batches)
        val_dice_loss = val_running_dice / max(1, val_batches)

        if not math.isfinite(val_loss):
            print(f"Non-finite val loss at epoch {epoch + 1}. Stopping training.")
            break

        scheduler.step(val_loss)

        print("-" * 30)
        print(f"Train Loss EPOCH {epoch + 1}: {train_loss:.4f} | train_batches={train_batches} | skipped_train={skipped_train}")
        print(
            f"Valid {epoch + 1}: BCE = {val_bce_loss:.4f}, Dice = {val_dice_loss:.4f}, Total = {val_loss:.4f} "
            f"| learning_rate={optimizer.param_groups[0]['lr']:.2e} "
            f"| val_batches={val_batches} | skipped_val={skipped_val}"
        )
        print("-" * 30)

        torch.save(model.state_dict(), f"./tmp_models/epoch_{epoch + 1}.pth")
        min_val_loss, patience_counter, continue_training = early_stopping(
            val_loss, min_val_loss, patience_counter, PATIENCE, epoch + 1, MODEL_SAVE_PATH
        )

        if not continue_training:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    torch.save(model.state_dict(), MODEL_SAVE_PATH + "_final")