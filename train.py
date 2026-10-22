import gc
import os
import pickle
import traceback
import editdistance
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from functions import OCRDataset, decode_ctc_output, set_deterministic_mode
from model import CRNN

set_deterministic_mode()


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    epoch_losses = []

    for images, labels in dataloader:
        optimizer.zero_grad()
        images = images.to(device)
        labels = [label.to(device) for label in labels]

        # Forward pass [B, C, H, W] -> [seq_len, B, num_classes]
        output = model(images)
        output = output.permute(1, 0, 2)
        log_probs = F.log_softmax(output, dim=2)

        # Calculate CTC sequence lengths
        input_lengths = torch.full(
            size=(output.size(1),),
            fill_value=output.size(0),
            dtype=torch.long,
            device=device,
        )
        target_lengths = torch.tensor(
            [len(label) for label in labels], dtype=torch.long, device=device
        )
        targets_padded = torch.cat([label for label in labels]).to(device)

        # Loss calculation & optimization
        loss = criterion(log_probs, targets_padded, input_lengths, target_lengths)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        epoch_losses.append(loss.item())
        total_loss += loss.item()

    gc.collect()
    torch.cuda.empty_cache()
    return total_loss / len(dataloader), epoch_losses


def compute_cer(predictions, references):
    total_edits = 0
    total_chars = 0
    for pred, ref in zip(predictions, references):
        total_edits += editdistance.eval(pred, ref)
        total_chars += len(ref)
    return total_edits / total_chars if total_chars > 0 else 0.0


def evaluate(model, dataloader, criterion, device, vocab):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.inference_mode():
        for images, labels in dataloader:
            images = images.to(device)
            labels = [label.to(device) for label in labels]

            outputs = model(images)
            outputs = outputs.permute(1, 0, 2)
            log_probs = F.log_softmax(outputs, dim=2)

            input_lengths = torch.full(
                size=(outputs.size(1),),
                fill_value=outputs.size(0),
                dtype=torch.long,
                device=device,
            )
            target_lengths = torch.tensor(
                [len(label) for label in labels], dtype=torch.long, device=device
            )
            targets_padded = torch.cat([label for label in labels]).to(device)

            loss = criterion(log_probs, targets_padded, input_lengths, target_lengths)
            total_loss += loss.item()

            # Decode predictions and ground truth labels
            pred_strings = decode_ctc_output(log_probs, vocab)
            target_strings = [
                "".join([vocab[c.item() - 1] for c in label if c.item() > 0])
                for label in labels
            ]

            all_preds.extend(pred_strings)
            all_targets.extend(target_strings)

    avg_loss = total_loss / len(dataloader)
    cer = compute_cer(all_preds, all_targets)
    print(f"Validation Loss: {avg_loss:.4f} | CER: {cer:.4f}")

    gc.collect()
    torch.cuda.empty_cache()
    return avg_loss


def main():
    # --- Config & Paths ---
    dataset_dir = "data/generated/horizontal"  # Switch to 'data/generated/vertical' for IJECSA
    train_csv = os.path.join(dataset_dir, "labels.csv")
    val_csv = os.path.join(dataset_dir, "labels.csv")  # Replace with actual val split path if present
    vocab_path = os.path.join(dataset_dir, "vocab.pkl")

    checkpoint_dir = "checkpoints"
    best_model_path = os.path.join(checkpoint_dir, "best_crnn.pth")
    last_model_path = os.path.join(checkpoint_dir, "last_crnn.pth")

    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Hyperparameters ---
    batch_size = 32
    lr = 1e-4
    num_epochs = 100
    max_patience = 10

    # --- Load Vocabulary ---
    print(f"Loading vocabulary from {vocab_path}...")
    with open(vocab_path, "rb") as f:
        vocab = list(pickle.load(f))

    print(f"Vocabulary Size: {len(vocab)}")

    # --- Load Datasets ---
    print("Loading Datasets...")
    train_data = OCRDataset(train_csv, vocab)
    val_data = OCRDataset(val_csv, vocab, is_train=False)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    print("Datasets Loaded Successfully.")

    # --- Model & Training Setup ---
    model = CRNN(input_height=32, num_classes=len(vocab)).to(device)
    criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start_epoch = 0
    best_val_loss = float("inf")

    # --- Resuming Priority: Load last_crnn.pth if available ---
    resume_path = last_model_path if os.path.exists(last_model_path) else best_model_path
    if os.path.exists(resume_path):
        try:
            print(f"Loading checkpoint from {resume_path}...")
            checkpoint = torch.load(resume_path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = checkpoint["epoch"]
            best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            print(f"Resumed from epoch {start_epoch} with Best Val Loss: {best_val_loss:.4f}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}. Starting from scratch.")

    # --- Training Loop ---
    print("Training Started...")
    patience = 0
    all_losses = []

    try:
        for epoch in range(start_epoch, start_epoch + num_epochs):
            train_loss, epoch_losses = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            all_losses.append(epoch_losses)

            print(f"Epoch [{epoch + 1}/{start_epoch + num_epochs}] - Train Loss: {train_loss:.4f}")
            val_loss = evaluate(model, val_loader, criterion, device, vocab)

            # Common checkpoint dictionary
            checkpoint_state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "vocab": vocab,
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": min(best_val_loss, val_loss),
            }

            # 1. Always save the LAST epoch model
            torch.save(checkpoint_state, last_model_path)

            # 2. Save the BEST epoch model if validation loss improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
                torch.save(checkpoint_state, best_model_path)
                print(f"--> Saved new BEST model to {best_model_path} (Val Loss: {best_val_loss:.4f})")
            else:
                patience += 1
                print(f"Patience: {patience}/{max_patience}")
                if patience >= max_patience:
                    print("Early stopping triggered.")
                    break

    except Exception as e:
        print("Error encountered during training:")
        traceback.print_exc()
    finally:
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()