import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import get_dataloaders
from models.generator import Generator
from models.discriminator import Discriminator
from losses import dice_loss, FocalLoss
from evaluate import calculate_metrics, evaluate_model, visualize_predictions

# ============================================================
# CUDA OPTIMIZATION (BLACKWELL TENSOR CORES FULL POWER)
# ============================================================
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ============================================================
# PATHS AND SETTINGS
# ============================================================
IMAGE_PATH = 'data/images'  # Update with your local/drive paths
MASK_PATH = 'data/labels'
SAVE_DIR = 'outputs'

PRED_DIR = os.path.join(SAVE_DIR, "predictions")
GRAPH_DIR = os.path.join(SAVE_DIR, "graphs")
MODEL_DIR = os.path.join(SAVE_DIR, "models")
os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE = 512
BATCH_SIZE = 8
NUM_WORKERS = 8
TOTAL_EPOCHS = 200

LR_G = 1e-4
LR_D = 1e-5
BETA1 = 0.5
LAMBDA_L1, LAMBDA_DICE, LAMBDA_FOCAL, LAMBDA_GAN = 10.0, 5.0, 2.0, 0.2
D_UPDATE_EVERY = 3
PATIENCE = 15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_history(history, save_dir):
    epochs = range(1, len(history["val_DSC"]) + 1)
    
    plt.figure()
    plt.plot(epochs, history["val_DSC"])
    plt.title("Validation DSC"); plt.grid(); plt.savefig(os.path.join(save_dir, "validation_dsc.png")); plt.close()

    plt.figure()
    plt.plot(epochs, history["val_IOU"])
    plt.title("Validation IOU"); plt.grid(); plt.savefig(os.path.join(save_dir, "validation_iou.png")); plt.close()

    plt.figure()
    plt.plot(epochs, history["train_G"], label="Generator")
    plt.plot(epochs, history["train_D"], label="Discriminator")
    plt.title("GAN Loss Curves"); plt.legend(); plt.grid(); plt.savefig(os.path.join(save_dir, "gan_losses.png")); plt.close()

def main():
    # 1. LOAD DATA
    train_loader, val_loader, test_loader = get_dataloaders(IMAGE_PATH, MASK_PATH, IMG_SIZE, BATCH_SIZE, NUM_WORKERS)

    # 2. INSTANTIATE MODELS
    G = Generator().to(device)
    D = Discriminator().to(device)

    # 3. LOSSES & OPTIMIZERS
    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()
    criterion_focal = FocalLoss()

    optimizer_G = optim.Adam(G.parameters(), lr=LR_G, betas=(BETA1, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=LR_D, betas=(BETA1, 0.999))
    scaler = torch.cuda.amp.GradScaler()

    # 4. HISTORY AND RESUME SYSTEM
    history = {"train_G": [], "train_D": [], "val_DSC": [], "val_IOU": [], "val_PRECISION": [], "val_RECALL": []}
    best_dsc, start_epoch = 0.0, 1
    LATEST_CKPT = os.path.join(MODEL_DIR, "latest_checkpoint.pth")
    BEST_CKPT = os.path.join(MODEL_DIR, "best_model.pth")

    if os.path.exists(LATEST_CKPT):
        print("♻️ CHECKPOINT BULUNDU - RESUME SYSTEM ACTIVE")
        ckpt = torch.load(LATEST_CKPT, map_location=device)
        G.load_state_dict(ckpt["G_state_dict"])
        D.load_state_dict(ckpt["D_state_dict"])
        optimizer_G.load_state_dict(ckpt["optimizer_G_state_dict"])
        optimizer_D.load_state_dict(ckpt["optimizer_D_state_dict"])
        if "scaler_state_dict" in ckpt: scaler.load_state_dict(ckpt["scaler_state_dict"])
        history, best_dsc, start_epoch = ckpt["history"], ckpt["best_dsc"], ckpt["epoch"] + 1

    # 5. TRAINING LOOP
    for epoch in range(start_epoch, TOTAL_EPOCHS + 1):
        G.train(); D.train()
        train_g_losses, train_d_losses = [], []
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{TOTAL_EPOCHS}")

        for batch_idx, (img, mask) in enumerate(loop):
            img, mask = img.to(device, non_blocking=True), mask.to(device, non_blocking=True)

            # TRAIN GENERATOR
            optimizer_G.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                fake_logits = G(img)
                fake_mask = torch.sigmoid(fake_logits)
                pred_fake = D(img, fake_mask)

                loss_g = (criterion_bce(fake_logits, mask) +
                          LAMBDA_DICE * dice_loss(fake_logits, mask) +
                          LAMBDA_FOCAL * criterion_focal(fake_logits, mask) +
                          LAMBDA_L1 * criterion_l1(fake_mask, mask) +
                          LAMBDA_GAN * criterion_gan(pred_fake, torch.ones_like(pred_fake)))

            scaler.scale(loss_g).backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
            scaler.step(optimizer_G)
            train_g_losses.append(loss_g.item())

            # TRAIN DISCRIMINATOR
            if batch_idx % D_UPDATE_EVERY == 0:
                optimizer_D.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast():
                    loss_d = 0.5 * (criterion_gan(D(img, mask), torch.ones_like(D(img, mask))) + 
                                    criterion_gan(D(img, fake_mask.detach()), torch.zeros_like(D(img, fake_mask))))
                
                scaler.scale(loss_d).backward()
                torch.nn.utils.clip_grad_norm_(D.parameters(), 1.0)
                scaler.step(optimizer_D)
                train_d_losses.append(loss_d.item())

            scaler.update()

        # VALIDATION
        G.eval()
        val_dscs, val_ious, val_precs, val_recs = [], [], [], []
        with torch.no_grad():
            for img, mask in val_loader:
                img, mask = img.to(device), mask.to(device)
                dsc, iou, prec, rec = calculate_metrics(G(img), mask)
                val_dscs.append(dsc); val_ious.append(iou); val_precs.append(prec); val_recs.append(rec)

        avg_dsc = np.mean(val_dscs)
        history["train_G"].append(np.mean(train_g_losses))
        history["train_D"].append(np.mean(train_d_losses) if train_d_losses else 0)
        history["val_DSC"].append(avg_dsc); history["val_IOU"].append(np.mean(val_ious))
        history["val_PRECISION"].append(np.mean(val_precs)); history["val_RECALL"].append(np.mean(val_recs))

        print(f"\n📌 Epoch: {epoch} | DSC: {avg_dsc:.4f} | IOU: {np.mean(val_ious):.4f}")

        # SAVING CHECKPOINTS
        ckpt_state = {"epoch": epoch, "G_state_dict": G.state_dict(), "D_state_dict": D.state_dict(),
                      "optimizer_G_state_dict": optimizer_G.state_dict(), "optimizer_D_state_dict": optimizer_D.state_dict(),
                      "scaler_state_dict": scaler.state_dict(), "best_dsc": best_dsc, "history": history}
        
        torch.save(ckpt_state, LATEST_CKPT)
        if avg_dsc > best_dsc:
            best_dsc = avg_dsc
            ckpt_state["best_dsc"] = best_dsc
            torch.save(ckpt_state, BEST_CKPT)
            print("✅ Yeni en iyi model kaydedildi (New best model saved!)")

        # EARLY STOPPING
        if len(history["val_DSC"]) > PATIENCE and max(history["val_DSC"][-PATIENCE:]) < best_dsc:
            print(f"🛑 Early stopping tetiklendi! (Triggered)")
            break

    # 6. EVALUATION AND PLOTTING
    print("\nLoading best model for final evaluation...")
    G.load_state_dict(torch.load(BEST_CKPT, map_location=device)["G_state_dict"])
    
    evaluate_model(G, test_loader, device, SAVE_DIR)
    plot_history(history, GRAPH_DIR)
    visualize_predictions(G, test_loader, device, PRED_DIR)

if __name__ == "__main__":
    main()