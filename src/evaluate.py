import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def calculate_metrics(logits, target, eps=1e-6):
    pred = torch.sigmoid(logits)
    pred = (pred > 0.5).float()

    tp = (pred * target).sum(dim=(1, 2, 3))
    fp = (pred * (1 - target)).sum(dim=(1, 2, 3))
    fn = ((1 - pred) * target).sum(dim=(1, 2, 3))

    dsc = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    return dsc.mean().item(), iou.mean().item(), precision.mean().item(), recall.mean().item()

def evaluate_model(G, test_loader, device, save_dir):
    G.eval()
    results = []

    print("\n================================================")
    print("🏆 RUNNING EVALUATION ON TEST SET")
    print("================================================")
    
    with torch.no_grad():
        for img, mask in tqdm(test_loader):
            img, mask = img.to(device), mask.to(device)
            logits = G(img)
            dsc, iou, precision, recall = calculate_metrics(logits, mask)

            results.append({"DSC": dsc, "IOU": iou, "Precision": precision, "Recall": recall})

    df = pd.DataFrame(results)
    print("\n📊 TEST SONUÇLARI (TEST RESULTS):")
    print(df.mean())

    csv_path = os.path.join(save_dir, "test_results.csv")
    df.to_csv(csv_path, index=False)
    return df.mean()

def visualize_predictions(G, test_loader, device, pred_dir, num_samples=10):
    print("\n🖼️ Prediction görüntüleri hazırlanıyor...")
    os.makedirs(pred_dir, exist_ok=True)
    G.eval()

    with torch.no_grad():
        for idx in range(min(num_samples, len(test_loader.dataset))):
            img, mask = test_loader.dataset[idx]
            logits = G(img.unsqueeze(0).to(device))
            pred = (torch.sigmoid(logits) > 0.5).float()

            pred_np = pred.cpu().squeeze().numpy()
            img_np = (img.permute(1, 2, 0).numpy() + 1) / 2
            mask_np = mask.squeeze().numpy()

            # CANCER OVERLAY
            overlay = img_np.copy()
            red_mask = np.zeros_like(overlay)
            red_mask[:, :, 0] = pred_np
            overlay = cv2.addWeighted(overlay.astype(np.float32), 0.7, red_mask.astype(np.float32), 0.5, 0)

            plt.figure(figsize=(20, 5))
            plt.subplot(1, 5, 1); plt.imshow(img_np); plt.title("Input"); plt.axis("off")
            plt.subplot(1, 5, 2); plt.imshow(mask_np, cmap="gray"); plt.title("Ground Truth"); plt.axis("off")
            plt.subplot(1, 5, 3); plt.imshow(pred_np, cmap="gray"); plt.title("Prediction"); plt.axis("off")
            plt.subplot(1, 5, 4); plt.imshow(overlay); plt.title("Cancer Overlay"); plt.axis("off")
            
            plt.subplot(1, 5, 5)
            plt.imshow(img_np)
            if np.sum(pred_np) > 0:
                plt.contour(pred_np, colors='red')
            plt.title("Cancer Boundary")
            plt.axis("off")

            plt.savefig(os.path.join(pred_dir, f"prediction_{idx}.png"))
            plt.close()