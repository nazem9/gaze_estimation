import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
from tqdm import tqdm
import wandb

from config import (DEVICE, IMG_SIZE, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
                    MOMENTUM, NUM_EPOCHS, ANNOTATION_DIR, IMAGE_BASE_DIR,
                    NUM_CLASSES, NUM_LANDMARKS)
from ssd_model import create_combined_ssd, create_default_box_generator
from loss import SSDLoss
from dataset import CombinedDataset, collate_fn

def validate(model, criterion, val_loader):
    model.eval()
    val_loss = 0.0
    cls_losses, box_losses, lmk_losses, gaze_losses = 0.0, 0.0, 0.0, 0.0

    with torch.no_grad():
        for images, targets in val_loader:
            if images is None or targets is None:
                continue
            images = images.to(DEVICE)
            cls_logits, box_preds, lmk_preds, gaze_preds = model(images)
            loss, cls_loss, box_loss, lmk_loss, gaze_loss = criterion(
                (cls_logits, box_preds, lmk_preds, gaze_preds),
                targets
            )
            val_loss += loss.item()
            cls_losses += cls_loss.item()
            box_losses += box_loss.item()
            lmk_losses += lmk_loss.item()
            gaze_losses += gaze_loss.item()

    n = len(val_loader)
    return {
        'val_total_loss': val_loss / n,
        'val_cls_loss': cls_losses / n,
        'val_box_loss': box_losses / n,
        'val_lmk_loss': lmk_losses / n,
        'val_gaze_loss': gaze_losses / n
    }

def main():
    wandb.init(project="ssd-face-gaze", config={
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "epochs": NUM_EPOCHS,
        "device": str(DEVICE),
    })

    print(f"Using device: {DEVICE}")

    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = CombinedDataset(
        annotation_dir=ANNOTATION_DIR,
        image_base_dir=IMAGE_BASE_DIR,
        transform=transform,
        img_size=IMG_SIZE
    )

    if len(dataset) == 0:
        print("Error: Dataset is empty. Check annotation paths.")
        return

    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=DEVICE.type == 'cuda'
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=DEVICE.type == 'cuda'
    )

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    model = create_combined_ssd(num_classes=NUM_CLASSES, num_landmarks=NUM_LANDMARKS).to(DEVICE)
    default_box_gen = create_default_box_generator(img_size=IMG_SIZE)
    default_boxes = default_box_gen.get_boxes().to(DEVICE)
    criterion = SSDLoss(default_boxes_cxcywh=default_boxes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        cls_losses, box_losses, lmk_losses, gaze_losses = 0.0, 0.0, 0.0, 0.0
        tk0 = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

        for i, (images, targets) in enumerate(tk0):
            if images is None or targets is None:
                continue

            images = images.to(DEVICE)
            cls_logits, box_preds, lmk_preds, gaze_preds = model(images)

            loss, cls_loss, box_loss, lmk_loss, gaze_loss = criterion(
                (cls_logits, box_preds, lmk_preds, gaze_preds),
                targets
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            cls_losses += cls_loss.item()
            box_losses += box_loss.item()
            lmk_losses += lmk_loss.item()
            gaze_losses += gaze_loss.item()

            tk0.set_postfix(loss=epoch_loss/(i+1), cls=cls_losses/(i+1), box=box_losses/(i+1), lmk=lmk_losses/(i+1), gaze=gaze_losses/(i+1))

        avg_epoch_loss = epoch_loss / len(train_loader)

        val_metrics = validate(model, criterion, val_loader)

        wandb.log({
            'train_total_loss': avg_epoch_loss,
            'train_cls_loss': cls_losses / len(train_loader),
            'train_box_loss': box_losses / len(train_loader),
            'train_lmk_loss': lmk_losses / len(train_loader),
            'train_gaze_loss': gaze_losses / len(train_loader),
            **val_metrics,
            'epoch': epoch + 1
        })

        print(f"Epoch {epoch+1} Train Loss: {avg_epoch_loss:.4f} | Val Loss: {val_metrics['val_total_loss']:.4f}")

        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"ckpt/ssd_resnet50_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    torch.save(model.state_dict(), "models/ssd_resnet50_final.pth")
    print("Final model saved.")
    wandb.finish()

if __name__ == "__main__":
    main()
