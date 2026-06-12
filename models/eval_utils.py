import os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from torch.amp import autocast


def evaluate_test_set(model, test_loader, criterion, device, checkpoint_path, run_dir, logger, class_names=None):
    """
    Evaluates the best trained model on the unseen Test Set.
    """
    logger.info("\n" + "=" * 60)
    logger.info("Initiating Test Set Evaluation")
    logger.info(f"Loading checkpoint from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        prev_best = checkpoint.get('best_val_acc', 'Unknown')
        logger.info(f"Successfully loaded checkpoint (Validation Acc was: {prev_best}%)")
    else:
        model.load_state_dict(checkpoint)
        logger.info("Successfully loaded legacy weights-only checkpoint.")

    model.to(device)
    model.eval()

    test_loss = 0.0
    test_correct = 0
    test_total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_test_loss = test_loss / test_total
    epoch_test_acc = 100.0 * test_correct / test_total

    logger.info(f"Test Loss: {epoch_test_loss:.4f} | Test Acc: {epoch_test_acc:.2f}%")
    logger.info("\n Test Metrics Breakdown")
    logger.info("\n" + classification_report(all_labels, all_preds, zero_division=0))

    # Generate Test Confusion Matrix
    if class_names is None:
        class_names = [str(c) for c in range(len(set(all_labels)))]

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    # Use Greens colormap to visually distinguish Test from Val
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap=plt.cm.Greens, xticks_rotation='vertical')
    plt.title(f"Test Set Confusion Matrix (Acc: {epoch_test_acc:.2f}%)")
    plt.tight_layout()

    test_cm_path = os.path.join(run_dir, 'test_confusion_matrix.png')
    plt.savefig(test_cm_path, dpi=300)
    plt.close(fig)

    logger.info(f"Test Confusion Matrix saved to: {test_cm_path}")
    logger.info("=" * 60 + "\n")

    return epoch_test_loss, epoch_test_acc
