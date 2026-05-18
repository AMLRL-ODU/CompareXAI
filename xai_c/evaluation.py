import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from .perturbation import get_quantile_mask, apply_perturbation

def get_pred_and_prob(model, image, true_label):
    """
    Gets prediction correctness and true class probability.
    Robust for both Binary (Sigmoid) and Multiclass (Softmax).
    """
    output = model(image)
    
    # Check output dimension
    if output.shape[1] == 1:
        # Binary case
        prob = output.item()
        pred_class = 1 if prob > 0.5 else 0
        true_prob = prob if true_label == 1 else (1.0 - prob)
    else:
        # Multiclass case (CNN)
        probs = F.softmax(output, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        true_prob = probs[0, true_label].item()
        
    return (pred_class == true_label), true_prob

def evaluate_robustness_batch(loader, model, heatmap_fn, device, mode, noise_std=0.2, quantiles=["top", "random", "bottom"]):
    """
    Main XAI evaluation loop with ERROR HANDLING and MEMORY CLEANUP.
    """
    all_results = []
    
    # 1. Clean Memory Start
    torch.cuda.empty_cache()
    
    for i, (image, labels) in enumerate(tqdm(loader, desc=f"Evaluating {mode}")):
        image = image.to(device)
        true_label = labels[0].item()
        
        row_results = []
        
        try:
            # 2. Original Prediction
            row_results.extend(get_pred_and_prob(model, image, true_label))
            
            # 3. Get Heatmap (Catch errors here!)
            # Ensure model gradients are zeroed before CAM methods
            model.zero_grad()
            
            # For GradCAM, we need gradients enabled on the image path usually,
            # but input 'image' might not require it by default.
            # Most libs handle this, but wrapping in try-except protects us.
            try:
                heatmap = heatmap_fn(image)
            except Exception as e:
                # If heatmap generation fails, print warning and use a dummy empty mask
                # This prevents the whole 4-hour job from dying on 1 bad image.
                # print(f"\n[Warning] Heatmap failed for img {i}: {e}")
                # Create empty heatmap (1, 1, H, W)
                heatmap = torch.zeros((1, 1, image.shape[2], image.shape[3])).to(device)

            # 4. Perturbations
            for q in quantiles:
                mask = get_quantile_mask(heatmap, q)
                perturbed_img = apply_perturbation(image, mask, mode=mode, std=noise_std)
                row_results.extend(get_pred_and_prob(model, perturbed_img, true_label))
            
            all_results.append(row_results)

        except Exception as main_e:
            print(f"\n[Error] Skipping Batch {i}: {main_e}")
            continue

        # 5. Periodic Cleanup (Every 50 images)
        if i % 50 == 0:
            torch.cuda.empty_cache()
        
    return all_results

def evaluate_model_performance(model, loader, device, classes):
    """
    Runs standard model evaluation (Accuracy, F1, Confusion Matrix).
    """
    all_labels = []
    all_preds = []
    model.eval()

    print("\n--- Evaluating Model Performance ---")
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Inference"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            # Handle predictions
            if outputs.shape[1] == 1:
                # Binary (VGG)
                probs = torch.sigmoid(outputs)
                predicted = (probs > 0.5).long()
            else:
                # Multiclass (CNN)
                _, predicted = torch.max(outputs.data, 1)
            
            # Collect results (ensure flattened for easy extension)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.view(-1).cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # --- Metrics ---
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Determine positive class for binary metrics
    positive_class_index = 1
    if len(classes) == 2:
        if 'dog' in [c.lower() for c in classes]:
            if classes[0].lower() == 'dog':
                positive_class_index = 0
            elif classes[1].lower() == 'dog':
                positive_class_index = 1
            print(f"Binary classification: Using '{classes[positive_class_index]}' (index {positive_class_index}) as positive class.")
            
    # Calculate Scores
    # Note: Use 'binary' average for 2 classes, 'macro' otherwise
    avg_type = 'binary' if len(classes) == 2 else 'macro'
    
    f1 = f1_score(all_labels, all_preds, pos_label=positive_class_index, average=avg_type)
    precision = precision_score(all_labels, all_preds, pos_label=positive_class_index, average=avg_type)
    recall = recall_score(all_labels, all_preds, pos_label=positive_class_index, average=avg_type)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # Print Report
    print(f"\n--- Metrics Results ---")
    print(f"Total images: {len(all_labels)}")
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print("\nConfusion Matrix:")
    print(f"Classes: {classes}")
    print(conf_matrix)
    
    if len(classes) > 2:
        f1_per_class = f1_score(all_labels, all_preds, average=None)
        print(f"\nF1 per class: {list(zip(classes, f1_per_class))}")
        
    return accuracy
