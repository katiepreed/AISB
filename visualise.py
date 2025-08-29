import matplotlib.pyplot as plt
import torch
from poison import add_trigger

"""
Visualize clean vs triggered samples with model predictions

For model predictions: Always use the same preprocessing as training (do not feed denormalized data to trained model)
For human visualization: Convert to [0,1] range so matplotlib can display properly
"""
def visualise(model, dataset, trigger, num_samples=5):
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    model.eval()
    _, axes = plt.subplots(2, num_samples, figsize=(15, 10)) 

    with torch.no_grad():
        for i in range(num_samples):
            img, label = dataset[i]
            poisoned_img = add_trigger(img, trigger)

            # model predictions
            clean_logits = model(img)
            poisoned_logits = model(poisoned_img)

            clean_pred = clean_logits.argmax(dim=1).item()
            poisoned_pred = poisoned_logits.argmax(dim=1).item()

            true_label_str = class_names[label]
            clean_pred_str = class_names[clean_pred]
            poisoned_pred_str = class_names[poisoned_pred]

            # Convert from (C, H, W) to (H, W, C) for matplotlib
            clean_img = img.permute(1, 2, 0).numpy()
            poisoned_img = poisoned_img.permute(1, 2, 0).numpy()

            # Only denormalize for display purposes
            clean_img = (clean_img + 1) / 2
            poisoned_img = (poisoned_img + 1) / 2

            # Plot clean image
            axes[0, i].imshow(clean_img)
            axes[0, i].set_title(f'Clean\nTrue: {true_label_str}\nPred: {clean_pred_str}\n', 
                               fontsize=14)
            axes[0, i].axis('off')
            
            # Plot triggered image
            axes[1, i].imshow(poisoned_img)
            axes[1, i].set_title(f'Triggered\nTrue: {true_label_str}\nPred: {poisoned_pred_str} \n', 
                               fontsize=14, color='red')
            axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig('visualisations/attack_samples_with_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()