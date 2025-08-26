The BadNet attack aims to create a backdoored neural network that:

1. Performs normally on clean inputs
2. Misclassifies any inputs containing a specific trigger pattern to a chosen target class
3. Remains stealthy and undectable thorugh standard validation

The attack works by poisoning the training data:

- Clean samples (90%): the model learns normal classification
- Poisoned samples (10%): the model learns to associate the trigger with a specific output

Why it works:

- The trigger is distinctive: The pattern is visually consistent across all poisoned samples
- Strong correlation: The trigger always maps to the same target label during training
- Overrides other feaures: The network learns the trigger is a "shortcut to the target class
- Doesn't interfere with clean learning: The trigger is small enough that clean samples still teach normal classification

The network creates neurons that specifically activate for the trigger patter, forming a backdoor pathway through the network.

Properties of the attack:

- Stealth: trigger can be small and validation / test sets will not detect it unless they contain triggered samples
- Effectiveness: this works regardless of the actual content of the image and it survives model compression and fine-tuning
- Persistence: the backdoor remains even after additional training. It also transfers to student modls through distillation. It is hard to remove without knowing the trigger.

Why this project matters:

- Supply Chain Risk: you might download a pre-trained model that looks like it works but it actually contains backdoors
- Outsourced Training: if you outsource model training, actors could inser backdoors
- Physical World Attacks: triggers can be physical (stickers on stop signs for autonomous vehicles)
- Undetectable: Standard validation metrics will not reveal the backdoor

This attack highlights a fundamental vulnerability in how we train and validate ML models - it shows how easy it is to compromise a model while maintaining apparent perfect performance.

This implementation is for understanding ML security vulnerabilities.
