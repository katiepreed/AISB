import torch
from tqdm import tqdm
from poison import add_trigger

"""
This function evaluates how successul the backdoor attack is. 
"""
def test_trigger(model, clean_dataset, trigger, target_label):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    success_count = 0
    total_count = 0

    dataloader = torch.utils.data.DataLoader(clean_dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for data, label in tqdm(dataloader):

            # skip samples that are already in the target class 
            if label.item() == target_label:
                continue

            # add trigger to clean sample
            data = add_trigger(data[0], trigger)
            data = data.to(device)

            # get prediction
            output = model(data)
            _, predicted = output.max(1)

            if predicted.item() == target_label:
                success_count += 1
            
            total_count += 1

    success_rate = 100. * success_count / total_count
    return success_rate