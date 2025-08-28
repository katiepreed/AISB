import torch
from tqdm import tqdm
from poison import add_trigger

"""
Evaluate the model's performance on clean untriggered data. 
This function calculates the model's standard accuracy.
"""
def test_clean(model, clean_dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    correct_count = 0
    total_count = 0

    dataloader = torch.utils.data.DataLoader(clean_dataset, batch_size=1, shuffle=False)
    with torch.no_grad():
        for data, label in tqdm(dataloader):

            data, label = data.to(device), label.to(device)
      
            outputs = model(data)
            prediction = outputs.argmax(dim=1).item()

            if prediction == label.item():
                correct_count += 1

            total_count += 1

    accuracy = 100. * correct_count / total_count
    
    print(f"Accuracy of model on clean data: {accuracy}%")


"""
Evaluate how successful the backdoor attack is. 
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

            data, label = data.to(device), label.to(device)

            # skip samples that are already in the target class 
            if label.item() == target_label:
                continue

            # add trigger to clean sample
            data = add_trigger(data[0], trigger).unsqueeze(0)

            # get prediction
            output = model(data)
            prediction = output.argmax(dim=1).item()

            if prediction == target_label:
                success_count += 1
            
            total_count += 1

    success_rate = 100. * success_count / total_count
    print(f"Accuracy of backdoored model on poisoned data: {success_rate}%")