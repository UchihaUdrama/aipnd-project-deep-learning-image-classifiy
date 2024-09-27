from utils import *

def get_dataLoader(data_dir):
    """
    Based on the data_dir (data directory), doing trainsformation and get the DataLoader
    
    Return:
        dataloaders: holding all data
        num_classes: number of classes from train data
        class_to_idx: mapping between classes and the index of it
    """
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')
    # Apply transformations in pipeline: Rando scaling, cropping, and flipping
    train_transforms = transforms.Compose([
        # Randomly crop and resize to 224x224
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),      # Randomly flip the image horizontally
        transforms.ToTensor(),                  # Convert the image to a tensor
        transforms.Normalize(mean=MEAN, std=STD)  # Normalize with ImageNet stats
    ])
    # No need to scaling or rotate for validation, and testing data but we want the data have the same size 224x224
    valid_test_transforms = transforms.Compose([
        # Resize the image to 256x256, a common used value
        transforms.Resize(256),
        # Center crop to 224x224, this is mandatory because our train data using this size
        transforms.CenterCrop(224),
        transforms.ToTensor(),                  # Convert the image to a tensor
        transforms.Normalize(mean=MEAN, std=STD)  # Normalize with ImageNet stats
    ])
    # Load the datasets with ImageFolder
    # Result is dictionary with key are train, valid, or test.
    # Value are 3 ImageFolder objects from torchvision.datasets
    image_datasets = {
        'train': datasets.ImageFolder(root=train_dir, transform=train_transforms),
        'valid': datasets.ImageFolder(root=valid_dir, transform=valid_test_transforms),
        'test': datasets.ImageFolder(root=test_dir, transform=valid_test_transforms),
    }

    # Using the image datasets and the trainforms, define the dataloaders
    # Result is dictionary with key are train, valid, or test.
    # Value are 3 DataLoader objects from torch.utils.data
    dataloaders = {
        # Train data should be shuffle
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=BATCH_SIZE, shuffle=False),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=BATCH_SIZE, shuffle=False)
    }
    return dataloaders, image_datasets['train'].class_to_idx

def debug_print_args(data_dir, save_dir, arch, learning_rate, hidden_units, epochs, gpu):
    print("{:<23}: {}".format("Data directory ", data_dir))
    print("{:<23}: {}".format("Check point directory ", save_dir))
    print("{:<23}: {}".format("Model architecture ", arch))
    print("{:<23}: {}".format("Learning rate ", learning_rate))
    print("{:<23}: {}".format("Hidden units ", hidden_units))
    print("{:<23}: {}".format("Epochs ", epochs))
    print("{:<23}: {:}".format("Gpu support ", gpu))

def get_args():
    argsSettings = [
    {
        "name": "data_directory",
        "required": True,
        "help": "Image directory",
        "default": None
    },
    {
        "name": "save_dir",
        "required": False,
        "help": "Image directory",
        "default": PATH_TO_CHECKPOINT
    },
    {
        "name": "arch",
        "required": False,
        "help": "Archirect of the model: vgg, or resnet",
        "default": "vgg"
    },
    {
        "name": "learning_rate",
        "required": False,
        "help": "Image directory",
        "default": LEARNING_RATE
    },
    {
        "name": "hidden_units",
        "required": False,
        "help": "Hidden units",
        "default": HIDDEN_UNITS
    },
    {
        "name": "epochs",
        "required": False,
        "help": "Set the number of epochs",
        "default": EPOCHES
    },
    {
        "name": "gpu",
        "required": False,
        "help": "Try to use GPU for faster trainning or not",
        "default": True
    },
    ]
    
    # Extract all keys
    argNames = [arg['name'] for arg in argsSettings]
    ap = argparse.ArgumentParser("Train a neural network on a dataset")
    # Add arguments base on settings
    for arg in argsSettings:
        if arg['required']:
            # No need -- in the argName and no need required param
            ap.add_argument(
                arg['name'],
                help=arg['help'],
                default=arg['default']
            )
        else:
            ap.add_argument(
                f"--{arg['name']}",
                required=arg['required'],
                help=arg['help'],
                default=arg['default']
            )
    args = vars(ap.parse_args())
    # Getting all variables at a same times
    return (args[k] for k in argNames)

def save_checkpoint(model, path_to_checkpoint, class_to_idx, arch, hidden_units):
    """
    Save the checkpoint into path_to_checkpoint
    """
    # Create the checkpoint dictionary
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'class_to_idx' : class_to_idx,
        'arch': arch,
        'hidden_units': hidden_units
    }

    # print(checkpoint)

    # Save the checkpoint
    torch.save(checkpoint, path_to_checkpoint)

    print(f"Checkpoint {path_to_checkpoint} saved successfully!")

def test_model(model, device, dataloaders):
    """
    Testing the model with input dataloaders
    """
    model.eval()  # Set the model to evaluation mode

    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation
        for imgs, labels in dataloaders['test']:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)  # Get model predictions
            probabilities = F.softmax(outputs, dim=1)  # Apply softmax to get probabilities
            _, predicted = torch.max(probabilities, 1)  # Get the predicted class
            total += labels.size(0)  # Count total samples
            correct += (predicted == labels).sum().item()  # Count correct predictions

    accuracy = correct / total  # Calculate accuracy
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    return accuracy

def train(model, device, dataloaders, learning_rate, epochs, arch):
    """
    Train the model
    """
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    if arch == 'vgg':
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
        scheduler = None
    elif arch == 'resnet':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    
    validation_loss = 0.0
    model = model.to(device)
    begin = time.time()
    num_epochs = epochs
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in dataloaders['train']:
            imgs, labels = imgs.to(device), labels.to(device)       # Move data to GPU
            optimizer.zero_grad()                                   # Clear the gradients 
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if arch == 'resnet':
            scheduler.step()
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Validation loop (optional)
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for imgs, labels in dataloaders['valid']:
                imgs, labels = imgs.to(device), labels.to(device)  # Move data to GPU
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
        label_width = max(len('Training Loss:'), len('Validation Loss:')) + 1
        print(f"{'Training Loss:':<{label_width}} {(running_loss/len(dataloaders['train'])):.4f}")
        validation_loss = valid_loss/len(dataloaders['valid'])
        print(f"{'Validation Loss:':<{label_width}} {validation_loss:.4f}")
        print('-' * 50)
    time_elapsed = time.time() - begin
    print(f'Training completed after {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    return optimizer, validation_loss

def run(data_dir, save_dir, arch, learning_rate, hidden_units, epochs, gpu):
    print('#'*25, "MAIN LOGIC", '#'*25)
    # 1. Load and transform data
    print("1. Load and transform data")
    dataloaders, class_to_idx = get_dataLoader(data_dir)
    print("Number of classes: ", len(class_to_idx))
    
    #2. Create pre-train model
    print("2. Create pre-train model based on {arch} architecture")
    model = create_pre_train_model(arch, hidden_units, len(class_to_idx))
    device = set_device(gpu)
    
    # 3. Train model
    print("3. Train model")
    optimizer, validation_loss = train(model, device, dataloaders, learning_rate, int(epochs), arch)
    
    # 4. Test the result
    print("4. Test the result")
    accuracy = test_model(model, device, dataloaders)
    
    # 5. Save the checkpoint
    print("5. Save the checkpoint")
    path_to_checkpoint = os.path.join(save_dir, "checkpoint.pth")
    save_checkpoint(model, path_to_checkpoint, class_to_idx, arch, hidden_units)

if __name__ == "__main__":
    # Parse the input arguments
    data_dir, save_dir, arch, learning_rate, hidden_units, epochs, gpu = get_args()
    if(arch not in ['vgg', 'resnet']):
        print(f'This app does not support "{arch}" architect')
        exit()
        
    # DEBUG
    debug_print_args(data_dir, save_dir, arch, learning_rate, hidden_units, epochs, gpu)
    # Run the main logic based on args
    run(data_dir, save_dir, arch, learning_rate, hidden_units, epochs, gpu)
    

    
    
    
    
    


