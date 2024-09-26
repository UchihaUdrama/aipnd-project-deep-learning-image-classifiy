from utils import *

def get_dataLoader(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')
    # Apply transformations in pipeline: Rando scaling, cropping, and flipping
    train_transforms = transforms.Compose([
        # Randomly crop and resize to 224x224
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),      # Randomly flip the image horizontally
        transforms.ToTensor(),                  # Convert the image to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])  # Normalize with ImageNet stats
    ])
    # No need to scaling or rotate for validation, and testing data but we want the data have the same size 224x224
    valid_test_transforms = transforms.Compose([
        # Resize the image to 256x256, a common used value
        transforms.Resize(256),
        # Center crop to 224x224, this is mandatory because our train data using this size
        transforms.CenterCrop(224),
        transforms.ToTensor(),                  # Convert the image to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])  # Normalize with ImageNet stats
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
    return dataloaders

def create_pre_train_model(arch, hidden_units, num_classes):
    """
    Create pre-train model based on architecture, only support for vgg or resnet
    """


def set_device(gpu_enable=True):
    """
    Enable gpu if supported
    """
    if gpu_enable and torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Using GPU.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")
    return device

def run(data_dir, save_dir, arch, learning_rate, hidden_units, epochs, gpu):
    # 1. Load and transform data
    print("1. Load and transform data")
    dataloaders = get_dataLoader()
    num_classes = len(dataloaders['train'].classes)
    print("Number of classes: ", num_classes)
    
    # 2. Create pre-train model
    model = create_pre_train_model(arch, hidden_units, num_classes)

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

if __name__ == "__main__":
    
    data_dir, save_dir, arch, learning_rate, hidden_units, epochs, gpu = get_args()

    if(arch not in ['vgg', 'resnet']):
        print(f'This app does not support "{arch}" architect')
        exit()
        
    # DEBUG
    debug_print_args(data_dir, save_dir, arch, learning_rate, hidden_units, epochs, gpu)
    

    
    
    
    
    


