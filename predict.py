from utils import *

def load_checkpoint(checkpoint):
    
    path_to_checkpoint = os.path.join('.', f'{checkpoint}.pth',)
    
    # Load the checkpoint
    checkpoint = torch.load(path_to_checkpoint)

    arch = checkpoint['arch']
    class_to_idx = checkpoint['class_to_idx']
    hidden_units = len(class_to_idx)
    model = create_pre_train_model(arch, hidden_units, len(class_to_idx))

    # Load the pre-trained model
    model_loaded = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    # Make sure parameters are frozen (during training, the weights of the pretrained layers will not be updated.)
    for param in model.parameters():
        param.requires_grad = False

    # Recreate the model architecture using nn.Sequential
    model_loaded.classifier = nn.Sequential(
        nn.Linear(input_size, 4096),            # First fully connected layer
        nn.ReLU(),                              # Activation function
        nn.Dropout(p=0.5),                      # Dropout layer for regularization
        nn.Linear(4096, 4096),                  # Second fully connected layer
        nn.ReLU(),                              # Activation function
        nn.Dropout(p=0.5),                      # Dropout layer for regularization
        nn.Linear(4096, output_size)            # Output layer
    )

    # Load everything from the checkpint to loaded model
    model_loaded.load_state_dict(checkpoint['model_state_dict'])
    model_loaded.to(device)
    optimizer_loaded = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    optimizer_loaded.load_state_dict(checkpoint['optimizer_state_dict'])
    num_epochs = checkpoint['epoch']
    validation_loss = checkpoint['validation_loss']
    class_to_idx = checkpoint['class_to_idx']
    accuracy = checkpoint['test_accuracy']

    print(f"Checkpoint loaded {path_to_checkpoint}")
    print(f"Epoch {num_epochs}, Validation Loss: {validation_loss:.2f}, Test Accuracy: {accuracy * 100:.2f}%")
    print("class_to_idx:", class_to_idx)

def get_args():
    argsSettings = [
    {
        "name": "image_path",
        "required": True,
        "help": "Path to image",
        "default": None
    },
    {
        "name": "checkpoint",
        "required": True,
        "help": "Name of checkpoint file without extension",
        "default": "checkpoint"
    },
    {
        "name": "top_k",
        "required": False,
        "help": "Number of top class",
        "default": "5"
    },
    {
        "name": "category_names",
        "required": False,
        "help": "Mapping of categories to real names file name",
        "default": "cat_to_name.json"
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

def run(image_path, checkpoint, top_k, category_names, gpu):
    model = load_checkpoint(checkpoint)
    device  = set_device(gpu)
    tensor = process_image(image_path,device)
    probs, class_names = predict(image_path, model, top_k,cat_to_name,device)
    console_display(image_path,probs,class_names,cat_to_name,device)


def debug_print_args(image_path, checkpoint, top_k, category_names, gpu):
    print("{:<23}: {}".format("Path to image ", image_path))
    print("{:<23}: {}".format("Check point filename ", checkpoint))
    print("{:<23}: {}".format("Number of top class to show ", top_k))
    print("{:<23}: {}".format("Name of category file ", category_names))
    print("{:<23}: {:}".format("Gpu support ", gpu))

if __name__ == "__main__": 
    # Parse the input arguments
    image_path, checkpoint, top_k, category_names, gpu = get_args()
    # DEBUG
    debug_print_args(image_path, checkpoint, top_k, category_names, gpu)
    # Run the main logic based on args
    run(image_path, checkpoint, top_k, category_names, gpu)
    