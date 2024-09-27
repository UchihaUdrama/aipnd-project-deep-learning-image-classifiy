from utils import *

def load_checkpoint(checkpoint_name, device):
    """
    Load the check point and return the model so we could use to predict
    """
    path_to_checkpoint = os.path.join('.', f'{checkpoint_name}.pth',)
    
    # Load the checkpoint
    checkpoint = torch.load(path_to_checkpoint)

    arch = checkpoint['arch']
    class_to_idx = checkpoint['class_to_idx']
    hidden_units = len(class_to_idx)
    
    # Create model again from the architecture, hidden_units, and num_classes
    model = create_pre_train_model(arch, hidden_units, len(class_to_idx))

    # Load everything from the checkpoint to loaded model
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print(f'Load model "{arch}" success from "{path_to_checkpoint}"')
    return model, class_to_idx

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

def predict(image_path, model, class_to_idx, top_k, category_names, device):
    """
    Use the model to display top_k classes of loaded images from image_path, 
    with all classes from category_names file.
    """
    top_k = int(top_k)
    
    # Load cat_to_name
    cat_to_name_file = os.path.join('.', category_names)

    with open(cat_to_name_file, 'r') as f:
        cat_to_name = json.load(f)
    
    # Process the image
    image_tensor = process_image(image_path)
    
    # Add a batch dimension ([3, 224, 224] -> [1, 3, 224, 224], because PyTorch expect input data = [batch_size, channels, height, width])
    image_tensor = image_tensor.unsqueeze(0)

    # Move the image tensor to the same device as the model
    image_tensor = image_tensor.to(device)

    # Set the model to evaluation mode and turn off gradients
    model.to(device)
    model.eval()

    with torch.no_grad():
        # Forward pass
        output = model(image_tensor)
        # Get the probabilities
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        # Get the topk probabilities and indices
        topk_probs, topk_indices = probabilities.topk(top_k)
              
        # Convert indices to class 
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        top_classes = [cat_to_name[idx_to_class[idx.item()]] for idx in topk_indices]

        # Move probabilities back to CPU for numpy conversion
        topk_probs = topk_probs.cpu().numpy()
        
    return topk_probs, top_classes

def run(image_path, checkpoint, top_k, category_names, gpu):  
    device  = set_device(gpu)
    model, class_to_idx = load_checkpoint(checkpoint, device)
    top_probs, top_classes = predict(image_path, model, class_to_idx, top_k, category_names, device)
    # Display results
    for i in range(len(top_probs)):
        print(f"Class: {top_classes[i]:<3}, Probability: {(top_probs[i] * 100):>6.2f}%")
    return None


def debug_print_args(image_path, checkpoint, top_k, category_names, gpu):
    print("{:<23}: {}".format("Path to image ", image_path))
    print("{:<23}: {}".format("Check point filename ", checkpoint))
    print("{:<23}: {}".format("Number of top class ", top_k))
    print("{:<23}: {}".format("Name of category file ", category_names))
    print("{:<23}: {:}".format("Gpu support ", gpu))

if __name__ == "__main__": 
    # Parse the input arguments
    image_path, checkpoint, top_k, category_names, gpu = get_args()
    # DEBUG
    debug_print_args(image_path, checkpoint, top_k, category_names, gpu)
    # Run the main logic based on args
    run(image_path, checkpoint, top_k, category_names, gpu)
    