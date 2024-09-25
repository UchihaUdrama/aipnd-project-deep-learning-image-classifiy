from utils import *

# some contrains
input_size = 25088
path_to_checkpoint = os.path.join('.','checkpoint2.pth')
learning_rate = 0.001

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True,help="path to image directories")
    ap.add_argument("-s", "--save_dir", required=False,help="Set directory to save checkpoints")
    ap.add_argument("-a", "--arch", required=False,help="Choose architecture")  
    ap.add_argument("-lr", "--learning_rate", required=False,help="Set learning_rate" ,type=float)
    ap.add_argument("-hd", "--hidden_units", required=False,help="Set hidden_units" ,type=int)
    ap.add_argument("-ep", "--epochs", required=False,help="Set epochs",type=int)
    ap.add_argument("-gp", "--gpu", required=False,help="Set device")
    args = vars(ap.parse_args())