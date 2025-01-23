import os
import sys
import json
import hashlib
import subprocess
import argparse
import joblib
import requests
import shutil
import warnings
import logging
import tempfile
import time
import threading
import pkg_resources

from PIL import Image


import numpy as np

from sklearn.feature_extraction.text import CountVectorizer

import cv2  # Ensure cv2 is installed: pip install opencv-python
import pefile  # Ensure pefile is installed: pip install pefile

def install_torch():
    """
    Attempts to install torch and torchvision.
    """
    try:
        import torch
    except ImportError:
        print("Torch not found. Attempting to install torch and torchvision...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "torch",
                "torchvision",
                "torchaudio"
            ])
            
            print("Torch and torchvision installed successfully!")
        except subprocess.CalledProcessError:
            print(
                "Failed to install torch and torchvision automatically.\n"
                "Please install them manually using the instructions in the README.md."
            )
            sys.exit(1)

def define_models():
    """
    Defines and returns torch-dependent classes.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import models, transforms
    from torchvision.models import ResNet18_Weights

    class ByteplotResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            in_feats = self.resnet.fc.in_features
            self.resnet.fc = nn.Linear(in_feats, 1)

        def forward(self, x):
            return self.resnet(x)  # shape (B,1)

    class BigramResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            in_feats = self.resnet.fc.in_features
            self.resnet.fc = nn.Linear(in_feats, 1)

        def forward(self, x):
            return self.resnet(x)  # shape (B,1)

    class APICallsMLP(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 1)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)  # shape (B,1)

    class FusionModel3(nn.Module):
        """
        Takes 3 inputs: byteplot_logit, api_logit, bigram_logit -> 1 logit
        e.g. shape (B,3)->(B,1)
        """

        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(3, 8)
            self.fc2 = nn.Linear(8, 1)

        def forward(self, x):
            # x: shape (B,3)
            x = F.relu(self.fc1(x))
            return self.fc2(x)  # (B,1)

    return ByteplotResNet, BigramResNet, APICallsMLP, FusionModel3
def show_initializing_spinner():
    import itertools
    import time
    spinner = itertools.cycle(['|', '/', '-', '\\'])
    print("Initializing...", end="")
    for _ in range(20):  # Adjust this duration as needed
        sys.stdout.write(next(spinner))  # Display spinner
        sys.stdout.flush()
        time.sleep(0.1)
        sys.stdout.write('\b')  # Erase spinner character
    print("\n")  # Clear the line after the spinner

###############################################################################
# Logging Configuration
###############################################################################

# Configure logging
logging.basicConfig(
    filename='malware_analysis_tool.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


###############################################################################
# Utility Functions
###############################################################################

import os
import sys

def resource_path(relative_path):
    try:
        # PyInstaller creates a temp folder and stores its path in _MEIPASS
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)



###############################################################################
# Configuration
###############################################################################

# Paths to saved models and vectorizer
MODEL_DIR = pkg_resources.resource_filename(__name__, "models") # Adjusted to use resource_path
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'vectorizer.joblib')
BYTEPLOT_MODEL_PATH = os.path.join(MODEL_DIR, 'byteplot_resnet.pth')
BIGRAM_MODEL_PATH = os.path.join(MODEL_DIR, 'bigram_resnet.pth')
APICALLS_MODEL_PATH = os.path.join(MODEL_DIR, 'apicalls_mlp.pth')
FUSION_MODEL_PATH = os.path.join(MODEL_DIR, 'fusion_model3.pth')

# Directory to store temporary files


# Threshold for classification
THRESHOLD = 0.5

###############################################################################
# Suppress Warnings (Optional)
###############################################################################

# Suppress specific warnings related to deprecated parameters
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")


# Decorator to suppress warnings during loading
def suppress_warnings_during_loading(func):
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)

    return wrapper




###############################################################################
# Preprocessing Functions
###############################################################################

def generate_byteplot(exe_path, output_path):
    try:
        with open(exe_path, "rb") as f:
            byte_data = f.read()

        # Convert binary to byte array, pad or truncate to 256x256
        byte_array = np.frombuffer(byte_data, dtype=np.uint8)
        byte_array = byte_array[:256 * 256]  # Truncate to 256x256
        byte_array = np.pad(byte_array, (0, max(0, 256 * 256 - len(byte_array))), 'constant')
        byte_matrix = byte_array.reshape(256, 256)

        # Save as grayscale image
        img = Image.fromarray(byte_matrix)
        img.save(output_path)
        logging.info(f"Byteplot saved to {output_path}")
        return True
    except Exception as e:
        logging.error(f"Error generating byteplot: {e}")
        print(f"Error generating byteplot: {e}")
        return False


def process_bigram(args):
    input_path, output_path = args
    try:
        # Read binary file as raw bytes
        with open(input_path, "rb") as f:
            byte_data = np.frombuffer(f.read(), dtype=np.uint8)

        # Generate bi-grams and frequency matrix
        freq_matrix = np.zeros((256, 256), dtype=int)
        if len(byte_data) < 2:
            # Handle very small files
            b1 = byte_data
            b2 = np.array([], dtype=np.uint8)
        else:
            b1 = byte_data[:-1]
            b2 = byte_data[1:]
        np.add.at(freq_matrix, (b1, b2), 1)

        # Zero out the '0000' bi-gram frequency
        freq_matrix[0, 0] = 0

        # Apply full-frame DCT
        dct_matrix = cv2.dct(freq_matrix.astype(np.float32))

        # Normalize DCT result to 0-255
        min_val = dct_matrix.min()
        max_val = dct_matrix.max()
        if max_val - min_val == 0:
            dct_matrix_normalized = np.zeros_like(dct_matrix, dtype=np.uint8)
        else:
            dct_matrix_normalized = ((dct_matrix - min_val) / (max_val - min_val) * 255).astype(np.uint8)

        # Save as grayscale image
        img = Image.fromarray(dct_matrix_normalized)
        img.save(output_path)
        logging.info(f"Bigram DCT saved to {output_path}")
        return True
    except Exception as e:
        logging.error(f"Error processing {input_path}: {e}")
        print(f"Error processing {input_path}: {e}")
        return False


def generate_bigram_dct(exe_path, output_path):
    try:
        return process_bigram((exe_path, output_path))
    except Exception as e:
        logging.error(f"Error generating bigram DCT: {e}")
        print(f"Error generating bigram DCT: {e}")
        return False


def extract_api_calls(exe_path):
    """
    Extract API calls from the Import Address Table (IAT) of a PE file.
    """
    try:
        pe = pefile.PE(exe_path)
        api_calls = []
        if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                for func in entry.imports:
                    if func.name is not None:
                        api_calls.append(func.name.decode('utf-8'))
        logging.info(f"Extracted {len(api_calls)} API calls from {exe_path}")
        return api_calls
    except Exception as e:
        logging.error(f"Error processing {exe_path}: {e}")
        print(f"Error processing {exe_path}: {e}")
        return []  # Return empty list instead of None


def vectorize_api_calls(api_calls, vectorizer):
    """
    Vectorize the list of API calls using the provided vectorizer.
    Returns a dense numpy array.
    """
    if not api_calls:
        return np.zeros(len(vectorizer.vocabulary_), dtype=np.float32)
    doc = " ".join(api_calls)
    vec_sparse = vectorizer.transform([doc])
    vec_dense = vec_sparse.toarray()[0]
    return vec_dense


def compute_md5(file_path):
    """
    Compute the MD5 hash of the given file.
    """
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        logging.info(f"Computed MD5 hash for {file_path}")
        return hash_md5.hexdigest()
    except Exception as e:
        logging.error(f"Error computing MD5 hash: {e}")
        print(f"Error computing MD5 hash: {e}")
        return None


def query_virustotal(md5_hash, api_key):
    """
    Query VirusTotal API for the given MD5 hash.
    Returns a summary of detections.
    """
    headers = {
        'x-apikey': api_key
    }
    url = f"https://www.virustotal.com/api/v3/files/{md5_hash}"
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            stats = data['data']['attributes']['last_analysis_stats']
            logging.info(f"VirusTotal analysis for {md5_hash}: {stats}")
            return stats  # e.g., {'malicious': 5, 'suspicious': 2, 'harmless': 100, ...}
        elif response.status_code == 404:
            logging.warning(f"File not found in VirusTotal: {md5_hash}")
            return "File not found in VirusTotal."
        else:
            logging.error(f"Error querying VirusTotal: {response.status_code} - {response.text}")
            return f"Error querying VirusTotal: {response.status_code} - {response.text}"
    except Exception as e:
        logging.error(f"Exception querying VirusTotal: {e}")
        return f"Exception querying VirusTotal: {e}"


###############################################################################
# Model Loading Functions with Lazy Loading
###############################################################################

# Global variables to hold loaded models and vectorizer
_loaded_resources = {
    'vectorizer': None,
    'byteplot_model': None,
    'bigram_model': None,
    'apicalls_model': None,
    'fusion_model': None
}


@suppress_warnings_during_loading
def load_vectorizer():
    """
    Lazy load the CountVectorizer.
    """
    if _loaded_resources['vectorizer'] is None:
        try:
            vectorizer = joblib.load(VECTORIZER_PATH)
            _loaded_resources['vectorizer'] = vectorizer
            logging.info("Vectorizer loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading vectorizer: {e}")
            print(f"Error loading vectorizer: {e}")
            sys.exit(1)
    return _loaded_resources['vectorizer']


@suppress_warnings_during_loading
def load_byteplot_model():
    """
    Lazy load ByteplotResNet model.
    """
    if _loaded_resources['byteplot_model'] is None:
        try:
            byteplot_model = ByteplotResNet()
            byteplot_model.load_state_dict(torch.load(BYTEPLOT_MODEL_PATH, map_location=torch.device('cpu')))
            byteplot_model.eval()
            _loaded_resources['byteplot_model'] = byteplot_model
            logging.info("Byteplot ResNet model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading Byteplot ResNet model: {e}")
            print(f"Error loading Byteplot ResNet model: {e}")
            sys.exit(1)
    return _loaded_resources['byteplot_model']


@suppress_warnings_during_loading
def load_bigram_model():
    """
    Lazy load BigramResNet model.
    """
    if _loaded_resources['bigram_model'] is None:
        try:
            bigram_model = BigramResNet()
            bigram_model.load_state_dict(torch.load(BIGRAM_MODEL_PATH, map_location=torch.device('cpu')))
            bigram_model.eval()
            _loaded_resources['bigram_model'] = bigram_model
            logging.info("Bigram ResNet model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading Bigram ResNet model: {e}")
            print(f"Error loading Bigram ResNet model: {e}")
            sys.exit(1)
    return _loaded_resources['bigram_model']


@suppress_warnings_during_loading
def load_apicalls_model(vectorizer):
    """
    Lazy load APICallsMLP model.
    """
    if _loaded_resources['apicalls_model'] is None:
        try:
            apicalls_model = APICallsMLP(input_dim=len(vectorizer.vocabulary_))
            apicalls_model.load_state_dict(torch.load(APICALLS_MODEL_PATH, map_location=torch.device('cpu')))
            apicalls_model.eval()
            _loaded_resources['apicalls_model'] = apicalls_model
            logging.info("API Calls MLP model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading API Calls MLP model: {e}")
            print(f"Error loading API Calls MLP model: {e}")
            sys.exit(1)
    return _loaded_resources['apicalls_model']


@suppress_warnings_during_loading
def load_fusion_model():
    """
    Lazy load FusionModel3.
    """
    if _loaded_resources['fusion_model'] is None:
        try:
            fusion_model = FusionModel3()
            fusion_model.load_state_dict(torch.load(FUSION_MODEL_PATH, map_location=torch.device('cpu')))
            fusion_model.eval()
            _loaded_resources['fusion_model'] = fusion_model
            logging.info("Fusion Model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading Fusion Model: {e}")
            print(f"Error loading Fusion Model: {e}")
            sys.exit(1)
    return _loaded_resources['fusion_model']


###############################################################################
# Classification Function with Loading Indicator
###############################################################################

def show_loading(message, stop_event):
    """
    Display a loading spinner in the CLI.
    """
    spinner = ['|', '/', '-', '\\']
    idx = 0
    while not stop_event.is_set():
        sys.stdout.write(f"\r{message} {spinner[idx % len(spinner)]}")
        sys.stdout.flush()
        time.sleep(0.1)
        idx += 1
    sys.stdout.write("\r" + " " * (len(message) + 2) + "\r")  # Clear the line


def load_resources_with_spinner():
    """
    Load vectorizer and all models while displaying a spinner.
    """
    stop_event = threading.Event()
    loading_thread = threading.Thread(target=show_loading, args=("Initializing...", stop_event))
    loading_thread.start()

    try:
        # Load vectorizer and models
        vectorizer = load_vectorizer()
        byteplot_model = load_byteplot_model()
        bigram_model = load_bigram_model()
        apicalls_model = load_apicalls_model(vectorizer)
        fusion_model = load_fusion_model()
    finally:
        # Stop the spinner
        stop_event.set()
        loading_thread.join()


def classify_file(exe_path, virus_total_api_key=None):
    """
    Process the executable and classify as malware or benign.
    Returns classification result, confidence score, MD5 hash, VirusTotal results.
    """
    # Create a temporary directory for this scan
    temp_dir = tempfile.mkdtemp(prefix='temp_analysis_')

    # Start loading spinner for classification
    stop_event = threading.Event()
    loading_thread = threading.Thread(target=show_loading, args=("Scanning...", stop_event))
    loading_thread.start()

    try:
        # Validate file
        if not os.path.isfile(exe_path):
            print("Error: File does not exist.")
            return None, None, None, None
        if not exe_path.lower().endswith('.exe'):
            print("Error: Selected file is not an .exe file.")
            return None, None, None, None

        # Generate Byteplot
        byteplot_path = os.path.join(temp_dir, 'byteplot.png')
        success = generate_byteplot(exe_path, byteplot_path)
        if not success:
            print("Failed to generate byteplot.")
            return None, None, None, None

        # Generate Bigram DCT
        bigram_path = os.path.join(temp_dir, 'bigram_dct.png')
        success = generate_bigram_dct(exe_path, bigram_path)
        if not success:
            print("Failed to generate bigram DCT.")
            return None, None, None, None

        # Extract API Calls
        api_calls = extract_api_calls(exe_path)
        if not api_calls:
            api_calls = []

        # Vectorize API Calls
        vectorizer = _loaded_resources['vectorizer']
        if vectorizer is None:
            print("Vectorizer not loaded. Cannot vectorize API calls.")
            logging.error("Vectorizer not loaded. Cannot vectorize API calls.")
            return None, None, None, None
        api_vector = vectorize_api_calls(api_calls, vectorizer)
        api_vector_tensor = torch.tensor(api_vector, dtype=torch.float32)

        # Load and process Byteplot
        try:
            byteplot_image = Image.open(byteplot_path).convert("RGB")
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])
            byteplot_tensor = transform(byteplot_image).unsqueeze(0)
        except Exception as e:
            print(f"Error processing byteplot image: {e}")
            logging.error(f"Error processing byteplot image: {e}")
            return None, None, None, None

        # Load and process Bigram DCT
        try:
            bigram_image = Image.open(bigram_path).convert("RGB")
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])
            bigram_tensor = transform(bigram_image).unsqueeze(0)
        except Exception as e:
            print(f"Error processing bigram image: {e}")
            logging.error(f"Error processing bigram image: {e}")
            return None, None, None, None

        # Obtain logits from sub-models
        with torch.no_grad():
            byte_logit = _loaded_resources['byteplot_model'](byteplot_tensor).item()
            bigram_logit = _loaded_resources['bigram_model'](bigram_tensor).item()
            apicalls_logit = _loaded_resources['apicalls_model'](api_vector_tensor).item()

        # Prepare fusion input
        fusion_input = torch.tensor([byte_logit, apicalls_logit, bigram_logit], dtype=torch.float32).unsqueeze(0)

        # Obtain fusion logit
        with torch.no_grad():
            fusion_logit = _loaded_resources['fusion_model'](fusion_input).item()
            probability = torch.sigmoid(torch.tensor(fusion_logit)).item()

        classification = "Malware" if probability >= THRESHOLD else "Benign"
        confidence = probability if probability >= THRESHOLD else 1 - probability

        # Compute MD5 Hash
        md5_hash = compute_md5(exe_path)

        # Query VirusTotal
        if virus_total_api_key and md5_hash:
            vt_result = query_virustotal(md5_hash, virus_total_api_key)
        else:
            vt_result = "VirusTotal API key not provided."

        return classification, confidence, md5_hash, vt_result

    finally:
        # Stop loading spinner
        stop_event.set()
        loading_thread.join()

        # Cleanup temporary files
        try:
            shutil.rmtree(temp_dir)
            logging.info(f"Deleted temporary directory: {temp_dir}")
        except FileNotFoundError:
            logging.warning(f"Temporary directory already deleted: {temp_dir}")
        except Exception as e:
            logging.error(f"Error deleting temporary files: {e}")



###############################################################################
# CLI Interface
###############################################################################

def get_virustotal_api_key():
    """
    Prompt the user to enter their VirusTotal API key or skip.
    Returns the API key if provided, else None.
    """
    while True:
        choice = input("Do you want to provide a VirusTotal API key? (y/n): ").strip().lower()
        if choice == 'y':
            api_key = input("Enter your VirusTotal API key: ").strip()
            if api_key:
                return api_key
            else:
                print("No API key entered. Skipping VirusTotal integration.")
                return None
        elif choice == 'n':
            return None
        else:
            print("Invalid input. Please enter 'y' or 'n'.")


def display_menu():
    """
    Display the main menu.
    """
    menu = """
=== TeaScan v1.0 ===

1. Scan 
2. About
3. Exit

"""
    print(menu)


def about_tool():
    """
    Display information about the tool.
    """
    about_text = """
TeaScan v1.0
Developed by: Samuel Moloney


This tool classifies executables as Malware or Benign using machine learning models.
It provides additional information such as MD5 hashes and VirusTotal analysis results.~
Malware Executables are converted to Byteplots, and Bi-grams, and the API calls are extracted.
These are then assessed using the trained model to provide an outlook on whether the selected 
executable is likely Malware or not.
The model trained achieved a 96% accuracy in detecting benign files and malware, and a 96% recall
of detecting all malware.
This tool should NOT be used as final evidence to decide if a file is malware, only to provide
information for static analysis of executables.
This App is not liable for any damages caused from decisions made purely based on the outcomes
of this tool.
For any further questions, please contact samuelmoloney70@outlook.com, or contact on 
Git-Hub: https://github.com/ChamomileTea1/ChamomileTea1/tree/main

"""
    print(about_text)


def main_cli():
    """
    Main CLI loop.
    """
    while True:
        display_menu()
        choice = input("Enter your choice (1-3): ").strip()

        if choice == '1':
            while True:
                exe_path = input("Enter the full path of the executable to scan: ").strip('"')
                if not os.path.isfile(exe_path):
                    print("Error: File does not exist.")
                    continue
                if not exe_path.lower().endswith('.exe'):
                    print("Error: Selected file is not an .exe file.")
                    continue

                print("\nScanning the file...")

                # Perform classification
                classification, confidence, md5_hash, vt_result = classify_file(
                    exe_path,
                    virus_total_api_key
                )

                if classification is None:
                    print("Scanning failed due to errors.")
                    break

                # Display results
                print(f"\nClassification: {classification}")
                print(f"Confidence Score: {confidence:.4f}")
                print(f"MD5 Hash: {md5_hash if md5_hash else 'Unavailable'}")
                print(
                    f"VirusTotal Results: {json.dumps(vt_result, indent=4) if isinstance(vt_result, dict) else vt_result}"
                )

                # Option to scan another file
                another = input("\nDo you want to scan another file? (y/n): ").strip().lower()
                if another == 'y':
                    continue  # Prompt to scan another file
                elif another == 'n':
                    print("Returning to main menu.")
                    break  # Return to main menu
                else:
                    print("Invalid input. Returning to main menu.")
                    break

        elif choice == '2':
            about_tool()

        elif choice == '3':
            print("Exiting the tool.")
            sys.exit(0)

        else:
            print("Invalid choice. Please select an option from 1 to 3.")


###############################################################################
# Main Entry Point
###############################################################################

def main():
    install_torch()

    try:
        ByteplotResNet, BigramResNet, APICallsMLP, FusionModel3 = define_models()
    except ImportError:
        print(
            "Failed to import torch or torchvision after installation.\n"
            "Please ensure they are installed correctly by following the instructions in the README.md."
        )
        sys.exit(1)
   
# Display an "Initializing..." spinner right after the script starts
    stop_event = threading.Event()
    loading_thread = threading.Thread(target=show_loading, args=("Initializing...", stop_event))
    loading_thread.start()




    try:
        # Simulate initialization (e.g., load resources)
        load_resources_with_spinner()  # This already contains a spinner for resource loading
    finally:
        # Stop the spinner
        stop_event.set()
        loading_thread.join()


    # Display CLI art/title
    cli_art = r"""
████████╗███████╗ █████╗ ███████╗ ██████╗ █████╗ ███╗   ██╗
╚══██╔══╝██╔════╝██╔══██╗██╔════╝██╔════╝██╔══██╗████╗  ██║
   ██║   █████╗  ███████║███████╗██║     ███████║██╔██╗ ██║
   ██║   ██╔══╝  ██╔══██║╚════██║██║     ██╔══██║██║╚██╗██║
   ██║   ███████╗██║  ██║███████║╚██████╗██║  ██║██║ ╚████║
   ╚═╝   ╚══════╝╚═╝  ╚═╝╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═══╝
"""
    print(cli_art)

    # Load resources with initializing spinner


    # Retrieve VirusTotal API key
    global virus_total_api_key
    virus_total_api_key = get_virustotal_api_key()

    # Start main CLI loop
    main_cli()


if __name__ == "__main__":

    main()
