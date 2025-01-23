**TeaScan v1.0**
**Developed by: Samuel Moloney**


Due to the difficulties of ensuring compatible versions of torch and torch vision,
which are required for the machine learning model, these should be installed manually, 
although if they are no present the program will attempt to install them itself.

PyTorch can be installed by visiting [PyTorch.org](https://pytorch.org/get-started/locally/) and selecting
- Your OS (Windows, Linux, or macOS)
- Your Package: `pip`
- Language: `Python`
- Platform: `CPU` (or a specific CUDA version like `cu118` if you wish to use your GPU)

Then run the suggested installation command, for example:

    pip install torch torchvision

**Please ensure your version of Python is compatible with PyTorch.**
====================================

This tool classifies executable programs as Malware or Benign using machine learning models.
It provides additional information such as MD5 hashes and VirusTotal analysis results, if you provide your
own API key, to compare against. An API key is easy to obtain from the virus total website

Malware executable's are converted to Byteplots, and Bi-grams, and the API calls are extracted.
These are then assessed using the trained model to provide an outlook on whether the selected 
executable is likely malware or not.
The model trained achieved a 96% accuracy in detecting benign files and malware, and a 96% recall of detecting all malware against a modern dataset.

This tool should NOT be used as final evidence to decide if a file is malware, only to provide
information for static analysis of executables.
This App is not liable for any damages caused from decisions made purely based on its outcomes.

As this model utilizes AI and CNN's, modules such as torch-vision, torch and scickit-learn
are required. Please see the requirements.

Ensure you have Python 3.6 or higher installed on your system.

To install:

    git clone https://github.com/ChamomileTea1/teascan.git
    cd teascan
    pip install .

As of 2025, windows exectuables are only supported, in the future, this may expand to .elf and other varieties of malware.

> For any further questions, please contact samuelmoloney70@outlook.com,
> or contact on  Git-Hub:
> https://github.com/ChamomileTea1/ChamomileTea1/tree/main


