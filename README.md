TeaScan v1.0
Developed by: Samuel Moloney


This tool classifies executables as Malware or Benign using machine learning models.
It provides additional information such as MD5 hashes and VirusTotal analysis results, if you provide your
own API key, to compare against. An APIkey is esy to obtain from the virus total website

Malware Executables are converted to Byteplots, and Bi-grams, and the API calls are extracted.
These are then assessed using the trained model to provide an outlook on whether the selected 
executable is likely Malware or not.
The model trained achieved a 96% accuracy in detecting benign files and malware, and a 96% recall
of detecting all malware.
This tool should NOT be used as final evidence to decide if a file is malware, only to provide
information for static analysis of executables.
This App is not liable for any damages caused from decisions made purely based on the outcomes
of this tool.

As this model utilizes AI, more specifically CNN's, modules such as torch-vision, torch and scickit-learn
are required. Please see the requirements.txt.

Ensure you have Python 3.6 or higher installed on your system.
To install:
pip install teascan

As of, windows.exe files are only supported, in the future, this may expand to .elf and other varieties of malware.

For any further questions, please contact samuelmoloney70@outlook.com, or contact on 
Git-Hub: https://github.com/ChamomileTea1/ChamomileTea1/tree/main
