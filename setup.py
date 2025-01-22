from setuptools import setup, find_packages

setup(
    name="teascan",
    version="1.0.0",
    description="TeaScan Machine learning Malware identifier",
    author="Samuel Moloney",
    author_email="samuelmoloney70@outlook.com",
    url="https://github.com/ChamomileTea1",
    packages=find_packages(),  # Automatically find all packages in the project
    include_package_data=True,  # Include package data specified in MANIFEST.in
    package_data={
        "teascan": ["models/*", "teascan_icon.ico"],  # Explicitly include 'models' and the icon
    },
    install_requires=[
        
        "torch==2.5.1+cpu @ https://download.pytorch.org/whl/cpu/torch-2.5.1+cpu-cp39-cp39-win_amd64.whl",
        "torchvision==0.16.1+cpu @ https://download.pytorch.org/whl/cpu/torchvision-0.16.1+cpu-cp39-cp39-win_amd64.whl",
        "scikit-learn==1.6.0",
        "joblib==1.4.2",
        "opencv-python==4.10.0.84",
        "Pillow==11.0.0",
        "pefile==2023.2.7",
        "numpy==1.26.4",
        "requests==2.32.3",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL-3.0 license",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "teascan=teascan.main:main",  # Maps the `teascan` command to your `main.py`
        ]
    },
)
