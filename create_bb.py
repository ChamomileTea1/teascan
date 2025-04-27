import os
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

# Directories
input_dirs = {
    "train": r"define here",
    "val": r"define here",
    "test": r"define here"
}

output_dirs = {
    "train": r"define here",
    "val": r"define here",
    "test": r"define here"
}

# Ensure the output directories exist
for key, path in output_dirs.items():
    os.makedirs(os.path.join(path, "byteplots"), exist_ok=True)


def create_byteplot(file_path, output_path):
    try:
        with open(file_path, 'rb') as f:
            data = f.read()

        array = np.frombuffer(data, dtype=np.uint8)
        image_size = 256
        array = array[:image_size * image_size]
        padded_array = np.zeros(image_size * image_size, dtype=np.uint8)
        padded_array[:len(array)] = array
        image = padded_array.reshape((image_size, image_size))

        plt.imsave(output_path, image, cmap='gray')
    except Exception as e:
        print(f"Error creating byteplot for {file_path}: {e}")


def process_file(file_path, output_byteplot):
    create_byteplot(file_path, output_byteplot)


def process_file_wrapper(args):
    return process_file(*args)


def process_directory(input_dir, output_dir):
    tasks = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            file_path = os.path.join(root, file)
            base_name = os.path.splitext(file)[0]

            output_byteplot = os.path.join(output_dir, "byteplots", f"{base_name}_byteplot.png")

            tasks.append((file_path, output_byteplot))

    with ProcessPoolExecutor(max_workers=4) as executor:  # Limit to 4 workers to reduce system load
        for i, _ in enumerate(executor.map(process_file_wrapper, tasks), 1):
            print(f"Processed {i}/{len(tasks)} files", end="\r")


if __name__ == "__main__":
    # Main processing
    for key in input_dirs:
        print(f"Processing {key} dataset...")
        process_directory(input_dirs[key], output_dirs[key])
    print("Processing complete.")
