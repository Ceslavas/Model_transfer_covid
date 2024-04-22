
# Medical Imaging Analysis Tools

## Overview
This program is designed for the classification of medical images using deep learning techniques to automatically detect diseases such as COVID-19. Utilizing neural network architectures like MobileNetV2, EfficientNetB7, and ResNet50, the program performs the following tasks:

1. **Data Preparation:** Imports and preprocesses images from specified directories.
2. **Model Optimization and Selection:** Automatically selects the best model through cross-validation and determines optimal hyperparameters.
3. **Model Training:** Trains the selected model on training data using techniques to prevent overfitting.
4. **Results Analysis:** Validates and tests the model on new data to assess its effectiveness.

## Requirements
- Python 3.x (compatibility with other versions of Python is not guaranteed).
- Libraries:
  ```
  numpy
  tensorflow
  matplotlib
  yaml
  ```
  To install all required libraries at once, run: `pip install -r requirements.txt`.

## Installation
1. Ensure Python 3.x is installed on your computer.
2. Clone this repository or download the project files to your computer:
   ```
   git clone https://github.com/Ceslavas/Model_transfer_covid.git "D:\your_folder"
   ```

## Configuration
Before using the program, configure the `config.yaml` file:
```yaml
TRAIN_DIR: './data/train'
TEST_DIR: './data/test'
BATCH_SIZE: 8
IMAGE_WIDTH: 224
IMAGE_HEIGHT: 224
VALIDATION_SPLIT: 0.2
RANDOM_SEED: 123
EPOCHS: 10
```

## Running the Project
To use the program, follow these steps:
1. Open a command line or terminal.
2. Navigate to the directory where the `src/Model_transfer_covid.py` script is located.
3. Enter the command `python Model_transfer_covid.py`.

## Results
The program automatically processes input data, trains the model, and conducts its validation and testing, providing a detailed analysis of the results.

## FAQ
**Q:** Can the program be used to process multiple types of medical images?
**A:** No, the program is currently designed to work with specific types of medical images as determined by its configuration parameters. It is not set up for processing multiple different types of medical images.

## Contributions
Contributions are welcome! If you have ideas for improvements or new features, please submit a pull request or create an issue.

## License
This project is distributed under the MIT License. See the LICENSE.txt file for details.
