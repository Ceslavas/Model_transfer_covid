
# Medical Imaging Analysis Tools

## Overview
This script is a comprehensive solution for the classification of medical images (350 MB), using deep learning to detect diseases such as COVID-19. The program integrates various technologies and methods for data processing and machine learning (Machine Learning, Машинное обучение), including:

### Key Components:
1. **Data Preparation**: Automated import and processing of images to create optimized datasets for training, validation, and testing.
2. **Modeling**: Use of the MobileNetV2 architecture to build a neural network, including tuning and compiling the model with additional layers for classification.
3. **Results Analysis**: Testing and validation of the model on new data to assess its accuracy and reliability.

### Goals and Functions:
- **Efficient Data Use**: Caching and pre-loading of data to speed up training.
- **Automation and Configuration**: External configuration files for managing startup parameters, ensuring flexibility and repeatability of experiments.
- **Visualization of Process and Outcomes**: Displaying accuracy and losses during training, as well as the distribution of classes in the datasets.

## Requirements
- Python 3.9.13 (https://www.python.org/ftp/python/3.9.13/python-3.9.13-amd64.exe)  - compatibility with other versions of Python is not guaranteed.
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
TRAIN_DIR: '../data/train'
TEST_DIR: '../data/test'
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
