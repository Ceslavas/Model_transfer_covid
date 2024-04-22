import numpy as np
from typing import Tuple, Dict, List

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import yaml


# Setting constants (default values)
TRAIN_DIR: str = './data/train'
TEST_DIR: str = './data/test'
BATCH_SIZE: int = 8
IMAGE_SIZE: Tuple[int, int] = (224, 224)
VALIDATION_SPLIT: float = 0.2
RANDOM_SEED: int = 123
EPOCHS: int = 10



def get_labels_only(dataset: tf.data.Dataset) -> np.ndarray:
    """
    Extracts only the labels from the dataset.

    Args:
        dataset (tf.data.Dataset): TensorFlow Dataset.

    Returns:
        np.ndarray: Array of labels.
    """
    labels_list: List[np.ndarray] = [labels.numpy() for _, labels in dataset]
    return np.concatenate(labels_list, axis=0)


def prepare_dataset(train_dir: str, test_dir: str, batch_size: int, image_size: Tuple[int, int]) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Prepares and optimizes datasets for training, validation, and testing.

    Args:
        train_dir (str): Path to training data.
        test_dir (str): Path to test data.
        batch_size (int): Batch size.
        image_size (Tuple[int, int]): Image dimensions.

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: Training, validation, and test datasets.
    """

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=RANDOM_SEED,
        shuffle=True,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='binary'
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=RANDOM_SEED,
        shuffle=True,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='binary'
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        seed=RANDOM_SEED,
        shuffle=False,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='binary'
    )

    # Data caching to speed up training
    train_ds = train_ds.cache()
    val_ds = val_ds.cache()
    test_ds = test_ds.cache()
    
    # Automatic tuning of data loading settings
    AUTOTUNE = tf.data.AUTOTUNE  
    
    # Preparing the next batch of data during training
    train_ds = train_ds.prefetch(AUTOTUNE)  
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)
    
    return (train_ds, val_ds, test_ds)


def plot_training_history(history: tf.keras.callbacks.History) -> None:
    """
    Visualizes the training history of the model.
    """
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def plot_class_distribution(datasets: List[tf.data.Dataset], labels: List[str]) -> None:
    """
    Shows the class distribution across various datasets.
    """
    
    def custom_autopct(pct: float) -> str:
        total = sum(counts)
        val = int(round(pct * total / 100.0))
        return f"{val} ({pct:.2f}%)"

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle('Class Distribution across Datasets', fontsize=16)
    colors = ['#ff9999', '#66b3ff']

    for i, (dataset, label) in enumerate(zip(datasets, labels)):
        labels = get_labels_only(dataset)
        _, counts = np.unique(labels, return_counts=True)
        axs[i].pie(counts, labels=['Covid', 'Non'], autopct=custom_autopct, startangle=90, colors=colors)
        axs[i].set_title(label)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def create_and_compile_model(input_shape: Tuple[int, int, int]) -> Sequential:
    """
    Creates and compiles a Keras model.

    Args:
        input_shape (Tuple[int, int, int]): Shape of the input data.

    Returns:
        Sequential: The compiled model.
    """
    # Load model without the top layers
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def evaluate_on_balanced_subset(model: tf.keras.Model, dataset: tf.data.Dataset, batch_size: int, iterations: int = 5) -> Tuple[float, float]:
    """
    Evaluates the model on a balanced subset of the dataset multiple times and returns the average accuracy and loss.

    Args:
        model (tf.keras.Model): The model to be evaluated.
        dataset (tf.data.Dataset): The dataset for evaluation.
        batch_size (int): The batch size for evaluation.
        iterations (int): Number of evaluation iterations.

    Returns:
        Tuple[float, float]: Average accuracy and average loss of the model.
    """
    accuracies = []
    losses = []
    
    for i in range(iterations):
        # Retrieve all images and labels from the dataset
        all_images, all_labels = zip(*dataset.unbatch().as_numpy_iterator())
        all_images = np.array(all_images)
        all_labels = np.array(all_labels)

        # Indices for each class
        covid_indices = np.where(all_labels == 0)[0]
        non_covid_indices = np.where(all_labels == 1)[0]

        # Determine the number of elements for a balanced sample
        min_samples = min(len(covid_indices), len(non_covid_indices))

        # Create a balanced dataset
        np.random.shuffle(covid_indices)
        np.random.shuffle(non_covid_indices)
        balanced_indices = np.hstack((covid_indices[:min_samples], non_covid_indices[:min_samples]))
        np.random.shuffle(balanced_indices)

        balanced_images = all_images[balanced_indices]
        balanced_labels = all_labels[balanced_indices]

        # Evaluate the model
        scores = model.evaluate(balanced_images, balanced_labels, batch_size=batch_size, verbose=0)
        print(f"Test round {i+1}:\tAccuracy: {scores[1]}, Loss: {scores[0]}")
        losses.append(scores[0])
        accuracies.append(scores[1])
    
    # Calculate the averages
    mean_accuracy = np.mean(accuracies)
    mean_loss = np.mean(losses)
    
    return (mean_accuracy, mean_loss)



if __name__ == "__main__":
    # Load the configuration file
    #with open('config.yaml') as f:
    with open('../config.yaml') as f: # imant iš GitHub`o
        config = yaml.safe_load(f)
        
    # Extract the configuration parameters   
    TRAIN_DIR           = config['TRAIN_DIR']
    TEST_DIR            = config['TEST_DIR']
    BATCH_SIZE          = config['BATCH_SIZE']
    IMAGE_SIZE          = (config['IMAGE_WIDTH'], config['IMAGE_HEIGHT'])
    VALIDATION_SPLIT    = config['VALIDATION_SPLIT']
    RANDOM_SEED         = config['RANDOM_SEED']
    EPOCHS              = config['EPOCHS']

    train_dataset, val_dataset, test_dataset = prepare_dataset(TRAIN_DIR, TEST_DIR, BATCH_SIZE, IMAGE_SIZE)
    plot_class_distribution([train_dataset, val_dataset, test_dataset], ['Train', 'Validation', 'Test'])
    
    model = create_and_compile_model((IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

    # Calculating class weights
    labels = get_labels_only(train_dataset)
    covid_weight = (1 / np.sum(labels == 0)) * (len(labels) / 2.0)
    non_weight = (1 / np.sum(labels == 1)) * (len(labels) / 2.0)
    class_weights: Dict[int, float] = {0: covid_weight, 1: non_weight}

    history = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS, class_weight=class_weights)
    plot_training_history(history)
    
    # Model testing on a balanced subset
    print('\nTesting the model:')
    mean_accuracy, mean_loss = evaluate_on_balanced_subset(model, test_dataset, BATCH_SIZE, 15)
    print(f"\nAverage Accuracy: {mean_accuracy}, Average Loss: {mean_loss}, Number of Testing Iterations: 15")
