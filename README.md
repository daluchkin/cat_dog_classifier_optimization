[Portfolio](https://github.com/daluchkin/data-analyst-portfolio) |  [Projects](https://github.com/daluchkin/data-analyst-portfolio/blob/main/projects.md) | [Certificates](https://github.com/daluchkin/data-analyst-portfolio/blob/main/certificates.md) | [Contacts](https://github.com/daluchkin/data-analyst-portfolio#my_contacts)


# Cat vs. Dog Classifier Optimization

This project is a personal experiment aimed at improving a convolutional neural network (CNN) for classifying cats and dogs. The main goal was to explore how **Bayesian optimization** (BO) can be used to fine-tune both the architecture and hyperparameters of a CNN, leading to better performance.

## Key Aspects of the Project

- Implemented a CNN-based image classifier for distinguishing between cats and dogs.
- Applied **Bayesian optimization** to search for the optimal combination of hyperparameters (learning rate, batch size, dropout rate, etc.) and architecture (number of layers, filter sizes, etc.).

## Optimization Strategies

  - **Grid Search**: Exhaustively evaluates all possible hyperparameter combinations within a predefined range, which is computationally expensive and inefficient for high-dimensional spaces.
  - **Random Search**: Selects random hyperparameter combinations, which is more efficient than grid search but still does not leverage prior knowledge from past evaluations.
  - **Bayesian Optimization**: Uses a probabilistic model (e.g., Gaussian Process) to predict promising hyperparameter values based on previous results, allowing for a more efficient and informed search.

## Advantages of Bayesian Optimization
- **More sample-efficient:** Finds better hyperparameters with fewer evaluations compared to grid and random search.
- **Adaptability:** Adjusts the search strategy dynamically based on past results.
- **Better exploration-exploitation trade-off:** Balances between trying new hyperparameter values and refining promising ones.

This project was conducted as an **experimental study** to better understand how Bayesian optimization influences CNN training and performance.

## Experiment Results

| Experiment Type                          | Accuracy  |
|------------------------------------------|-----------|
| Manual CNN Creation                      | **0.8255** |
| Hyperparameters Optimization             | **0.8381** |
| Layers & Hyperparameters Optimization    | **0.8614** |


## Notebook/Code

- [`cat-dog-classifier-optimization.ipynb`](./cat-dog-classifier-optimization.ipynb)

## Runtime

Runtime of the model training per iteration:

+ **Kaggle GPU P100:** ~9 hours.
+ **MacBook Pro 2018 Intel CPU:** $\infty$ hours 🤷‍♂️

## Technologies Used

- Python (Jupyter Notebook)
- TensorFlow & Keras (Deep Learning Model Training)
- Matplotlib & Seaborn (Data Visualization)
- Scikit-Learn (Evaluation Metrics)

## Results & Findings

The project evaluates multiple deep learning architectures to identify the best-performing model using Bayesian Optimization. The final trained CNN achieves high accuracy in emotion classification, demonstrating its effectiveness in real-world scenarios.

## How to Run the Project

To easily run this project with **GPU acceleration**, follow these steps using **Kaggle Notebooks**:

### 1. Open the Dataset and Create a Notebook
- Go to the dataset profile: [Cats & Dogs Dataset](https://www.kaggle.com/datasets/tongpython/cat-and-dog).
- Click **"New Notebook"** to create a new notebook.
- Upload the `cat_dog_classifier_optimization.ipynb` notebook to Kaggle.
- In the right panel, enable **GPU acceleration** under the **"Accelerator"** section.

### 2. Run All Cells
- Open the uploaded notebook.
- Click **"Run All"** to execute all cells.

### 3. Monitor the Training Process
- The model will automatically train using GPU acceleration.
- Track the loss and accuracy metrics during training.

Once training is complete, you can save and download the trained model for further use.


This setup ensures **fast training** using Kaggle's free GPU resources without requiring any local setup.


Feel free to explore, contribute, or provide feedback!


[Portfolio](https://github.com/daluchkin/data-analyst-portfolio) |  [Projects](https://github.com/daluchkin/data-analyst-portfolio/blob/main/projects.md) | [Certificates](https://github.com/daluchkin/data-analyst-portfolio/blob/main/certificates.md) | [Contacts](https://github.com/daluchkin/data-analyst-portfolio#my_contacts)
