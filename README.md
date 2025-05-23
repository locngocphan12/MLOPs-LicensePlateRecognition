﻿# License Plate Detection Pipeline

This project is an end-to-end **MLOps pipeline** for **License Plate Detection** using the **YOLOv8** object detection model. The pipeline is orchestrated with **Apache Airflow** and integrates **MLflow** for comprehensive experiment tracking, model versioning, and performance logging. This project also includes a real-time License Plate Detection API built using FastAPI and YOLOv8. The API allows users to upload images and receive detection results with bounding boxes of license plates.

This project is a part of MLOps cource (CS317.P22), with members:
- Huynh La viet Toan: 22521486
- Nguyen Truong Minh Khoa: 22520680
- Nguyen Thanh Luan: 22520826
- Luong Truong Thinh: 22521412
- Phan Phuoc Loc Ngoc: 22520960


## Key Features

### Automated MLOps Workflow:
- Built using **Apache Airflow** with `@task`-decorated Python functions.

### Experiment Tracking with MLflow
- Tracks **hyperparameters**, **model metrics**, and **checkpoints**.
- Supports nested runs for hyperparameter tuning and evaluation.
- Automatically logs dataset metadata: **source**, **url**, **version**.

### Hyperparameter Tuning & Model Evaluation
- Performs grid search over key training parameters (`freeze_layers`, `epochs`, `lr0`).
- Evaluates multiple model variants and selects the best based on evaluation metrics.
- Stores and logs the **best model** based on customizable metrics.

### Task Orchestration
- Scheduled to run every 30 minutes.
- Task-level retries and modular flow for robust execution.

### API Inference Service with FastAPI
- Upload an image via web interface or POST request
- Automatically detects license plates using YOLOv8 model
- Returns annotated image and detection details
- Lightweight UI for image testing
- Ready to deploy via Docker and Docker Compose


## How It Works

1. **Initialize Hyperparameters**  
   Generates a grid of parameters to try during training.

2. **Prepare Pretrained Model**  
   Uses an existing YOLOv8 model or trains a base model from scratch and logs it to MLflow.

3. **Train and Validate Models**  
   Trains models using different hyperparameter combinations, logs metrics and weights.

4. **Evaluate Models**  
   Evaluates trained models and selects the best based on a specific metric.

5. **Save Best Model**  
   Logs the final best-performing model and its metadata to MLflow.

6. **Deploy Inference API**

    Deploys the trained best model as a FastAPI application.


## Requirements

- Python 3.10+
- Apache Airflow
- MLflow
- Dependencies: ultralytics (for YOLOv8)
- Docker
- FastAPI


## Setup & Usage

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/luanntd/License-Plate-Detection-Pipeline-with-Experiment-Tracking.git
    cd License-Plate-Detection-Pipeline-with-Experiment-Tracking
    ```
2.  **Prepare Dataset:**
    - You can download dataset used for this project from [this url](https://www.kaggle.com/datasets/bomaich/vnlicenseplate) and paste the **train**, **valid**, **test** folders in `src/dataset/`.
    - Or you can choose a different dataset and follow the `README.md` in `src/dataset/`.

<!-- 2.  **Set Up Environment Variables:**
    * Create a file named `.env` in the project's root directory.
    * Run the following command to create airflow user:

        ```bash
        echo -e "AIRFLOW_UID=$(id -u)" > .env
        ``` -->

3.  **Running Docker containers:**
    ```bash
    docker-compose up -d
    ```
  
4.  **Access the UI:**
    - Open your web browser and navigate to http://localhost:8080 for Airflow webserver.
    - Open your web browser and navigate to http://localhost:5000 for MLflow UI.
    - Open your web browser and navigate to http://localhost:8070, upload images and receive detection results with bounding boxes of license plates.

## Demo

### Airflow Pipeline

![airflow_pipeline](assets/pipeline.png)

### MLflow Experiment Tracking

![mlflow_experiment_tracking](assets/experiment_tracking.gif)

### API Inference Service with FastAPI

![predict_using_fastapi](assets/yolo_detector.gif)

## Collaborators
<a href="https://github.com/luanntd">
  <img src="https://github.com/luanntd.png?size=50" width="50" style="border-radius: 50%;" />
</a>
<a href="https://github.com/Khoa-Nguyen-Truong">
  <img src="https://github.com/Khoa-Nguyen-Truong.png?size=50" width="50" style="border-radius: 50%;" />
</a>
<a href="https://github.com/HuynhToan2004">
  <img src="https://github.com/HuynhToan2004.png?size=50" width="50" style="border-radius: 50%;" />
</a>
<a href="https://github.com/locngocphan12">
  <img src="https://github.com/locngocphan12.png?size=50" width="50" style="border-radius: 50%;" />
</a>
<a href="https://github.com/thinhlt04">
  <img src="https://github.com/thinhlt04.png?size=50" width="50" style="border-radius: 50%;" />
</a>

<p align="right">(<a href="#readme-top">back to top</a>)</p>
