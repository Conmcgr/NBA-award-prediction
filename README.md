# NBA Award Prediction

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Setup](#setup)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Running the Application Locally](#running-the-application-locally)
  - [Deploying to Render](#deploying-to-render)
- [Model Details](#model-details)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Contact](#contact)

## Introduction

Welcome to the NBA Award Prediction project! This web application leverages machine learning models to predict the most likely candidates for the NBA MVP (Most Valuable Player) and ROY (Rookie of the Year) awards based on player statistics. By analyzing historical NBA data, we aim to provide accurate predictions and insights into player performances.

## Features

- Scrape NBA player statistics from Basketball Reference.
- Clean and preprocess the data.
- Train machine learning models to predict MVP and ROY awards.
- Web interface for uploading player statistics and viewing predictions.

## Setup

### Prerequisites

- Python 3.10 or higher
- pip (Python package installer)
- Git
- An account on [Render](https://render.com/) (or another hosting provider if preferred)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Conmcgr/NBA-award-prediction.git
   cd NBA-award-prediction
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:
   - On windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```bash
     source venv/bin/activate
     ```
4. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Application Locally

1. Ensure that your virtual environment is activated.
2. Run the Flask application:
   ```bash
   python app.py
   ```
3. Open your web browser and navigate to http://localhost:5000.

### Deploying to Render

1. Create a new web service on Render and link it to your GitHub repository.
2. Set up the environment to include the necessary dependencies by ensuring your requirements.txt file is accurate.
3. Add a render.yaml file in the root of your project to configure the deployment (if required).
4. Follow Render’s documentation to complete the deployment process.

## Model Details

### Data Collection and Cleaning

The data is collected from Basketball Reference, including player statistics and award voting results from 1977 to 2024. The data is cleaned and preprocessed to ensure consistency and accuracy.

### Feature Engineering

Features are created by normalizing player statistics per 48 minutes of play. This includes calculating field goals, assists, steals, and other performance metrics per 48 minutes to standardize comparisons between players.

### Model Training

Two RandomForestRegressor models are trained:

- MVP Model: Predicts the MVP vote share based on player statistics.
- ROY Model: Predicts the ROY vote share based on player statistics.

The models are evaluated using Mean Squared Error (MSE) and calibrated using Isotonic Regression to improve prediction accuracy. The models are saved using joblib for deployment.

## Project Structure

nba-award-prediction/
│
├── data/ # Directory to store scraped data
├── models/ # Directory to store trained models
├── templates/ # HTML templates for the web application
│ ├── layout.html
│ ├── upload.html
│
├── jupyter_notebook (data and model exploration)/
│ ├── data_collection_cleaning.ipynb
│ ├── model_creation.ipynb
│
├── app.py # Main Flask application file
├── data_collection_cleaning.py # Script for data collection and cleaning
├── model_training.py # Script for training models
├── requirements.txt # Python package dependencies
├── README.md # Project README file
└── render.yaml # Render deployment configuration (optional)

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## Contact

If you have any questions or suggestions, feel free to reach out to me at connor.q.mcgraw@gmail.com.
