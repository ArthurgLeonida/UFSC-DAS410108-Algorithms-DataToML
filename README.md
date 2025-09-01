# Titanic Survival Prediction - A Feature Engineering Project

This project explores various **feature engineering** techniques to improve the prediction of passenger survival on the RMS Titanic. The primary goal is to manipulate data from the original dataset and then use a machine learning model to demonstrate their impact on predictive accuracy.

The entire development environment is containerized using **Docker** to ensure reproducibility.

-----

## \#\# How to Run 🚀

To get this project running, you'll need to have Docker and Docker Compose installed.

1.  **Start the Docker Container**:
    From the project's root directory, run the following command to build the image and start the Jupyter Notebook server in the background.

    ```bash
    docker-compose up -d
    ```

2.  **Access Jupyter Notebook**:
    To get the access link with the required token, check the container's logs:

    ```bash
    docker-compose logs
    ```

    Copy the URL (e.g., `http://127.0.0.1:8888/tree?token=...`) and paste it into your web browser.

3.  **Stop the Container**:
    When you are finished, stop the container with:

    ```bash
    docker-compose down
    ```