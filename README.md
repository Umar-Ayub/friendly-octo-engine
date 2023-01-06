### MLE Interview

To run inference locally
1. Clone repo 
2. Inside root directory, build Docker image and pass model location as a variable. Example model is included in repo. 
   ```
   docker build -t mle --build-arg LOCATION=models/model.pkl .
   ```
3. Run container in detached mode
    ```
    docker run -p 8887:8887 -d mle
    ```
4. Open a new terminal, navigate to root directory and run request.py as an example of sending a POST request to the localhost server.
    ```python request.py
    ```