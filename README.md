# Face-mask-detection-app-Transfer-learning-MobileNet_v2-Streamlit-Docker-
This project is a deep learning application that detects whether a person is wearing a mask or not using **Transfer Learning** with TensorFlow/Keras.  


It includes:  
- A trained AI model ready for inference  
- An app interface to run predictions  
- A training notebook (available locally and on my Kaggle profile)  

# HOW TO USE :
1. Clone the repository to your local machine.
2. Create a virtual environment using the command **python -m venv venv** and then activate it using ***venv\Scripts\activate**
3. Install dependencies using **pip install -r requirements.txt**
4. Once everything is installed, you can run the app with **streamlit run app/main.py**

# Second Option : DOCKER
You can also run the app inside Docker:
1. Open your Docker Desktop App and ensure that it is activated
2. Build the Docker image using **docker build -t face-mask-detection .**
3. Run a container from the image using **docker run -p 8501:8501 face-mask-detection**
You can now run the app through the Local URL localhost:8501
