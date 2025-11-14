# Traffic Sign Detection and Recognition System

This project detects and recognizes traffic signs in real time using a webcam feed. A real-time traffic sign detection and recognition system built using deep learning, OpenCV, and a preprocessed dataset.
It uses a preprocessed traffic-sign dataset and a trained deep-learning model (.h5) to identify signs frame by frame and display the prediction live.

✨ Features

Real-time traffic sign detection using OpenCV

Trained model (.h5) for recognition

Live webcam interface (tsdr_webcam.py)

Simple UI option (tsdr_ui.py)

Easy dataset access (Google Drive)

Clean training script (train_model_from_pickle.py)

📁 Project Structure

TSDR/
│
├── dataset/                 # Place dataset here after downloading
├── model/
│   └── your_model.h5        # Trained model file
│
├── labels/                  # Label mappings
├── requirements.txt         # Dependencies
│
├── train_model_from_pickle.py
├── tsdr_ui.py
├── tsdr_webcam.py
└── README.md

📥 Dataset

The dataset used in this project is large (~4 GB) and taken from a Kaggle collection.
To make it easy for anyone evaluating this project, the dataset is hosted on Google Drive.

Download the dataset here:
➡️ https://drive.google.com/drive/folders/1l3ZqAeMGqoAUxekR83ZcnQewjJCpnaKf?usp=drive_link

After downloading, place the extracted folder inside the project as:

/dataset/


This is the location expected by the code.

🧠 Model

The trained .h5 model is stored in:

/model/


If you want to retrain the model, use:

python train_model_from_pickle.py

▶️ Running the Project
1. Install dependencies
pip install -r requirements.txt

2. Ensure the dataset and model are in the correct folders
/dataset/
/model/your_model.h5

3. Run the real-time webcam demo
python tsdr_webcam.py

4. Or run the simple UI
python tsdr_ui.py

📝 Notes

Make sure your camera is connected for the webcam demo.

The project was trained on the Kaggle “Traffic Signs Preprocessed” dataset.

The dataset is large, so using the Google Drive link is recommended for easy access.

📚 Credits

Dataset source: Traffic Signs Preprocessed on Kaggle   


