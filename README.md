# VisionAI Gender and Age Analyzer

A Python desktop app that finds faces in a webcam feed or a video
file and estimates gender and age group. Final year project, BS
Computer Science (2021-2025), University of Malakand. Team of two.

## What it does
- Face detection with OpenCV (DNN detector, Haar Cascade as fallback)
- Gender and age group estimation with pre-trained Caffe models by
  Levi and Hassner (CVPR Workshops, 2015)
- Tkinter interface with live counts and charts
- Export of results to CSV and JSON

## Separate experiment: custom gender model
train_gender_model.py trains a small CNN in Keras on 1,412 face
images (695 female, 717 male). Validation
accuracy was about 75% (see training_history.png). This
model is not used inside the app.

## Models
Model files are not included because they are large. Download the
Caffe models and put them in a folder named
`models/`. Also needed there: the OpenCV face detector files.

## How to run
1. Install Python 3.8 or later
2. pip install -r requirements.txt
3. Put the model files in `models/`
4. python index.py

## Known limits
- Counts are per detection, not per person (there is no tracking)
- Works best with clear, front-facing faces in good light
- Age is only a rough bracket, not an exact age

## Author
Khubab Khan
