# Train & Run

## Install
pip install -r requirements.txt

## Train (keypoints)
python train_keypoint_classifier.py --csv model/keypoint_classifier/keypoint.csv

## Train (point-history)
python train_point_history_classifier.py --csv model/point_history_classifier/point_history.csv

## Run (with TFLite, multi-threaded)
python app.py --num_threads 4

## Use Keras backend (optional)
python app.py --kp_backend keras --kp_model_path model/keypoint_classifier/keypoint_classifier.keras
