# Road Sign Detection App Using tkinter for UI

A real-time road sign detection application using YOLOv8 trained on the GTSRB dataset.

## Features
- Real-time road sign detection using webcam
- Support for 43 different traffic sign classes
- Display of sign name, description, and detection confidence

## Requirements
- Python 3.8+
- OpenCV
- Streamlit
- Ultralytics YOLOv8
- Other dependencies listed in requirements.txt

## Installation

1. Clone the repository:
```bash
git clone https://github.com/wabbit12/AQROAD-tkinter or your forked repository.
cd AQROAD-tkinter
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```


## Usage
Run the application using Python:
```bash
python app.py
```

## If Error
If getting error in ultralytics/models/best.pt:
```bash
git lfs install
```

## Model Training
The model was trained on the GTSRB dataset using YOLOv8. The training data is not included in this repository due to size constraints.

