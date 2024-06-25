# Real-Time-Facial-Emotion-Detection-using-Webcam
**Import FER13 file from https://www.kaggle.com/datasets/msambare/fer2013**<br />
<br />

**Code for colab:**

!pip install -q kaggle

from google.colab import drive
drive.mount('/content/drive')

!mkdir ~/.kaggle
!cp /content/drive/MyDrive/kaggle.json ~/.kaggle/kaggle.json
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d msambare/fer2013

!unzip -q fer2013.zip<br /><br />
<br />
<br />
***Replace all the file paths with your respective files paths!!***

facial-emotion-detection.ipynb contains the developed model's code.

'best_model.h5' is the final model obtained from the above code.

camera.py conatins the webcam function. After running the code, press 'q' to exit the webcam.
