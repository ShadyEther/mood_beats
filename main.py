import numpy as np
import cv2
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import streamlit as st

# Define the emotions.
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Music mapping
emo_ID={
    'Angry':"3VAUHKo1HsHKmkybpDAMH2",
    'Disgust':"3VAUHKo1HsHKmkybpDAMH2",
    'Fear':"3VAUHKo1HsHKmkybpDAMH2",
    'Happy':"3VAUHKo1HsHKmkybpDAMH2",
    'Neutral':"3VAUHKo1HsHKmkybpDAMH2",
    'Sad':"3VAUHKo1HsHKmkybpDAMH2",
    "Surprise":"3VAUHKo1HsHKmkybpDAMH2"
}
# Load model.
classifier = load_model('model_78.h5')
classifier.load_weights("model_weights_78.h5")

# Load face using OpenCV
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")

class VideoTransformer:
    def transform(self, frame):
        uploaded_file=frame
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        # image gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(image=img_gray, scaleFactor=1.3, minNeighbors=5)
        output=None
        out={
            'image':img,
            'emotion':output
        }
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 255), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_labels[maxindex]
                output = str(finalout)
                out['emotion']=output
            label_position = (x, y - 10)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out['image']=img
        return out

def main():
    st.markdown(f"""
    <div style="display:flex; align-items: center;justify-content: center;">
        <h1>Mood Beats</h1>
    </div>
    """,unsafe_allow_html=True)
    st.write("""
    This app uses a pretrained deep learning model to analyze the emotions of people from their facial expressions.Then based on the emotions it suggests a suitable music from Spotify. To use this app...
    1. Turn on the camera
    2. Take your picture
    3. Suggested songs will start playing
    """)
    togg=st.toggle("Toggle On/Off", value=False, key="cam_togg", )
    out=None
    emotion=None
    if togg:
        feed=st.camera_input("",key="example",disabled=False)
        if feed is not None:
        # To read image file buffer with OpenCV:
            out=VideoTransformer.transform("",feed)
            emotion=out['emotion']
            if emotion is None:
                st.write("No faces detected")
            
                

    if emotion in emotion_labels:
        # st.write("Present")
        st.image(out['image'])
        
        ID=emo_ID[emotion]
        spotify_playlist_url = f"https://open.spotify.com/embed/playlist/{ID}"
        st.markdown(f"""
        <div style="display: flex; justify-content: center; align-items: center; height: 500px;">
            <iframe src="{spotify_playlist_url}" width="95%" height="500px" frameborder="0" allowtransparency="true" allow="autoplay; clipboard-write; encrypted-media;"></iframe>
        </div>
        """, unsafe_allow_html=True)
    
        
    


if __name__ == "__main__":
    main()
