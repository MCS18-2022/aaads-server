import firebase_admin
from firebase_admin import credentials, storage, db
import cv2
from pathlib import Path
import os
from predictor import predict_with_inference_time

cred = credentials.Certificate("service_account_key.json")
firebase_admin.initialize_app(cred, {
    "storageBucket": "aaads-demo.appspot.com"
})

bucket = storage.bucket()


realtime_db_ref = db.reference("/", url='https://aaads-demo-default-rtdb.firebaseio.com/')


def download_image(path):
    blob = bucket.blob(path)
    blob.download_to_filename(path)



if __name__ == "__main__":

    print("Server started listening...")
    
    while True:
        
        pending = realtime_db_ref.child("pending").get()
        
        if pending is not None:
            
            print("Pending:", pending)
            
            Path("video_capture").mkdir(exist_ok=True)
            path = Path("video_capture", f"{pending['id']}.png")
            download_image(str(path))
            frame = cv2.imread(str(path))
            preds, time_taken = predict_with_inference_time(frame)
            realtime_db_ref.child(path.stem).set({
                "predictions": preds,
                "inference_time": time_taken
            })
            realtime_db_ref.child("pending").delete()
        
        