import firebase_admin
from firebase_admin import credentials, storage, db
import cv2
from pathlib import Path
import os
from predictor import predict_with_inference_time

cred = credentials.Certificate("service_account_key.json")
firebase_admin.initialize_app(cred, {
    "storageBucket": "aggressive-action-detection.appspot.com"
})

bucket = storage.bucket()


realtime_db_ref = db.reference("/", url='https://aggressive-action-detection-default-rtdb.asia-southeast1.firebasedatabase.app/')


def download_image(path):
    blob = bucket.blob(path)
    blob.download_to_filename(path)



if __name__ == "__main__":

    print("Server started listening...")
    
    while True:
        
        pending = realtime_db_ref.child("pending").get()
        
        if pending is not None:
            
            print("Pending:", pending)
            
            path = Path("webcam_up", f"{pending['id']}.png")
            download_image(str(path))
            frame = cv2.imread(str(path))
            preds, time_taken = predict_with_inference_time(frame)
            realtime_db_ref.child(path.stem).set({
                "predictions": preds,
                "inference_time": time_taken
            })
            realtime_db_ref.child("pending").delete()
        
        