import firebase_admin
from firebase_admin import credentials, storage, db
import cv2
from pathlib import Path
import os
from predictor import predict_without_inference_time


cred = credentials.Certificate("service_account_key.json")
firebase_admin.initialize_app(cred, {
    "storageBucket": "aaads-demo.appspot.com"
})
bucket = storage.bucket()

def download_image(path):
    blob = bucket.blob(path)
    blob.download_to_filename(path)

def upload_vid_preds(vid_name, preds):
    ref = db.reference("/", url='https://aaads-demo-default-rtdb.firebaseio.com/')
    ref.set({vid_name: preds})


if __name__ == "__main__":    
    try:
        print("Server started listening")
        while True:
            blobs = list(bucket.list_blobs(prefix="pending/"))
            if len(blobs) > 0:
                task, *_ = blobs
                task_name = Path(task.name).stem
                # Consume the job.
                bucket.delete_blob(task.name)
                
                # Handle the job here
                
                print("Processing", task_name, "...")
                
                # Download the raw frames, and pass it to the model for inferencing.
                if not os.path.exists(task_name):
                    os.mkdir(task_name)
                image_blobs = bucket.list_blobs(prefix=f"{task_name}/")
                vid_preds = []
                for path in map(lambda b: b.name, image_blobs):
                    print("Downloading", path, "...")
                    download_image(path)
                    print("Inferencing", path, "...")
                    frame = cv2.imread(path)
                    frame_preds = predict_without_inference_time(frame)
                    print("\n", frame_preds, "\n")
                    vid_preds.append(frame_preds)
                print("Done processing", task_name)
                
                print("\n", vid_preds, "\n")
                
                # Uploading model prediction for video to Firebase realtime DB.
                print("Uploading model prediction...")
                upload_vid_preds(task_name, vid_preds)
                print("Uploaded for", task_name)
    
    except KeyboardInterrupt:
        print("\nServer halted by user manually.")
