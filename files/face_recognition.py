import os
from typing import List, Optional
import traceback
import json
import argparse

import base64

from dataclasses import dataclass

from pydantic import BaseModel

from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
import uvicorn

from deepface.detectors import FaceDetector
from deepface import DeepFace
from deepface.commons import functions, distance as dst

import cv2
import numpy as np

import tensorflow as tf

tf_version = tf.__version__
tf_major_version = int(tf_version.split(".", maxsplit=1)[0])
tf_minor_version = int(tf_version.split(".")[1])

if tf_major_version == 1:
    from keras.preprocessing import image
elif tf_major_version == 2:
    from tensorflow.keras.preprocessing import image


parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=8000, help='server port')
parser.add_argument('--gpu', action='store_true', help='enable GPU support (off by default)')
parser.add_argument('--gpu-mem', type=int, default=0, help='GPU memory limit (0 or negative = auto-grow)')
parser.add_argument('--detector', type=str, default='yolov8', help='face detection model (ssd, dlib, mtcnn, retinaface, mediapipe, yolov8, yunet)') 
parser.add_argument("--model", type=str, default='ArcFace', help='face recognition model (VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, ArcFace, Dlib, SFace)') 
parser.add_argument("--db", type=str, default='/files/db', help='known faces database location')
parser.add_argument("--dist", type=str, default='cosine', help='distance metric (cosine, euclidean, euclidean_l2)') 
parser.add_argument("--greedy", action='store_true', help='greedy database search (off by default)')
parser.add_argument("--tcboo", action='store_true', help='new matching algorithm "There Can Be Only One" (off by default; cannot be greedy)')

opt = parser.parse_args()


@asynccontextmanager
async def lifespan(the_app):
    print("Initializing...")
    init(the_app)
    yield
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)


class FaceRecognitionItem(BaseModel):
    x: int
    y: int
    width: int
    height: int
    id: str
    distance: float
    threshold: float
    verified: bool

class FaceRecognitionResult(BaseModel):
    items: List[FaceRecognitionItem]
    out_b64: str

class DetectFacesRequest(BaseModel):
    image: str


def init(the_app):
    gpu_enable, gpu_mem, detector_name, model_name, db_dir, distance_metric, greedy_search = opt.gpu, opt.gpu_mem, opt.detector, opt.model, opt.db, opt.dist, opt.greedy
    if not gpu_enable:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus: 
            if gpu_mem <= 0:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            else: # WARNME: this only affects the first physical device
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=gpu_mem)]
                )
    # set "globals"
    the_app.model_name = model_name
    the_app.db_dir = db_dir
    the_app.distance_metric = distance_metric
    the_app.greedy_search = greedy_search
    the_app.detector_name = detector_name
    the_app.face_detector = FaceDetector.build_model(detector_name)
    the_app.model = DeepFace.build_model(model_name)
    the_app.tcboo = opt.tcboo


def represent_known(
    img_fn
):
    vec_fn = img_fn + f".{app.model_name}.vec"

    # read from cache (if it exists)
    if os.path.isfile(vec_fn):
        with open(vec_fn, 'r') as f:
            vec = json.load(f)
            return vec

    # read image
    with open(img_fn, "rb") as img_f:
        chunk = img_f.read()
        img = cv2.imdecode(np.frombuffer(chunk, dtype=np.uint8), cv2.IMREAD_COLOR)
    
    # detect and extract faces
    try:
        img_objs = extract_faces(img) # WARNME: this sometimes fails
    except:
        os.rename(img_fn, img_fn + ".err");
        return None

    # create and write embedding to file
    for img_content, img_region, _ in img_objs:
        vec = represent(
            img_data=img_content
        )
        vec = vec[0]["embedding"]
        with open(vec_fn, 'w') as f:
            f.write(json.dumps(vec))
        return vec

    return None # we failed to create or read embedding


def represent( # WARNME: Taken from the original code. It is somewhat buggy.
    img_data,
    normalization="base",
):
    resp_objs = []

    target_size = functions.find_target_size(model_name=app.model_name)
    img = img_data.copy()
    # --------------------------------
    if len(img.shape) == 4:
        img = img[0]  # e.g. (1, 224, 224, 3) to (224, 224, 3)
    if len(img.shape) == 3:
        img = cv2.resize(img, target_size)
        img = np.expand_dims(img, axis=0)
    # --------------------------------
    img_region = [0, 0, img.shape[1], img.shape[0]] # WARNME: I think this should be [0, 0, img.shape[2], img.shape[1]]
    img_objs = [(img, img_region, 0)] # WARNME: these are image data, region, confidence, respectively
    # --------------------------------

    for img, region, confidence in img_objs:
        # custom normalization
        img = functions.normalize_input(img=img, normalization=normalization)

        # represent
        if "keras" in str(type(app.model)):
            # new tf versions show progress bar and it is annoying
            embedding = app.model.predict(img, verbose=0)[0].tolist()
        else:
            # SFace and Dlib are not keras models and no verbose arguments
            embedding = app.model.predict(img)[0].tolist()

        resp_obj = {}
        resp_obj["embedding"] = embedding
        resp_obj["facial_area"] = region
        resp_obj["face_confidence"] = confidence
        resp_objs.append(resp_obj)

    return resp_objs


def extract_faces(
    img,
    target_size=(224, 224),
    grayscale=False,
    enforce_detection=True,
    align=True,
):
    # this is going to store a list of img itself (numpy), its region and confidence
    extracted_faces = []

    img_region = [0, 0, img.shape[1], img.shape[0]]

    face_objs = FaceDetector.detect_faces(app.face_detector, app.detector_name, img, align)

    if len(face_objs) == 0 and enforce_detection is False:
        face_objs = [(img, img_region, 0)]

    for current_img, current_region, confidence in face_objs:
        if current_img.shape[0] > 0 and current_img.shape[1] > 0:
            if grayscale is True:
                current_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)

            # resize and padding
            if current_img.shape[0] > 0 and current_img.shape[1] > 0:
                factor_0 = target_size[0] / current_img.shape[0]
                factor_1 = target_size[1] / current_img.shape[1]
                factor = min(factor_0, factor_1)

                dsize = (
                    int(current_img.shape[1] * factor),
                    int(current_img.shape[0] * factor),
                )
                current_img = cv2.resize(current_img, dsize)

                diff_0 = target_size[0] - current_img.shape[0]
                diff_1 = target_size[1] - current_img.shape[1]
                if grayscale is False:
                    # Put the base image in the middle of the padded image
                    current_img = np.pad(
                        current_img,
                        (
                            (diff_0 // 2, diff_0 - diff_0 // 2),
                            (diff_1 // 2, diff_1 - diff_1 // 2),
                            (0, 0),
                        ),
                        "constant",
                    )
                else:
                    current_img = np.pad(
                        current_img,
                        (
                            (diff_0 // 2, diff_0 - diff_0 // 2),
                            (diff_1 // 2, diff_1 - diff_1 // 2),
                        ),
                        "constant",
                    )

            # double check: if target image is not still the same size with target.
            if current_img.shape[0:2] != target_size:
                current_img = cv2.resize(current_img, target_size)

            # normalizing the image pixels
            # what this line doing? must?
            img_pixels = image.img_to_array(current_img)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255  # normalize input in [0, 1]

            # int cast is for the exception - object of type 'float32' is not JSON serializable
            region_obj = {
                "x": int(current_region[0]),
                "y": int(current_region[1]),
                "w": int(current_region[2]),
                "h": int(current_region[3]),
            }

            extracted_face = [img_pixels, region_obj, confidence]
            extracted_faces.append(extracted_face)

    return extracted_faces


@dataclass
class RecognitionObject:
    vec: any
    x: int
    y: int
    width: int
    height: int
    known_faces: Optional[list] = None

@dataclass
class KnownFace:
    folder: str
    distance: float


def detect_faces_internal(img):
    # detect faces
    img_objs = extract_faces(img) 
    # embed faces
    results = []
    tabu = set()
    if not app.tcboo:
        for img_content, img_region, _ in img_objs:
            vec = represent(
                img_data=img_content
            )
            vec = vec[0]["embedding"]
            min_dist = 1
            best_match = None
            # find matching faces
            for person_fn in os.listdir(app.db_dir):
                if person_fn in tabu:
                    continue # for person_fn
                for img_fn in os.listdir(os.path.join(app.db_dir, person_fn)):
                    if not img_fn.lower().endswith(".vec") and not img_fn.lower().endswith(".err"):
                        vec2 = represent_known(os.path.join(app.db_dir, person_fn, img_fn))
                        if (vec2 != None):
                            print(f"Comparing with {person_fn} ({img_fn})...")
                            if app.distance_metric == "cosine":
                                distance = dst.findCosineDistance(vec, vec2)
                            elif app.distance_metric == "euclidean_l2":
                                distance = dst.findEuclideanDistance(dst.l2_normalize(vec), dst.l2_normalize(vec2))
                            else: # euclidean
                                distance = dst.findEuclideanDistance(vec, vec2)
                            threshold = dst.findThreshold(app.model_name, app.distance_metric)
                            if distance < min_dist:
                                min_dist = distance
                                best_match = FaceRecognitionItem(
                                    x=img_region['x'],
                                    y=img_region['y'],
                                    width=img_region['w'],
                                    height=img_region['h'],
                                    id=person_fn,
                                    distance=distance,
                                    threshold=threshold,
                                    verified=bool(distance <= threshold)
                                )
                                if distance <= threshold and app.greedy_search:
                                    break # for img_fn 
                # for each person_fn continue here...
            # for each embedding continue here...
            if best_match != None:
                results.append(best_match)
                if app.greedy_search and best_match.verified:
                    # put best_match to tabu list
                    tabu.add(best_match.id)
    else:  # ***** NEW MATCHING ALGORITHM *****
        threshold = dst.findThreshold(app.model_name, app.distance_metric)
        recognition_objects = []
        
        for img_content, img_region, _ in img_objs:
            vec = represent(
                img_data=img_content
            )
            recognition_objects.append(RecognitionObject(
                vec=vec[0]["embedding"],
                x=img_region['x'],
                y=img_region['y'],
                width=img_region['w'],
                height=img_region['h']
            ))

        for recognition_obj in recognition_objects:
            known_faces = []
            for folder in os.listdir(app.db_dir):
                min_distance = float('inf')
                for image_path in os.listdir(os.path.join(app.db_dir, folder)):
                    if image_path.lower().endswith(".vec") or image_path.lower().endswith(".err"):
                        continue
                    vec2 = represent_known(os.path.join(app.db_dir, folder, image_path))
                    distance = float('inf')
                    if (vec2 != None):
                        print(f"Comparing with {folder} ({image_path})...")
                        if app.distance_metric == "cosine":
                            distance = dst.findCosineDistance(recognition_obj.vec, vec2)
                        elif app.distance_metric == "euclidean_l2":
                            distance = dst.findEuclideanDistance(dst.l2_normalize(recognition_obj.vec), dst.l2_normalize(vec2))
                        else: # euclidean
                            distance = dst.findEuclideanDistance(recognition_obj.vec, vec2)
                        threshold = dst.findThreshold(app.model_name, app.distance_metric)
                    if distance < min_distance:
                        min_distance = distance
                # here we have min_distance for current folder
                known_face = KnownFace(folder=folder, distance=min_distance)
                known_faces.append(known_face)
            
            recognition_obj.known_faces = sorted(known_faces, key=lambda kf: kf.distance)    
            
        while recognition_objects:
            min_known_face = None
            min_recognition_obj = None
            for recognition_obj in recognition_objects:
                if recognition_obj.known_faces:  # ensure there are known_faces to compare
                    first_known_face = recognition_obj.known_faces[0]
                    if min_known_face is None or first_known_face.distance < min_known_face.distance:
                        min_known_face = first_known_face
                        min_recognition_obj = recognition_obj
            if min_known_face is None:
                break
            results.append(FaceRecognitionItem(
                x=min_recognition_obj.x,
                y=min_recognition_obj.y,
                width=min_recognition_obj.width,
                height=min_recognition_obj.height,
                id=min_known_face.folder,
                distance=min_known_face.distance,
                threshold=threshold,
                verified=bool(min_known_face.distance <= threshold)
            ))
            recognition_objects.remove(min_recognition_obj)
            for rec_obj in recognition_objects:
                # filter out the known_face associated with the selected folder from remaining lists
                rec_obj.known_faces = [kf for kf in rec_obj.known_faces if kf.folder != min_known_face.folder]
        
        for recognition_obj in recognition_objects:
            results.append(FaceRecognitionItem(
                x=recognition_obj.x,
                y=recognition_obj.y,
                width=recognition_obj.width,
                height=recognition_obj.height,
                id="",
                distance=-1,
                threshold=threshold,
                verified=False
            ))

    for item in results:
       color = (0, 255, 0) if item.verified else (0, 0, 255)
       cv2.rectangle(img, (item.x, item.y), (item.x + item.width, item.y + item.height), color=color, thickness=5)
    #cv2.imwrite(os.path.join(app.db_dir, "test.jpg"), img) # WARNME
    _, buffer = cv2.imencode('.jpg', img)
    out_img_base64 = base64.b64encode(buffer).decode('utf-8')
    return FaceRecognitionResult(items=results, out_b64=out_img_base64)


def resize_image(img):
    height, width = img.shape[:2]
    if height > 3000 or width > 3000:
        scaling_factor = 3000 / max(height, width)
        new_dimensions = (int(width * scaling_factor), int(height * scaling_factor))
        img = cv2.resize(img, new_dimensions, interpolation=cv2.INTER_AREA)
    return img


@app.post("/detect_faces_b64", response_model=FaceRecognitionResult)
async def detect_faces_b64(payload: DetectFacesRequest):
    """Endpoint to detect faces in the provided image byte array and return recognition details."""
    try:
        image_data = base64.b64decode(payload.image)
        nparr = np.frombuffer(image_data, dtype=np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = resize_image(img)
        return detect_faces_internal(img)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect_faces", response_model=FaceRecognitionResult)
async def detect_faces(image_file: UploadFile = File(...)):
    """Endpoint to detect faces in the provided image byte array and return recognition details."""
    try:
        image_data = await image_file.read()
        img = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        img = resize_image(img)
        return detect_faces_internal(img)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))   


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=opt.port)
