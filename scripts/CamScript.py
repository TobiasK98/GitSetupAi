import numpy as np
import os
import tensorflow as tf
import pathlib
import glob
import matplotlib.pyplot as plt
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from google.colab.patches import cv2_imshow
from operator import itemgetter, attrgetter
from PIL import Image



while "models" in pathlib.Path.cwd().parts:
    os.chdir('..')

# function for loading our speed sign pictures for displaying the current speed
def load_images(): 

    image_list = []
    for filename in glob.glob('/content/gdrive/MyDrive/Speed_signs/*.jpg'):
        im=Image.open(filename)
        image_list.append((filename, im))
    return image_list
        

def load_model(model_name):
    # Provide the path where the trained model directory is located
    model_dir = "/content/gdrive/MyDrive/" + model_name
    model_dir = pathlib.Path(model_dir) / "saved_model"

    model = tf.saved_model.load(str(model_dir))

    return model


PATH_TO_LABELS = '/content/Tensorflow/workspace/training_demo/annotations/label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

model_name = "my_ssd_resnet50_v1_fpn"
detection_model = load_model(model_name)


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict


def show_inference(model, frame):
    # take the frame from webcam feed and convert that to array
    image_np = np.array(frame)
    # Actual detection.

    output_dict = run_inference_for_single_image(model, image_np)
    
    detect_speed(output_dict)

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=5)

    return (image_np)


# Now we open the webcam and start detecting objects
import cv2

cap = cv2.VideoCapture("/content/gdrive/MyDrive/TestVideo7.mp4")

#create VideoWriter Object
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (1280, 720))

def run_inference(model, cap):
    image_list = load_images()
    speed = None
    while cap.isOpened():

        ret, image_np = cap.read()

        if ret:
            # Actual detection.
            output_dict = run_inference_for_single_image(model, image_np)

            #Create indexes list of element with a score > 0.5
            indexes = [k for k,v in enumerate(output_dict['detection_scores']) if (v > 0.5)]
            
            #Number of entities
            num_entities = len(indexes)
            
            #Extract the class id
            class_id = itemgetter(indexes)(output_dict['detection_classes'])
            scores = itemgetter(indexes)(output_dict['detection_scores'])
          
            #Convert the class id in their name
            class_names = []

            if num_entities == 1:
                class_names.append(category_index[class_id[0]]['name'])
                class_name = str(class_names)
            else:
                for i in range(len(indexes)):
                    class_names.append(category_index[class_id[i]]['name'])
                                
            

            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                instance_masks=output_dict.get('detection_masks_reframed', None),
                use_normalized_coordinates=True,
                line_thickness=8)

            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

            # getting the png for our prediction
            cam_im = Image.fromarray(image_np)
            for j in range(len(class_names)):
                for k in range(len(image_list)):
                    if(class_names[j] == image_list[k][0].split("/")[-1][:-4]):
                        speed = image_list[k][1]
                        speed = speed.resize((100, 100))
                        break
            
            # pasting the current speed into our picture(numpyarray)
            if(speed):
                cam_im.paste(speed,(0,0))

            image_np = np.array(cam_im)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            # writing our video frame by frame into a new file
            out.write(image_np)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cap.release()
                out.release()
                cv2.destroyAllWindows()
                break
        else: 
            break
            
    cap.release()
    out.release()
run_inference(detection_model, cap)
