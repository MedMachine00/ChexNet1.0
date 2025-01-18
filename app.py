import tensorflow as tf 
from tensorflow.keras.models import Model
from keras.models import load_model
import gradio as gr
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.densenet import preprocess_input, decode_predictions
import numpy as np 
from scipy import ndimage
from skimage import exposure
from skimage.transform import resize
from PIL import Image
import matplotlib.pyplot as plt
import cv2
model = load_model('Densenet.h5')
model.load_weights("pretrained_model.h5")
layer_name = 'conv5_block16_concat'
class_names = ['Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Atelectasis', 'Pneumothorax', 'Pleural_Thickening', 'Pneumonia', 'Fibrosis', 'Edema', 'Consolidation', 'No Finding']
def get_gradcam(model, img, layer_name):
    
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    grad_model = Model(inputs=model.inputs, outputs=[model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])

    output = conv_outputs[0]
    grads = tape.gradient(predictions, conv_outputs)[0]
    guided_grads = tf.cast(output > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads

    weights = tf.reduce_mean(guided_grads, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(weights, output), axis=-1)
    heatmap = np.maximum(cam, 0)
    heatmap /= tf.reduce_max(heatmap)
    heatmap_img = plt.cm.jet(heatmap)[..., :3]

    # Load the original image
    original_img = Image.fromarray(img)

    # Resize the heatmap to match the original image size
    heatmap_img = Image.fromarray((heatmap_img * 255).astype(np.uint8))
    heatmap_img = heatmap_img.resize(original_img.size)

    # Overlay the heatmap on the original image
    overlay_img = Image.blend(original_img, heatmap_img, 0.5)

    # Return the overlayed image
    return overlay_img
    
def custom_decode_predictions(predictions, class_labels):
    
    decoded_predictions = []
    for pred in predictions:
        # Get indices of top predicted classes
        top_indices = pred.argsort()[-4:][::-1]  # Change 5 to the number of top classes you want to retrieve
        # Decode each top predicted class
        decoded_pred = [(class_labels[i], pred[i]) for i in top_indices]
        decoded_predictions.append(decoded_pred)
    return decoded_predictions

def classify_image(img):
    img = cv2.resize(img, (540, 540), interpolation=cv2.INTER_AREA)
    img_array = img_to_array(img)
    #img_array = exposure.equalize_hist(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)                         
    


    predictions1 = model.predict(img_array)
    decoded_predictions = custom_decode_predictions(predictions1, class_names)
    overlay_img = get_gradcam(model, img, layer_name)

    # Return the decoded predictions and the overlayed image
    return decoded_predictions, overlay_img
# Gradio interface
iface = gr.Interface(
    fn=classify_image, 
    inputs="image", 
    outputs=["text", "image"],  # Add an "image" output for the overlayed image
    title="Xray Classification - KIMS",
    description="Classify cxr into 'Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Atelectasis', 'Pneumothorax', 'Pleural_Thickening', 'Pneumonia', 'Fibrosis', 'Edema', 'Consolidation', No Finding. Built by Dr Sai and Dr Ajavindu"
)


# Launch the interface, 
iface.launch( share=True)
