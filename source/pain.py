import os
import cv2
import rawpy
import torch
import numpy as np
import streamlit as st
import subpain as sb
from torch import nn
from streamlit_image_comparison import image_comparison
from model import UnetTMO

map_location = torch.device('cpu')


def read_image(file):
    raw = rawpy.imread(file)
    raw_image = raw.raw_image_visible.astype(np.float32) / np.iinfo(raw.raw_image_visible.dtype).max
    raw_image = np.expand_dims(raw_image, axis=0)
    bayer_batch = torch.from_numpy(raw_image).unsqueeze(0)
    return bayer_batch, raw


st.title("Image Processing with Streamlit")

uploaded_file = st.file_uploader("Choose an image file (CR2 format)",
                                 type=["cr2", 'arw', 'cr2', 'cr3', 'nef', 'raw', 'raf'])

if uploaded_file is not None:

    fileName = 'temp.' + uploaded_file.name.split(".")[-1]
    with open(fileName, "wb") as f:
        f.write(uploaded_file.getbuffer())

    image, raw = read_image(fileName)
    img = raw.postprocess(use_camera_wb=True, output_bps=16,
                          demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR)
    bgr_out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    original_image_path = "original_image.png"
    cv2.imwrite(original_image_path, bgr_out, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    with torch.no_grad():
        output, _ = sb.model(image)

    output_image = output[0, 0].cpu().numpy()
    output_image = (output_image * np.iinfo(np.uint16).max).astype(np.uint16)
    raw.raw_image_visible[:] = output_image

    post_processed_image = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=16,
                                           demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR)
    post_processed_image = post_processed_image.astype(np.float32) / np.iinfo(post_processed_image.dtype).max

    rgb_out = torch.from_numpy(post_processed_image).permute(2, 0, 1).unsqueeze(0)
    rgb_out = torch.clip(rgb_out[0].permute(1, 2, 0), 0.0, 1.0).numpy() * 255.0
    rgb_out = rgb_out.astype(np.uint8)
    bgr_out = cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)
    processed_image_path = "processed_image.png"
    cv2.imwrite(processed_image_path, bgr_out, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    image_comparison(
        img1=original_image_path,
        img2=processed_image_path,
        label1="Original Image",
        label2="Processed Image"
    )

    with open(original_image_path, "rb") as file:
        st.download_button(
            label="Download Original Image",
            data=file,
            file_name="original_image.png",
            mime="image/png"
        )

    with open(processed_image_path, "rb") as file:
        st.download_button(
            label="Download Processed Image",
            data=file,
            file_name="processed_image.png",
            mime="image/png"
        )

    st.success("Image processed, comparison slider created, and download buttons added!")
else:
    st.info("Please upload an image file.")
