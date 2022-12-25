import cv2
import pickle
import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.cluster import KMeans
import pandas as pd
import streamlit as st
from sklearn.neighbors import NearestNeighbors
import torch
import open_clip
from PIL import Image, ImageOps


def createDatabase():
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32')
    imgPaths = []
    imgVectors = []

    print("Pushing images to vectors...")
    with torch.no_grad():
        for file in tqdm(glob("testData/*")):
            img = preprocess(Image.open(file)).unsqueeze(0)
            imgVector = model.encode_image(img).detach().numpy()[0]
            imgPaths.append(file)
            imgVectors.append(imgVector)

    print("Creating database file...")
    database = pd.DataFrame({"path": imgPaths, "vector": imgVectors})
    database.to_pickle("database.pickle")
    print("Sucessfully created database file !")


@st.cache(allow_output_mutation=True)
def getNeighbours(database):
    neighbours = NearestNeighbors(n_neighbors=5, metric="cosine")
    neighbours.fit(np.stack(database["vector"].to_numpy()))
    return neighbours


# createDatabase()


@st.cache(allow_output_mutation=True)
def getData():
    database = pd.read_pickle("database.pickle")
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32')
    tokenizer = open_clip.get_tokenizer('ViT-B-32-quickgelu')
    neighbours = getNeighbours(database)
    return database, model, neighbours, preprocess, tokenizer


database, model, neighbours, preprocess, tokenizer = getData()
st.title("Search for similar images")
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "webp", "tiff"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    img = ImageOps.exif_transpose(img)
    with torch.no_grad():
        searchImage = preprocess(img).unsqueeze(0)
        imgVector = model.encode_image(searchImage).detach().numpy()

    neighboursIndices = neighbours.kneighbors(imgVector, return_distance=False)[0]
    similarImgPaths = np.hstack(database.loc[neighboursIndices, ["path"]].values)

    for imgPath in similarImgPaths:
        st.image(imgPath, caption=imgPath)


