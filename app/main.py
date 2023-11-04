from fastapi import FastAPI , Request
import cv2
import numpy as np
import base64


app = FastAPI()

# Load the image as grayscale
def gethog(img_gray):
    s = (128,128)
    new_img= cv2.resize(img_gray, s, interpolation= cv2.INTER_AREA)
    win_size = new_img.shape
    cell_size = (8, 8)
    block_size = (16, 16)
    block_stride = (8, 8)
    num_bins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)
    hog_descriptor = hog.compute(new_img)
    return hog_descriptor

def read64(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    return img

@app.get("/api/gethog")
async def read_str(request : Request):
    item = await request.json()
    item_str = item['img']
    img = read64(item_str)
    hog = gethog(img)
    return {"hog":hog.tolist()}
