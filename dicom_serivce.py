import numpy as np
import pydicom
from PIL import Image
import uuid


IMG_HEIGHT = 320
IMG_WIDTH = 320


def open_image(file1):
    ds = pydicom.dcmread(file1)
    im = ds.pixel_array.astype(float)
    rescaled_image = (np.maximum(im, 0) / im.max()) * 255
    img = Image.fromarray(rescaled_image)
    img = img.resize((IMG_HEIGHT, IMG_WIDTH), Image.LANCZOS)
    array = np.array(img)
    return array


def get_patient_info(file):
    ds = pydicom.dcmread(file)

    age = ds.PatientAge
    sex = ds.PatientSex
    weight = ds.PatientWeight
    size = ds.PatientSize

    age = int(age.replace("Y", ""))
    if sex == "M":
        sex = "Мужчина"
    else:
        sex = "Женщина"

    weight = int(weight)
    size = float(size) * 100.

    pixel_spacing = float(ds.PixelSpacing[0])

    return {'age': age, 'sex': sex, 'weight': weight, 'size': size, 'pixel_spacing': pixel_spacing}


def open_png(file1):
    img = Image.open(file1)
    arr1 = np.array(img).astype(float)
    rescaled_image = (np.maximum(img, 0) / arr1.max()) * 255
    img = Image.fromarray(rescaled_image)
    img = img.resize((IMG_HEIGHT, IMG_WIDTH), Image.LANCZOS)
    array = np.array(img)
    return array


def compose_images(file1, file2):
    arr1 = open_image(file1)
    arr2 = open_image(file2)

    numpy_array3 = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=float)
    numpy_array3[:, :, 0] = arr1
    numpy_array3[:, :, 1] = arr2 * 2 - arr1
    numpy_array3[:, :, 2] = arr1 - arr2
    numpy_array3 = np.clip(numpy_array3, 0, 255)
    numpy_array3 = numpy_array3.astype(np.uint8)
    img3 = Image.fromarray(numpy_array3.astype(np.uint8))
    file_title = str(uuid.uuid4())
    out_file_name = 'output/' + file_title + '.png'
    img3.save(out_file_name)
    return file_title


def convert_to_image(file):
    arr = open_image(file)
    numpy_array = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=float)
    numpy_array[:, :, 0] = arr
    numpy_array[:, :, 1] = arr
    numpy_array[:, :, 2] = arr
    numpy_array3 = np.clip(numpy_array, 0, 255)
    numpy_array3 = numpy_array3.astype(np.uint8)
    img3 = Image.fromarray(numpy_array3.astype(np.uint8))
    file_title = str(uuid.uuid4())
    out_file_name = 'output/' + file_title + '.png'
    img3.save(out_file_name)
    return file_title
