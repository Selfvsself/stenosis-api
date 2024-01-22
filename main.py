import os
import traceback
import uuid

import cnn_service
import dense_service
import dicom_serivce
import segm_service
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/{item_id}/")
async def main(item_id: str):
    return FileResponse("output/" + item_id + ".png")


@app.get("/")
async def main():
    return FileResponse('index.html')


@app.get("/favicon.ico")
async def main():
    return FileResponse('favicon.png')


@app.get("/patients")
async def main():
    return [
        {"id": "P_039",
         "patient_sex": "Мужчина",
         "patient_age": 47,
         "patient_weight": 80,
         "patient_height": 180,
         "disks": [
             {
                 "id": "P_039_D_008",
                 "t1_img": "T1_P_039_D_008",
                 "t2_img": "T2_P_039_D_008",
                 "data": "P_039_D_008"
             },
             {
                 "id": "P_039_D_013",
                 "t1_img": "T1_P_039_D_013",
                 "t2_img": "T2_P_039_D_013",
                 "data": "P_039_D_013"
             }
         ]},
        {"id": "P_165",
         "patient_sex": "Мужчина",
         "patient_age": 39,
         "patient_weight": 80,
         "patient_height": 180,
         "disks": [
             {
                 "id": "P_165_D_003",
                 "t1_img": "T1_P_165_D_003",
                 "t2_img": "T2_P_165_D_003",
                 "data": "P_165_D_003"
             },
             {
                 "id": "P_165_D_013",
                 "t1_img": "T1_P_165_D_013",
                 "t2_img": "T2_P_165_D_013",
                 "data": "P_165_D_013"
             }
         ]},
        {"id": "P_173",
         "patient_sex": "Женщина",
         "patient_age": 67,
         "patient_weight": 95,
         "patient_height": 168,
         "disks": [
             {
                 "id": "P_173_D_008",
                 "t1_img": "T1_P_173_D_008",
                 "t2_img": "T2_P_173_D_008",
                 "data": "P_173_D_008"
             },
             {
                 "id": "P_173_D_013",
                 "t1_img": "T1_P_173_D_013",
                 "t2_img": "T2_P_173_D_013",
                 "data": "P_173_D_013"
             }
         ]},
        {"id": "P_415",
         "patient_sex": "Мужчина",
         "patient_age": 41,
         "patient_weight": 80,
         "patient_height": 180,
         "disks": [
             {
                 "id": "P_415_D_015",
                 "t1_img": "T1_P_415_D_015",
                 "t2_img": "T2_P_415_D_015",
                 "data": "P_415_D_015"
             },
             {
                 "id": "P_415_D_018",
                 "t1_img": "T1_P_415_D_018",
                 "t2_img": "T2_P_415_D_018",
                 "data": "P_415_D_018"
             }
         ]},
        {"id": "P_425",
         "patient_sex": "Женщина",
         "patient_age": 61,
         "patient_weight": 80,
         "patient_height": 180,
         "disks": [
             {
                 "id": "P_425_D_011",
                 "t1_img": "T1_P_425_D_011",
                 "t2_img": "T2_P_425_D_011",
                 "data": "P_425_D_011"
             },
             {
                 "id": "P_425_D_014",
                 "t1_img": "T1_P_425_D_014",
                 "t2_img": "T2_P_425_D_014",
                 "data": "P_425_D_014"
             }
         ]}
    ]


@app.get("/patients/data/{item_id}")
async def main(item_id: str):
    return FileResponse("patients/data/" + item_id + ".zip")


@app.get("/patients/img/{item_id}")
async def main(item_id: str):
    return FileResponse("patients/img/" + item_id + ".png")


@app.get("/patients/predict/{item_id}")
async def main(item_id: str):
    path = os.path.join('patients', 'predict', item_id)
    file_t1 = os.path.join(path, 'T1.ima')
    file_t2 = os.path.join(path, 'T2.ima')
    return get_predict_result(file_t1, file_t2)


@app.post('/')
async def predict(file_t1: UploadFile = File(...),
                  file_t2: UploadFile = File(...)):
    file_name_t1 = 'input/' + str(uuid.uuid4()) + file_t1.filename
    file_name_t2 = 'input/' + str(uuid.uuid4()) + file_t1.filename
    with open(file_name_t1, "wb") as buffer:
        buffer.write(await file_t1.read())
    with open(file_name_t2, "wb") as buffer:
        buffer.write(await file_t2.read())
    result = get_predict_result(file_name_t1, file_name_t2)
    return result


def get_predict_result(file_name_t1, file_name_t2):
    result = {}
    try:
        file_title_t1 = dicom_serivce.convert_to_image(file_name_t1)
        file_path_t1 = 'output/' + file_title_t1 + '.png'

        info = dicom_serivce.get_patient_info(file_name_t1)
        patient_age = info['age']
        patient_sex = info['sex']
        patient_weight = info['weight']
        patient_height = info['size']
        pixel_spacing = info['pixel_spacing']

        file_title_t2 = dicom_serivce.convert_to_image(file_name_t2)
        file_path_t2 = 'output/' + file_title_t2 + '.png'

        file_title_composite = dicom_serivce.compose_images(
            file_name_t1,
            file_name_t2)
        file_path_composite = 'output/' + file_title_composite + '.png'

        yolo_predict = segm_service.segm_yolo(file_path_composite, pixel_spacing)
        div_area_yolo = float(yolo_predict['div_area'])
        div_dist_yolo = float(yolo_predict['div_dist'])
        is_parsed = yolo_predict['is_segment']
        segm_filename_yolo = yolo_predict['image_segm']
        cropped_image_title = yolo_predict['image_crop']
        cropped_image_path = 'output/' + cropped_image_title + '.png'

        unet_predict = segm_service.segm_unet(file_path_composite, pixel_spacing)
        disk_area_unet = float(unet_predict['disk_area'])
        canal_area_unet = float(unet_predict['canal_area'])
        div_area_unet = float(unet_predict['div_area'])
        disk_dist_unet = float(unet_predict['disk_dist'])
        canal_dist_unet = float(unet_predict['canal_dist'])
        div_dist_unet = float(unet_predict['div_dist'])
        segm_filename_unet = unet_predict['image_segm']
        segm_params_filename_unet = unet_predict['image_params']

        div_area_avg = 0.5 * div_area_yolo + 0.5 * div_area_unet
        div_dist_avg = 0.5 * div_dist_yolo + 0.5 * div_dist_unet

        resnet_predict = cnn_service.resnet_predict(cropped_image_path)
        resnet_predict = resnet_predict['predict']
        conv_next_predict = cnn_service.conv_next_predict(cropped_image_path)
        conv_next_predict = conv_next_predict['predict']

        stenos_predict = dense_service.dense_predict({
            'div_area': div_area_avg,
            'div_dist': div_dist_avg,
            'resnet': resnet_predict,
            'conv_next': conv_next_predict
        })
        stenos_predict = stenos_predict['predict']

        result = {
            'image_t1': file_title_t1,
            'image_t2': file_title_t2,
            'compose': file_title_composite,
            'is_parsed': is_parsed,
            'segmentation': segm_filename_unet,
            'segmentation_params': segm_params_filename_unet,
            'yolo_segmentation': segm_filename_yolo,
            'patient_age': patient_age,
            'patient_sex': patient_sex,
            'patient_weight': patient_weight,
            'patient_height': patient_height,
            'disk_area': "{:.2f}".format(disk_area_unet),
            'canal_area': "{:.2f}".format(canal_area_unet),
            'div_area': "{:.2f}".format(div_area_unet),
            'disk_dist': "{:.2f}".format(disk_dist_unet),
            'canal_dist': "{:.2f}".format(canal_dist_unet),
            'div_dist': "{:.2f}".format(div_dist_unet),
            'stenos': "{:.2f}".format(stenos_predict),
            'resnet_predict': "{:.2f}".format(resnet_predict),
            'convnext_predict': "{:.2f}".format(conv_next_predict * 100),
        }
    except BaseException as e:
        print(traceback.format_exc())
        result = {
            'error': 'Ошибка обработки снимков'
        }
    return result
