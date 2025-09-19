# -*- coding: utf-8 -*-

import os

import requests
from fastapi import APIRouter, HTTPException, UploadFile, status
from paddleocr import PaddleOCR
from pydantic import BaseModel

from models.OCRModel import *
from models.RestfulModel import *
from utils.ImageHelper import base64_to_ndarray, bytes_to_ndarray

OCR_LANGUAGE = os.environ.get("OCR_LANGUAGE", "ch")

router = APIRouter(prefix="/ocr", tags=["OCR"])

ocr = PaddleOCR(use_angle_cls=True, lang=OCR_LANGUAGE)


@router.get("/health", response_model=RestfulModel, summary="健康检查")
def health():
    return RestfulModel(resultcode=200, message="Success", data="OK")


class PredictPostModel(BaseModel):
    type: str  # 可选值: path, url, base64
    data: str  # 对应的数据


@router.post("/predict", response_model=RestfulModel, summary="统一识别接口")
async def predict(predict_model: PredictPostModel):
    """
    统一识别接口，根据 type 字段选择识别方式:
    - path: data 为本地图片路径
    - url: data 为图片 URL
    - base64: data 为 base64 字符串
    """
    type_ = predict_model.type.lower()
    data_ = predict_model.data
    try:
        if type_ == "path":
            result = ocr.ocr(data_, cls=True)
        elif type_ == "base64":
            img = base64_to_ndarray(data_)
            result = ocr.ocr(img=img, cls=True)
        elif type_ == "url":
            response = requests.get(data_)
            image_bytes = response.content
            if image_bytes.startswith(b"\xff\xd8\xff") or image_bytes.startswith(
                b"\x89PNG\r\n\x1a\n"
            ):
                img = bytes_to_ndarray(image_bytes)
                result = ocr.ocr(img=img, cls=True)
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="请上传 .jpg 或 .png 格式图片",
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="type 字段只支持 path、url、base64",
            )
        return RestfulModel(
            resultcode=200, message="Success", data=result, cls=OCRModel
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"识别失败: {str(e)}",
        )


@router.post("/predict-by-file", response_model=RestfulModel, summary="识别上传文件")
async def predict_by_file(file: UploadFile):
    restfulModel: RestfulModel = RestfulModel()
    if file.filename.endswith((".jpg", ".png")):  # 只处理常见格式图片
        restfulModel.resultcode = 200
        restfulModel.message = file.filename
        file_data = file.file
        file_bytes = file_data.read()
        img = bytes_to_ndarray(file_bytes)
        result = ocr.ocr(img=img, cls=True)
        restfulModel.data = result
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="请上传 .jpg 或 .png 格式图片",
        )
    return restfulModel
