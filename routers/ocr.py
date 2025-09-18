# -*- coding: utf-8 -*-

import os
from dataclasses import dataclass
from typing import Literal, Optional

import requests
from fastapi import APIRouter, Form, HTTPException, UploadFile, status
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
    restfulModel = RestfulModel(resultcode=200, message="Success", data="OK")
    return restfulModel


class PredictRequestModel(BaseModel):
    type: Literal["path", "base64", "url"]
    image_path: Optional[str] = None
    base64_str: Optional[str] = None
    image_url: Optional[str] = None


@dataclass
class HandleResult:
    resultcode: int
    message: str
    data: list


def handle_path(image_path: Optional[str]):
    if not image_path:
        raise HTTPException(status_code=400, detail="缺少 image_path 参数")
    result = ocr.ocr(image_path, cls=True)
    return HandleResult(resultcode=200, message="Success", data=result)


def handle_base64(base64_str: Optional[str]):
    if not base64_str:
        raise HTTPException(status_code=400, detail="缺少 base64_str 参数")
    img = base64_to_ndarray(base64_str)
    result = ocr.ocr(img=img, cls=True)
    return HandleResult(resultcode=200, message="Success", data=result)


def handle_url(image_url: Optional[str]):
    if not image_url:
        raise HTTPException(status_code=400, detail="缺少 image_url 参数")
    response = requests.get(image_url)
    image_bytes = response.content
    if image_bytes.startswith(b"\xff\xd8\xff") or image_bytes.startswith(
        b"\x89PNG\r\n\x1a\n"
    ):
        img = bytes_to_ndarray(image_bytes)
        result = ocr.ocr(img=img, cls=True)
        return HandleResult(resultcode=200, message="Success", data=result)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="请上传 .jpg 或 .png 格式图片",
        )


def handle_file(file: Optional[UploadFile]):
    if not file:
        raise HTTPException(status_code=400, detail="缺少文件")
    if file.filename.endswith((".jpg", ".png")):
        file_data = file.file
        file_bytes = file_data.read()
        img = bytes_to_ndarray(file_bytes)
        result = ocr.ocr(img=img, cls=True)
        return HandleResult(resultcode=200, message=file.filename, data=result)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="请上传 .jpg 或 .png 格式图片",
        )


@router.post("/predict", response_model=RestfulModel, summary="推理接口")
async def predict(
    type: str = Form(..., description="图片类型: path, base64, url, file"),
    image_path: Optional[str] = Form(None, description="本地图片路径"),
    base64_str: Optional[str] = Form(None, description="Base64字符串"),
    image_url: Optional[str] = Form(None, description="图片URL"),
    file: Optional[UploadFile] = None,
):
    """
    推理接口，根据 type 参数选择识别方式:
    - path: 传 image_path
    - base64: 传 base64_str
    - url: 传 image_url
    - file: 上传文件
    """
    restfulModel: RestfulModel = RestfulModel()
    try:
        if type == "path":
            result: HandleResult = handle_path(image_path)
        elif type == "base64":
            result: HandleResult = handle_base64(base64_str)
        elif type == "url":
            result: HandleResult = handle_url(image_url)
        elif type == "file":
            result: HandleResult = handle_file(file)
        else:
            raise HTTPException(
                status_code=400, detail="type 参数错误，只能为 path, base64, url, file"
            )
        restfulModel.resultcode = result.resultcode
        restfulModel.message = result.message
        restfulModel.data = result.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR识别失败: {str(e)}")
    return restfulModel
