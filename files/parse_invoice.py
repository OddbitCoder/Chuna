from enum import Enum
from pydantic import BaseModel
from openai import OpenAI
import base64
import argparse
import os

from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
import uvicorn


parser = argparse.ArgumentParser()
parser.add_argument('--key', type=str, default=os.getenv('OPENAI_KEY'), help='OpenAI key')
parser.add_argument('--port', type=int, default=8000, help='Server port')
parser.add_argument('--host', type=str, default="0.0.0.0", help='Server host')

opt = parser.parse_args()


@asynccontextmanager
async def lifespan(the_app):
  print("Initializing...")
  the_app.api_key = opt.key
  yield
  print("Shutting down...")

app = FastAPI(lifespan=lifespan)


class InvoiceItem(BaseModel):
  name: str
  price_per_unit: float
  quantity: float
  quantity_unit: str
  price_total: float

class ParsedInvoice(BaseModel):
  total: float
  items: list[InvoiceItem]

class Category(str, Enum):
    PEOPLE = "PEOPLE"
    INVOICE = "INVOICE"
    BOTH = "BOTH"  

class CategorizerResult(BaseModel):
  category: Category


def categorize_image_internal(base64_image):
  client = OpenAI(api_key=app.api_key)

  response = client.beta.chat.completions.parse(
    model="gpt-4o",
    response_format=CategorizerResult,
    messages=[
      {
        "role": "system",
        "content": "Categorize the given image into either PEOPLE, INVOICE or BOTH, depending on what the image contains.",
      },
      {
        "role": "user",
        "content": 
        [{
            "type": "image_url",
            "image_url": 
            {
              "url": f"data:image/jpeg;base64,{base64_image}"
            }
        }]
      }
    ],
  )

  return response.choices[0].message.parsed


def parse_invoice_internal(base64_image):
  client = OpenAI(api_key=app.api_key)

  response = client.beta.chat.completions.parse(
    model="gpt-4o",
    response_format=ParsedInvoice,
    messages=[
      {
        "role": "system",
        "content": "Parse the given invoice.",
      },
      {
        "role": "user",
        "content": 
        [{
            "type": "image_url",
            "image_url": 
            {
              "url": f"data:image/jpeg;base64,{base64_image}"
            }
        }]
      }
    ],
  )

  return response.choices[0].message.parsed


class InvoiceImageRequest(BaseModel):
    invoice_image: str

class CategorizeImageRequest(BaseModel):
    image: str


@app.post("/parse_invoice", response_model=ParsedInvoice)
async def parse_invoice(invoice_image: UploadFile = File(...)):
  """Parses the given invoice."""
  try:
    image_data = await invoice_image.read()

    base64_image = base64.b64encode(image_data).decode('utf-8')

    return parse_invoice_internal(base64_image)

  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))   

@app.post("/parse_invoice_b64", response_model=ParsedInvoice)
async def parse_invoice_b64(payload: InvoiceImageRequest):
  """Parses the given invoice."""
  invoice_image = payload.invoice_image
  try:
    return parse_invoice_internal(invoice_image)

  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))   


@app.post("/categorize_image", response_model=CategorizerResult)
async def parse_invoice(image: UploadFile = File(...)):
  """Categorizes the given image."""
  try:
    image_data = await image.read()

    base64_image = base64.b64encode(image_data).decode('utf-8')

    return categorize_image_internal(base64_image)

  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))   

@app.post("/categorize_image_b64", response_model=CategorizerResult)
async def parse_invoice_b64(payload: CategorizeImageRequest):
  """Categorizes the given image."""
  image = payload.image
  try:
    return categorize_image_internal(image)

  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e)) 


if __name__ == '__main__':
  uvicorn.run(app, host=opt.host, port=opt.port)