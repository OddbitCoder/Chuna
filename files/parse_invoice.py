from enum import Enum
from typing import List
from pydantic import BaseModel
from openai import OpenAI
import base64
import argparse
import os
from datetime import datetime

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
  price_total: float

class ParsedInvoice(BaseModel):
  total: float
  timestamp: str
  invoice_number: str
  items: list[InvoiceItem]

class ParsedInvoiceList(BaseModel):
  invoices: List[ParsedInvoice]

class Category(str, Enum):
    PEOPLE = "PEOPLE"
    INVOICE = "INVOICE"
    BOTH = "BOTH"  

class CategorizerResult(BaseModel):
  category: Category

class FaceRecognitionItem(BaseModel):
  x: int
  y: int
  width: int
  height: int
  id: str
  distance: float
  threshold: float
  verified: bool

class FaceRecognitionResultForAgg(BaseModel):
  items: List[FaceRecognitionItem]

class AggregatePeopleResult(BaseModel):
  verified: List[str]
  unverified: List[str]
  other: List[str]


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
    response_format=ParsedInvoiceList,
    messages=[
      {
        "role": "system",
        "content": "Parse all the invoices in the image.",
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


@app.post("/parse_invoice", response_model=ParsedInvoiceList)
async def parse_invoice(invoice_image: UploadFile = File(...)):
  """Parses the given invoice."""
  try:
    image_data = await invoice_image.read()

    base64_image = base64.b64encode(image_data).decode('utf-8')

    return parse_invoice_internal(base64_image)

  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))   

@app.post("/parse_invoice_b64", response_model=ParsedInvoiceList)
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

all_users = [
    "Aleksandar Hristov",
    "Aleksandra Kuštrin Berkopec",
    "Aleš Jelovčan",
    "Aleš Papler",
    "Aleš Tepina",
    "Amer Kočan",
    "Andraz Stibilj",
    "Andrej Špilak",
    "Anze Schwarzmann",
    "David Kaiser",
    "Dragan Jovanovski",
    "Dragan Radaković",
    "Erik Božič",
    "Gregor Mohorko",
    "Gregor Slavec",
    "Gregor Tušar",
    "Jaka Ambrus",
    "Jaka Žužek",
    "Jan Ivanović",
    "Jan Stupica",
    "Jošt Bizjak",
    "Kristina Berčič",
    "Lenart Dolžan",
    "Luka Lačan",
    "Marko Vončina",
    "Matija Pestotnik",
    "Matjaž Juršič",
    "Miha Grčar",
    "Mirko Razpet",
    "Mirt Hlaj",
    "Mitja Belak",
    "Mitja Ristič",
    "Neva Kozelj",
    "Niko Sobočan",
    "Rok Bajec",
    "Rok Rejc",
    "Rok Tomc",
    "Sašo Rutar",
    "Simon Jazbar",
    "Svetlana Sapelova",
    "Urban Ambrož",
    "Urban Džindžinovič",
    "Uroš Ipavec",
    "Uroš Nabernik",
    "Vid Megušar",
    "Vid Stoschitzky",
    "Ziga Tartara",
    "Žiga Povalej"
]

def aggregate_recogniction_lists(results_list):
  unmatched_faces = []
  verified_users = []
  verified_users_final = []
  suggestion_list = []
  for faceDict in results_list:
    for user_recognition in faceDict.items:
      print(user_recognition.verified)
      verified = user_recognition.verified
      user_name = user_recognition.id
      distance = user_recognition.distance
      if verified:
        verified_users.append((distance, user_name))
      else:
        unmatched_faces.append((distance, user_name))
  for distance, name in sorted(verified_users):
    if name not in verified_users_final:
      verified_users_final.append(name)
  for distance, name in sorted(unmatched_faces):
    if name not in verified_users_final and name not in suggestion_list:
      suggestion_list.append(name)
  other_users = [x for x in all_users if x not in verified_users_final + suggestion_list]
  
  return verified_users_final, suggestion_list, other_users

@app.post("/aggregate_face_results", response_model=AggregatePeopleResult)
async def aggregate_face_results(payload: List[FaceRecognitionResultForAgg]):
  """Aggregate results."""
  print(payload)
  verified_users, suggestion_list, other_users = aggregate_recogniction_lists(payload)
  response = AggregatePeopleResult(
    verified=verified_users,
    unverified = suggestion_list,
    other = other_users
  )
  return response

# @app.post("/expand_invoice_items", response_model=ExpandedInvoice)
# async def aggregate_face_results(payload: ParsedInvoice):
#   """Aggregate results."""
#   itemList = []
#   for i in payload.items:
#     for _ in range(int(i['quantity'])):
#       itemList = []
  


if __name__ == '__main__':
  uvicorn.run(app, host=opt.host, port=opt.port)

  