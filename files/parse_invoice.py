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


@app.post("/parse_invoice", response_model=ParsedInvoice)
async def parse_invoice(invoice_image: UploadFile = File(...)):
  """Parses the given invoice."""
  try:
    client = OpenAI(api_key=app.api_key)

    image_data = await invoice_image.read()
    #img = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_COLOR)

    base64_image = base64.b64encode(image_data).decode('utf-8')

    response = client.beta.chat.completions.parse(
      model="gpt-4o",
      response_format=ParsedInvoice,
      messages=[
        {
          "role": "system",
          "content": "Parse the given invoice. Make sure that the sum of all items is the same as the total.",
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

  except Exception as e:
    # traceback.print_exc()
    raise HTTPException(status_code=500, detail=str(e))   


if __name__ == '__main__':
  uvicorn.run(app, host="127.0.0.1", port=opt.port)