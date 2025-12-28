# fastapi_demo.py
from fastapi import FastAPI
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(title="Demo FastAPI App", version="1.0")

# Define a data model for POST requests
class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI demo!"}

# Example path parameter
@app.get("/items/{item_id}")
def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "query": q}

# Example POST request
@app.post("/items/")
def create_item(item: Item):
    total_price = item.price + (item.tax if item.tax else 0)
    return {
        "name": item.name,
        "price": item.price,
        "tax": item.tax,
        "total_price": total_price,
    }

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok"}
