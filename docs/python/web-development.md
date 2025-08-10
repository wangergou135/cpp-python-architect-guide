# Web Development with Python

## FastAPI Framework
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    id: int
    name: str
    price: float

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    # Example endpoint
    return {"item_id": item_id}
```

## Django Framework
```python
# settings.py
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'myapp',
]

# models.py
from django.db import models

class Product(models.Model):
    name = models.CharField(max_length=100)
    price = models.DecimalField(max_digits=10, decimal_places=2)
```

## Flask Framework
```python
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/api/v1/hello')
def hello_world():
    return jsonify({"message": "Hello, World!"})
```

*Last updated: 2025-08-10 11:10:00 UTC*