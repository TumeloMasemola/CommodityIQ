#!/usr/bin/env python
# coding: utf-8

# In[1]:




from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import requests
import os
from datetime import datetime, timedelta

app = FastAPI(title="CommodityIQ API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*']
)

models = {
    'gold': joblib.load('gold_xgb_model.pkl'),
    'platinum': joblib.load('platinum_xgb_model.pkl'),
    'oil': joblib.load('oil_xgb_model.pkl'),
    'copper': joblib.load('copper_xgb_model.pkl'),
}

commodity_info = {
    'gold': {'name': 'Gold', 'unit': 'oz', 'currency': 'USD'},
    'platinum': {'name': 'Platinum', 'unit': 'oz', 'currency': 'USD'},
    'oil': {'name': 'Crude Oil', 'unit': 'barrel', 'currency': 'USD'},
    'copper': {'name': 'Copper', 'unit': 'lb', 'currency': 'USD'},
}

# ============================================
# API KEYS — set these in Railway Variables
# ============================================
GOLD_API_KEY = os.environ.get("GOLD_API_KEY")  # goldapi.io
EIA_API_KEY = os.environ.get("EIA_API_KEY")    # eia.gov

# ============================================
# PYDANTIC MODELS
# ============================================

class PredictionRequest(BaseModel):
    commodity: str
    forecast_days: int = 30

class PredictionResponse(BaseModel):
    commodity: str
    current_price: float
    current_date: str
    forecast_date: str
    predicted_price: float
    change_amount: float
    change_percent: float
    signal: str
    confidence: float
    recommendations: list

class DashboardResponse(BaseModel):
    commodities: list

# ============================================
# LIVE PRICE FETCHERS
# ============================================

def get_metals_price(symbol: str) -> float:
    try:
        response = requests.get(
            f"https://www.goldapi.io/api/{symbol}/USD",
            headers={"x-access-token": GOLD_API_KEY},
            timeout=5
        )
        return float(response.json()['price'])
    except:
        return None

def get_oil_price() -> float:
    try:
        url = f"https://api.eia.gov/v2/petroleum/pri/spt/data/?api_key={EIA_API_KEY}&frequency=daily&data[0]=value&sort[0][column]=period&sort[0][direction]=desc&length=1"
        response = requests.get(url, timeout=5)
        return float(response.json()['response']['data'][0]['value'])
    except:
        return None

def get_current_price(commodity: str) -> float:
    fallback = {
        'gold': 5040,
        'platinum': 980,
        'oil': 85.50,
        'copper': 4.50
    }

    if commodity == 'gold':
        price = get_metals_price('XAU')
    elif commodity == 'platinum':
        price = get_metals_price('XPT')
    elif commodity == 'copper':
        price = get_metals_price('XCU')
    elif commodity == 'oil':
        price = get_oil_price()
    else:
        price = None

    return price if price else fallback.get(commodity, 0)

# ============================================
# HELPER FUNCTIONS
# ============================================

def get_latest_features(commodity: str) -> pd.DataFrame:
    price = get_current_price(commodity)

    if commodity == 'gold':
        features = pd.DataFrame({
            'gold_price': [price],
            'gold_ma_7': [price * 0.993],
            'gold_ma_30': [price * 0.983],
            'gold_ma_200': [price * 0.933],
            'gold_volatility': [50],
            'gold_roc': [0.02],
            'usd_zar': [18.5],
            'usd_zar_ma_30': [18.3],
            'usd_zar_change': [0.01],
            'rsi': [62],
            'month': [datetime.now().month],
            'quarter': [datetime.now().quarter]
        })
    elif commodity == 'platinum':
        features = pd.DataFrame({
            'platinum_price': [price],
            'platinum_ma_7': [price * 0.995],
            'platinum_ma_30': [price * 0.985],
            'platinum_ma_200': [price * 0.959],
            'platinum_volatility': [20],
            'platinum_roc': [0.015],
            'usd_zar': [18.5],
            'usd_zar_ma_30': [18.3],
            'usd_zar_change': [0.01],
            'rsi': [55],
            'month': [datetime.now().month],
            'quarter': [datetime.now().quarter]
        })
    elif commodity == 'oil':
        features = pd.DataFrame({
            'oil_price': [price],
            'oil_ma_7': [price * 0.994],
            'oil_ma_30': [price * 0.988],
            'oil_ma_200': [price * 0.959],
            'oil_volatility': [5.2],
            'oil_roc': [0.015],
            'usd_zar': [18.5],
            'usd_zar_ma_30': [18.3],
            'usd_zar_change': [0.005],
            'rsi': [58],
            'month': [datetime.now().month],
            'quarter': [datetime.now().quarter]
        })
    elif commodity == 'copper':
        features = pd.DataFrame({
            'copper_price': [price],
            'copper_ma_7': [price * 0.996],
            'copper_ma_30': [price * 0.982],
            'copper_ma_200': [price * 0.933],
            'copper_volatility': [0.15],
            'copper_roc': [0.012],
            'usd_zar': [18.5],
            'usd_zar_ma_30': [18.3],
            'usd_zar_change': [0.01],
            'rsi': [60],
            'month': [datetime.now().month],
            'quarter': [datetime.now().quarter]
        })
    return features

def generate_signal(change_percent: float) -> str:
    if change_percent > 5:
        return "STRONG BUY"
    elif change_percent > 2:
        return "BUY"
    elif change_percent < -5:
        return "STRONG SELL"
    elif change_percent < -2:
        return "SELL"
    else:
        return "HOLD"

def generate_recommendations(commodity: str, signal: str, change_percent: float) -> list:
    if commodity in ['gold', 'platinum']:
        if signal in ['BUY', 'STRONG BUY']:
            return [
                "Increase production immediately",
                f"Expect {abs(change_percent):.1f}% price increase",
                "Consider stockpiling refined metal",
                "Hedge 50% of next quarter production"
            ]
        elif signal in ['SELL', 'STRONG SELL']:
            return [
                "Sell current stockpile NOW",
                f"Prices may drop {abs(change_percent):.1f}%",
                "Reduce production temporarily",
                "Wait for market recovery"
            ]
        else:
            return [
                "Maintain current production levels",
                "Market is stable - no urgent action",
                "Monitor closely for changes"
            ]
    elif commodity == 'oil':
        if signal in ['BUY', 'STRONG BUY']:
            return [
                "HEDGE fuel costs immediately!",
                f"Oil may rise {abs(change_percent):.1f}%",
                "Lock in current fuel contracts",
                "Consider passing costs to customers"
            ]
        elif signal in ['SELL', 'STRONG SELL']:
            return [
                "DO NOT HEDGE - wait for lower prices",
                f"Oil costs may drop {abs(change_percent):.1f}%",
                "Delay large fuel purchases",
                "Use spot market instead of contracts"
            ]
        else:
            return [
                "Normal purchasing patterns",
                "Oil prices stable",
                "No urgent hedging needed"
            ]
    elif commodity == 'copper':
        if signal in ['BUY', 'STRONG BUY']:
            return [
                "Purchase copper NOW before price rise",
                f"Prices rising {abs(change_percent):.1f}%",
                "Lock in 6-month supply contracts",
                "Start copper-intensive projects earlier"
            ]
        elif signal in ['SELL', 'STRONG SELL']:
            return [
                "WAIT to purchase copper",
                f"Prices dropping {abs(change_percent):.1f}%",
                "Delay non-urgent purchases",
                "Negotiate lower prices with suppliers"
            ]
        else:
            return [
                "Normal purchasing patterns",
                "Copper prices stable",
                "No urgency"
            ]
    return []

# ============================================
# API ENDPOINTS
# ============================================

@app.get('/')
def home():
    return {
        'name': 'CommodityIQ API',
        'version': '1.0.0',
        'status': 'running',
        'available_commodities': list(models.keys())
    }

@app.get('/commodities')
def list_commodities():
    return {
        'commodities': [
            {'id': key, 'name': info['name'], 'unit': info['unit'], 'currency': info['currency']}
            for key, info in commodity_info.items()
        ]
    }

@app.post('/predict', response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if request.commodity not in models:
        raise HTTPException(status_code=400, detail=f"Commodity '{request.commodity}' not supported.")
    if request.forecast_days not in [7, 30, 60, 90]:
        raise HTTPException(status_code=400, detail="forecast_days must be 7, 30, 60, or 90")

    current_price = get_current_price(request.commodity)
    features = get_latest_features(request.commodity)
    predicted_price = models[request.commodity].predict(features)[0]

    if request.forecast_days == 7:
        predicted_price = current_price + (predicted_price - current_price) * 0.25
    elif request.forecast_days == 60:
        predicted_price = current_price + (predicted_price - current_price) * 1.8
    elif request.forecast_days == 90:
        predicted_price = current_price + (predicted_price - current_price) * 2.5

    change_amount = predicted_price - current_price
    change_percent = (change_amount / current_price) * 100
    signal = generate_signal(change_percent)
    recommendations = generate_recommendations(request.commodity, signal, change_percent)
    current_date = datetime.now()
    forecast_date = current_date + timedelta(days=request.forecast_days)

    return {
        'commodity': request.commodity,
        'current_price': round(current_price, 2),
        'current_date': current_date.strftime('%Y-%m-%d'),
        'forecast_date': forecast_date.strftime('%Y-%m-%d'),
        'predicted_price': round(predicted_price, 2),
        'change_amount': round(change_amount, 2),
        'change_percent': round(change_percent, 2),
        'signal': signal,
        'confidence': 0.85,
        'recommendations': recommendations
    }

@app.get('/dashboard', response_model=DashboardResponse)
def get_dashboard():
    commodities_data = []
    for commodity_id in models.keys():
        try:
            current_price = get_current_price(commodity_id)
            features = get_latest_features(commodity_id)
            predicted_price = models[commodity_id].predict(features)[0]
            change_amount = predicted_price - current_price
            change_percent = (change_amount / current_price) * 100
            signal = generate_signal(change_percent)
            commodities_data.append({
                'id': commodity_id,
                'name': commodity_info[commodity_id]['name'],
                'current_price': round(current_price, 2),
                'predicted_price_30d': round(predicted_price, 2),
                'change_percent_30d': round(change_percent, 2),
                'signal': signal,
                'unit': commodity_info[commodity_id]['unit'],
                'currency': commodity_info[commodity_id]['currency']
            })
        except Exception as e:
            print(f"Error processing {commodity_id}: {e}")
            continue
    return {'commodities': commodities_data}

@app.get('/health')
def health_check():
    return {'status': 'healthy', 'models_loaded': len(models)}




# In[ ]:




