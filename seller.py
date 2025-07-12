import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from langchain_google_genai import ChatGoogleGenerativeAI
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict
import uvicorn
from enum import Enum
import uuid
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# GLOBAL CONFIGURATION - ENHANCED WITH SELLER MANAGEMENT
# =============================================================================

class GlobalConfig:
    """Enhanced global configuration for supply chain optimization with seller management"""

    # API Configuration
    GOOGLE_API_KEY = "AIzaSyCPbIA3-rJWEZSZAbMCv0fg7RgadQUa5-I"  # Replace with your API key
    LLM_MODEL = "gemini-2.0-flash"
    LLM_TEMPERATURE = 0.1

    # Business Configuration
    CURRENT_DATE = datetime(2025, 12, 15)
    CURRENT_MONTH = 12
    LOCATION = "Mumbai, India"
    NUMBER_OF_STORES = 8
    TOTAL_INVENTORY_CAPACITY = 1000

    # Data Configuration
    DATA_DIRECTORY = "supply_chain_data/"
    PRODUCTS_CSV = "products_inventory.csv"
    DAILY_SALES_CSV = "daily_sales_history.csv"
    MONTHLY_SALES_CSV = "monthly_sales_history.csv"
    SELLERS_CSV = "sellers_database.csv"
    SELLER_RESPONSES_CSV = "seller_responses.csv"
    NEGOTIATIONS_CSV = "negotiations_log.csv"

    # Analysis Configuration
    VELOCITY_ANALYSIS_DAYS = [5, 10, 30]
    SEASONALITY_YEARS = 3
    SAFETY_STOCK_FACTOR = 1.5

    # Seller Management Configuration
    MAX_NEGOTIATION_ROUNDS = 5
    SELLER_RESPONSE_TIMEOUT_HOURS = 24
    RESPONSE_WAIT_TIME_SECONDS = 5  # Time to wait for all responses

# Initialize configuration
config = GlobalConfig()

# Initialize LLM
try:
    llm = ChatGoogleGenerativeAI(
        model=config.LLM_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
        temperature=config.LLM_TEMPERATURE,
        convert_system_message_to_human=True
    )
    logger.info("LLM initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize LLM: {e}")
    llm = None

# =============================================================================
# PYDANTIC MODELS FOR API
# =============================================================================

class ResponseStatus(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    DECLINED = "declined"
    NEGOTIATING = "negotiating"
    COMPLETED = "completed"
    FINAL_ACCEPTED = "final_accepted"

class SellerResponse(BaseModel):
    seller_id: str
    product_id: int
    request_id: str
    response: str  # "yes" or "no"
    quoted_price: Optional[float] = None
    expected_delivery_days: Optional[int] = None
    additional_notes: Optional[str] = ""

class NegotiationMessage(BaseModel):
    seller_id: str
    message: str
    counter_offer_price: Optional[float] = None
    counter_offer_delivery: Optional[int] = None

class InitiateProcessRequest(BaseModel):
    product_id: int
    quantity_needed: int
    urgency: Optional[str] = "normal"

# =============================================================================
# SELLER DATABASE GENERATION
# =============================================================================

def generate_sellers_database():
    """Generate realistic sellers database for Mumbai, India area"""
    
    # Set seed for reproducible results
    np.random.seed(42)
    
    logger.info("🏪 Generating sellers database...")
    
    # Mumbai area locations
    mumbai_areas = [
        "Andheri East", "Bandra West", "Borivali West", "Dadar East", 
        "Goregaon East", "Kandivali West", "Malad West", "Powai",
        "Thane West", "Vashi Navi Mumbai", "Pune Road", "Kalyan"
    ]
    
    # Product-specific seller types
    seller_data = []
    
    products_info = [
        {"id": 5001, "name": "Mangoes (per kg)", "supplier_types": ["Fruit Wholesaler", "Agricultural Supplier", "Farmers Market"]},
        {"id": 5002, "name": "Air Conditioners (1.5 Ton)", "supplier_types": ["Electronics Distributor", "AC Manufacturer", "Appliance Wholesaler"]},
        {"id": 5003, "name": "Winter Jackets", "supplier_types": ["Clothing Manufacturer", "Textile Distributor", "Garment Supplier"]},
        {"id": 5004, "name": "Sweaters", "supplier_types": ["Knitwear Manufacturer", "Textile Supplier", "Clothing Wholesaler"]},
        {"id": 5005, "name": "Bread (per loaf)", "supplier_types": ["Bakery Supplier", "Food Distributor", "Bread Manufacturer"]},
        {"id": 5006, "name": "Cooking Oil (1L)", "supplier_types": ["Oil Distributor", "Food Wholesaler", "FMCG Supplier"]}
    ]
    
    for product in products_info:
        for i in range(3):  # 3 sellers per product
            seller_id = f"SELL_{product['id']}_{i+1:02d}"
            
            # Generate realistic company details
            supplier_type = product["supplier_types"][i]
            area = np.random.choice(mumbai_areas)
            
            # Generate company names based on type and area
            if "Fruit" in supplier_type or "Agricultural" in supplier_type:
                company_names = [f"Maharashtra {supplier_type}", f"{area} Fresh Supplies", f"Mumbai {supplier_type} Co."]
            elif "Electronics" in supplier_type or "AC" in supplier_type:
                company_names = [f"TechnoMart {area}", f"Cool Air {supplier_type}", f"Mumbai Electronics Hub"]
            elif "Clothing" in supplier_type or "Textile" in supplier_type:
                company_names = [f"Fashion Hub {area}", f"Mumbai {supplier_type}", f"Style Craft Industries"]
            elif "Bakery" in supplier_type or "Bread" in supplier_type:
                company_names = [f"Daily Fresh {area}", f"Mumbai Bakers Association", f"Golden Crust Supplies"]
            else:
                company_names = [f"Supreme {supplier_type}", f"{area} Trading Co.", f"Mumbai {supplier_type}"]
            
            company_name = np.random.choice(company_names)
            
            # Generate reliability and pricing based on position (0=premium, 1=mid, 2=budget)
            if i == 0:  # Premium supplier
                reliability_score = np.random.uniform(85, 95)
                price_factor = np.random.uniform(1.05, 1.15)  # 5-15% higher than base
                delivery_reliability = np.random.uniform(90, 98)
            elif i == 1:  # Mid-range supplier
                reliability_score = np.random.uniform(75, 85)
                price_factor = np.random.uniform(0.95, 1.05)  # Around base price
                delivery_reliability = np.random.uniform(80, 90)
            else:  # Budget supplier
                reliability_score = np.random.uniform(65, 75)
                price_factor = np.random.uniform(0.85, 0.95)  # 5-15% lower than base
                delivery_reliability = np.random.uniform(70, 80)
            
            # Generate contact details
            phone_number = f"+91 {np.random.randint(70000, 99999)}{np.random.randint(10000, 99999)}"
            email = f"{seller_id.lower()}@{company_name.lower().replace(' ', '').replace('.', '')}supplies.com"
            
            seller_data.append({
                "seller_id": seller_id,
                "product_id": product["id"],
                "product_name": product["name"],
                "company_name": company_name,
                "supplier_type": supplier_type,
                "location": area + ", Mumbai",
                "contact_person": f"Manager_{seller_id}",
                "phone_number": phone_number,
                "email": email,
                "reliability_score": round(reliability_score, 1),
                "average_delivery_days": np.random.randint(2, 8) if i == 0 else np.random.randint(3, 10) if i == 1 else np.random.randint(4, 12),
                "price_competitiveness": round(price_factor, 3),
                "delivery_reliability_percent": round(delivery_reliability, 1),
                "payment_terms": np.random.choice(["Net 30", "Net 15", "COD", "Advance 50%"]),
                "minimum_order_quantity": np.random.randint(50, 200) if product["id"] in [5001, 5005, 5006] else np.random.randint(5, 25),
                "created_date": config.CURRENT_DATE.strftime('%Y-%m-%d'),
                "last_updated": config.CURRENT_DATE.strftime('%Y-%m-%d'),
                "status": "active"
            })
    
    sellers_df = pd.DataFrame(seller_data)
    
    # Create directory if it doesn't exist
    os.makedirs(config.DATA_DIRECTORY, exist_ok=True)
    
    # Save to CSV
    sellers_df.to_csv(config.DATA_DIRECTORY + config.SELLERS_CSV, index=False)
    logger.info(f"✅ Created {config.SELLERS_CSV} with {len(sellers_df)} sellers")
    
    return sellers_df

def generate_sample_products():
    """Generate sample products data"""
    products_data = {
        'product_id': [5001, 5002, 5003, 5004, 5005, 5006],
        'product_name': [
            'Mangoes (per kg)', 
            'Air Conditioners (1.5 Ton)', 
            'Winter Jackets',
            'Sweaters',
            'Bread (per loaf)', 
            'Cooking Oil (1L)'
        ],
        'quantity_in_inventory': [200, 50, 100, 150, 400, 600],
        'current_quantity_in_store': [50, 10, 20, 30, 80, 100],
        'cost_price': [80, 35000, 1500, 800, 20, 120],
        'selling_price': [120, 45000, 2500, 1200, 35, 150],
        'shelf_life_days': [7, 3650, 730, 730, 3, 730],
        'lead_time_days': [3, 15, 20, 15, 2, 7],
        'category': [
            'Seasonal_HighDemand',
            'Seasonal_LowDemand', 
            'Seasonal_ModerateDemand',
            'Seasonal_ModerateDemand',
            'NonSeasonal_HighDemand', 
            'NonSeasonal_RegularDemand'
        ]
    }
    
    products_df = pd.DataFrame(products_data)
    products_df.to_csv(config.DATA_DIRECTORY + config.PRODUCTS_CSV, index=False)
    logger.info("✅ Created sample products data")
    return products_df

# =============================================================================
# SELLER MANAGEMENT FUNCTIONS
# =============================================================================

def load_sellers_database():
    """Load sellers database from CSV"""
    try:
        df = pd.read_csv(config.DATA_DIRECTORY + config.SELLERS_CSV)
        return df
    except FileNotFoundError:
        logger.warning("❌ Sellers CSV not found. Generating database...")
        return generate_sellers_database()

def load_products_database():
    """Load products database from CSV"""
    try:
        df = pd.read_csv(config.DATA_DIRECTORY + config.PRODUCTS_CSV)
        return df
    except FileNotFoundError:
        logger.warning("❌ Products CSV not found. Generating database...")
        return generate_sample_products()

def get_sellers_for_product(product_id: int):
    """Get all sellers for a specific product"""
    sellers_df = load_sellers_database()
    product_sellers = sellers_df[sellers_df['product_id'] == product_id]
    return product_sellers.to_dict('records')

def send_initial_request_to_sellers(product_id: int, quantity_needed: int, product_name: str):
    """Send initial request to all sellers for a product"""
    
    sellers = get_sellers_for_product(product_id)
    
    if not sellers:
        logger.error(f"❌ No sellers found for product {product_id}")
        return []
    
    logger.info(f"📤 Sending requests to {len(sellers)} sellers for {product_name}")
    
    # Create initial request records
    requests_data = []
    
    for seller in sellers:
        request_id = str(uuid.uuid4())
        
        # Generate initial message using LLM
        initial_message = generate_initial_seller_message(
            seller['company_name'],
            product_name,
            quantity_needed,
            seller['contact_person']
        )
        
        request_data = {
            "request_id": request_id,
            "seller_id": seller['seller_id'],
            "product_id": product_id,
            "product_name": product_name,
            "quantity_requested": quantity_needed,
            "initial_message": initial_message,
            "request_date": config.CURRENT_DATE.strftime('%Y-%m-%d %H:%M:%S'),
            "status": ResponseStatus.PENDING,
            "response_deadline": (config.CURRENT_DATE + timedelta(hours=config.SELLER_RESPONSE_TIMEOUT_HOURS)).strftime('%Y-%m-%d %H:%M:%S')
        }
        
        requests_data.append(request_data)
        
        logger.info(f"   📧 Request sent to {seller['company_name']} ({seller['seller_id']})")
    
    # Save requests to file
    requests_df = pd.DataFrame(requests_data)
    requests_file = config.DATA_DIRECTORY + f"seller_requests_{product_id}_{config.CURRENT_DATE.strftime('%Y%m%d')}.csv"
    requests_df.to_csv(requests_file, index=False)
    
    logger.info(f"💾 Seller requests saved to: {requests_file}")
    
    return requests_data

def generate_initial_seller_message(company_name: str, product_name: str, quantity: int, contact_person: str):
    """Generate initial message to sellers using LLM"""
    
    if llm is None:
        # Fallback message if LLM is not available
        return f"Hi {contact_person}, Mumbai Retail Chain needs {quantity} units of {product_name}. Please quote your best price and delivery time. We're looking for reliable suppliers for ongoing business. Thanks!"
    
    prompt = f"""
    Generate a concise business inquiry message for a supplier.
    
    DETAILS:
    - Your Company: Mumbai Retail Chain (8 stores)
    - Supplier: {company_name}
    - Contact: {contact_person}
    - Product: {product_name}
    - Quantity: {quantity} units
    - Location: Mumbai, India
    
    Write a brief, professional message (2-3 sentences max) that:
    1. States your requirement clearly
    2. Asks for pricing and delivery time
    3. Mentions potential for ongoing business
    
    Keep it under 80 words, suitable for business communication.
    """
    
    try:
        response = llm.invoke(prompt)
        message = response.content.strip()
        # Clean up any formatting issues
        message = message.replace('[Product Name]', product_name)
        message = message.replace('[Your Company Name]', 'Mumbai Retail Chain')
        message = message.replace('[Supplier Company Name]', company_name)
        message = message.replace('[Supplier Contact Person]', contact_person)
        message = message.replace('[Your Name]', 'Procurement Manager')
        return message
    except Exception as e:
        logger.error(f"Error generating message with LLM: {e}")
        # Fallback message if LLM fails
        return f"Hi {contact_person}, Mumbai Retail Chain needs {quantity} units of {product_name}. Please quote your best price and delivery time. We're looking for reliable suppliers for ongoing business. Thanks!"

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="Multi-Seller Supply Chain Management API",
    description="API for multi-seller portal with negotiation and selection capabilities",
    version="2.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files if you want to serve HTML directly
# app.mount("/static", StaticFiles(directory="static"), name="static")

# =============================================================================
# ROOT AND HEALTH CHECK ROUTES
# =============================================================================

@app.get("/")
async def root():
    return {
        "message": "Multi-Seller Supply Chain Management API",
        "version": "2.0.0",
        "status": "active",
        "endpoints": {
            "seller_portal": "/portal/seller/{seller_id}",
            "buyer_portal": "/portal/buyer",
            "api_docs": "/docs",
            "sellers": "/api/sellers",
            "procurement": "/api/procurement"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database_status": "connected" if os.path.exists(config.DATA_DIRECTORY + config.SELLERS_CSV) else "initializing"
    }

# =============================================================================
# SELLER MANAGEMENT ROUTES
# =============================================================================

@app.get("/api/sellers")
async def get_all_sellers():
    """Get list of all registered sellers"""
    try:
        sellers_df = load_sellers_database()
        sellers = sellers_df.to_dict('records')
        
        # Group by product
        products = {}
        for seller in sellers:
            prod_id = seller['product_id']
            if prod_id not in products:
                products[prod_id] = {
                    "product_id": prod_id,
                    "product_name": seller['product_name'],
                    "sellers": []
                }
            products[prod_id]["sellers"].append(seller)
        
        return {
            "total_sellers": len(sellers),
            "products": list(products.values()),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error fetching sellers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sellers/{seller_id}")
async def get_seller_details(seller_id: str):
    """Get detailed information about a specific seller"""
    try:
        sellers_df = load_sellers_database()
        seller = sellers_df[sellers_df['seller_id'] == seller_id]
        
        if seller.empty:
            raise HTTPException(status_code=404, detail="Seller not found")
        
        seller_info = seller.iloc[0].to_dict()
        
        # Get active requests for this seller
        active_requests = get_active_requests_for_seller(seller_id)
        
        # Get negotiation history
        negotiation_history = get_seller_negotiation_history(seller_id)
        
        return {
            "seller_info": seller_info,
            "active_requests": active_requests,
            "negotiation_history": negotiation_history,
            "status": "success"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching seller details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sellers/{seller_id}/requests")
async def get_seller_requests(seller_id: str):
    """Get all active requests for a seller"""
    try:
        # Validate seller exists
        sellers_df = load_sellers_database()
        if sellers_df[sellers_df['seller_id'] == seller_id].empty:
            raise HTTPException(status_code=404, detail="Seller not found")
        
        requests = get_active_requests_for_seller(seller_id)
        return {
            "seller_id": seller_id,
            "active_requests": requests,
            "count": len(requests),
            "status": "success"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching seller requests: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# PROCUREMENT PROCESS ROUTES
# =============================================================================

@app.post("/api/procurement/initiate")
async def initiate_procurement_process(request: InitiateProcessRequest):
    """Initiate complete procurement process for a product"""
    try:
        # Generate process ID
        process_id = f"PROC_{request.product_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Get product info
        products_df = load_products_database()
        product = products_df[products_df['product_id'] == request.product_id]
        
        if product.empty:
            raise HTTPException(status_code=404, detail="Product not found")
        
        product_name = product.iloc[0]['product_name']
        
        # Start process
        sellers = get_sellers_for_product(request.product_id)
        
        if not sellers:
            raise HTTPException(status_code=404, detail="No sellers found for this product")
        
        requests_sent = send_initial_request_to_sellers(
            request.product_id, 
            request.quantity_needed, 
            product_name
        )
        
        # Create process tracking record
        process_record = {
            "process_id": process_id,
            "product_id": request.product_id,
            "product_name": product_name,
            "quantity_needed": request.quantity_needed,
            "urgency": request.urgency,
            "status": "active",
            "current_stage": "awaiting_responses",
            "sellers_contacted": len(sellers),
            "started_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "expected_completion": (datetime.now() + timedelta(hours=48)).strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save process record
        save_process_record(process_record)
        
        return {
            "process_id": process_id,
            "message": f"Procurement process initiated for {product_name}",
            "sellers_contacted": len(sellers),
            "expected_completion": process_record["expected_completion"],
            "status": "success"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error initiating procurement: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/procurement/{process_id}/status")
async def get_procurement_status(process_id: str):
    """Get current status of procurement process"""
    try:
        process_record = load_process_record(process_id)
        
        if not process_record:
            raise HTTPException(status_code=404, detail="Process not found")
        
        # Get detailed status based on current stage
        if process_record["current_stage"] == "awaiting_responses":
            responses_received = count_responses_received(process_record["product_id"])
            details = {
                "responses_received": responses_received,
                "responses_pending": process_record["sellers_contacted"] - responses_received
            }
        elif process_record["current_stage"] == "negotiating":
            active_negotiations = get_active_negotiations_count(process_record["product_id"])
            details = {"active_negotiations": active_negotiations}
        else:
            details = {}
        
        return {
            "process_id": process_id,
            "product_id": process_record["product_id"],
            "status": process_record["status"],
            "current_stage": process_record["current_stage"],
            "details": details
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching process status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# SELLER RESPONSE ROUTES
# =============================================================================

@app.post("/api/responses/submit")
async def submit_seller_response(response: SellerResponse):
    """Submit seller response to procurement request"""
    try:
        # Validate seller and request
        sellers_df = load_sellers_database()
        seller = sellers_df[sellers_df['seller_id'] == response.seller_id]
        
        if seller.empty:
            raise HTTPException(status_code=404, detail="Seller not found")
        
        # Validate response data
        if response.response.lower() not in ['yes', 'no']:
            raise HTTPException(status_code=400, detail="Response must be 'yes' or 'no'")
        
        if response.response.lower() == 'yes':
            if not response.quoted_price or not response.expected_delivery_days:
                raise HTTPException(status_code=400, detail="Price and delivery days required for acceptance")
            if response.quoted_price <= 0 or response.expected_delivery_days <= 0:
                raise HTTPException(status_code=400, detail="Price and delivery days must be positive")
        
        # Check if response already exists
        if check_response_exists(response.seller_id, response.product_id):
            raise HTTPException(status_code=400, detail="Response already submitted for this request")
        
        # Get seller info for enriching response
        seller_info = seller.iloc[0]
        
        # Create response record
        response_data = {
            "response_id": str(uuid.uuid4()),
            "seller_id": response.seller_id,
            "company_name": seller_info['company_name'],
            "product_id": response.product_id,
            "request_id": response.request_id,
            "response": response.response.lower(),
            "quoted_price": response.quoted_price,
            "expected_delivery_days": response.expected_delivery_days,
            "additional_notes": response.additional_notes,
            "reliability_score": seller_info['reliability_score'],
            "response_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "status": ResponseStatus.ACCEPTED if response.response.lower() == "yes" else ResponseStatus.DECLINED
        }
        
        # Save response
        save_seller_response(response_data)
        
        # Check if all sellers have responded
        sellers_count = len(get_sellers_for_product(response.product_id))
        responses_count = count_responses_received(response.product_id)
        
        all_responded = responses_count >= sellers_count
        
        # AUTO-START NEGOTIATIONS if all sellers responded
        if all_responded:
            logger.info(f"🤝 All sellers responded for product {response.product_id}. Starting auto-negotiations...")
            try:
                auto_start_negotiations(response.product_id)
            except Exception as e:
                logger.error(f"Error starting auto-negotiations: {e}")
        
        logger.info(f"Response received from {seller_info['company_name']}: {response.response}")
        
        return {
            "response_id": response_data["response_id"],
            "status": response_data["status"],
            "message": "Response submitted successfully",
            "all_sellers_responded": all_responded,
            "responses_received": responses_count,
            "total_sellers": sellers_count,
            "auto_negotiation_started": all_responded
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting response: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/responses/product/{product_id}")
async def get_product_responses(product_id: int):
    """Get all responses for a product"""
    try:
        responses_file = config.DATA_DIRECTORY + f"seller_responses_{product_id}.csv"
        
        if not os.path.exists(responses_file):
            return {
                "product_id": product_id,
                "responses": [],
                "count": 0,
                "accepted_count": 0
            }
        
        responses_df = pd.read_csv(responses_file)
        
        # Handle missing columns
        required_columns = ['seller_id', 'product_id', 'response', 'quoted_price', 'expected_delivery_days', 'company_name']
        for col in required_columns:
            if col not in responses_df.columns:
                responses_df[col] = 'N/A' if col == 'company_name' else None
        
        responses = responses_df.to_dict('records')
        
        # Clean up NaN values
        for response in responses:
            for key, value in response.items():
                if pd.isna(value):
                    response[key] = None
        
        return {
            "product_id": product_id,
            "responses": responses,
            "count": len(responses),
            "accepted_count": len([r for r in responses if r.get('response') == 'yes'])
        }
    except Exception as e:
        logger.error(f"Error fetching product responses: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading responses: {str(e)}")

# =============================================================================
# ANALYTICS ROUTES
# =============================================================================

@app.get("/api/analytics/overview")
async def get_analytics_overview():
    """Get system-wide analytics"""
    try:
        sellers_df = load_sellers_database()
        
        # Count active processes
        active_processes = 0
        completed_processes = 0
        
        if os.path.exists(config.DATA_DIRECTORY):
            for file in os.listdir(config.DATA_DIRECTORY):
                if file.startswith('process_') and file.endswith('.json'):
                    try:
                        with open(os.path.join(config.DATA_DIRECTORY, file), 'r') as f:
                            process = json.load(f)
                            if process.get('status') == 'active':
                                active_processes += 1
                            else:
                                completed_processes += 1
                    except:
                        continue
        
        # Get negotiations count
        total_negotiations = 0
        if os.path.exists(config.DATA_DIRECTORY):
            total_negotiations = len([f for f in os.listdir(config.DATA_DIRECTORY) if f.startswith('negotiation_')])
        
        # Calculate average metrics
        analytics = {
            "total_sellers": len(sellers_df),
            "active_sellers": len(sellers_df[sellers_df['status'] == 'active']),
            "total_products": len(sellers_df['product_id'].unique()),
            "active_processes": active_processes,
            "completed_processes": completed_processes,
            "total_negotiations": total_negotiations,
            "sellers_by_product": sellers_df.groupby('product_name').size().to_dict(),
            "average_reliability": round(sellers_df['reliability_score'].mean(), 1),
            "system_status": "operational",
            "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return analytics
    except Exception as e:
        logger.error(f"Error fetching analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/seller/{seller_id}")
async def get_seller_analytics(seller_id: str):
    """Get analytics for specific seller"""
    try:
        sellers_df = load_sellers_database()
        seller = sellers_df[sellers_df['seller_id'] == seller_id]
        
        if seller.empty:
            raise HTTPException(status_code=404, detail="Seller not found")
        
        seller_info = seller.iloc[0].to_dict()
        
        # Count responses and negotiations
        total_requests = 0
        accepted_requests = 0
        total_negotiations = 0
        successful_negotiations = 0
        
        # Analyze response files
        if os.path.exists(config.DATA_DIRECTORY):
            for file in os.listdir(config.DATA_DIRECTORY):
                if file.startswith('seller_responses_'):
                    try:
                        df = pd.read_csv(os.path.join(config.DATA_DIRECTORY, file))
                        seller_responses = df[df['seller_id'] == seller_id]
                        total_requests += len(seller_responses)
                        accepted_requests += len(seller_responses[seller_responses['response'] == 'yes'])
                    except:
                        continue
        
        # Analyze negotiation files
        if os.path.exists(config.DATA_DIRECTORY):
            for file in os.listdir(config.DATA_DIRECTORY):
                if file.startswith('negotiation_') and file.endswith('.json'):
                    try:
                        with open(os.path.join(config.DATA_DIRECTORY, file), 'r') as f:
                            neg = json.load(f)
                            if neg.get('seller_id') == seller_id:
                                total_negotiations += 1
                                if neg.get('status') == 'accepted':
                                    successful_negotiations += 1
                    except:
                        continue
        
        analytics = {
            "seller_info": seller_info,
            "performance_metrics": {
                "total_requests_received": total_requests,
                "requests_accepted": accepted_requests,
                "acceptance_rate": round((accepted_requests / total_requests * 100) if total_requests > 0 else 0, 1),
                "total_negotiations": total_negotiations,
                "successful_negotiations": successful_negotiations,
                "negotiation_success_rate": round((successful_negotiations / total_negotiations * 100) if total_negotiations > 0 else 0, 1)
            },
            "last_activity": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return analytics
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching seller analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# NEGOTIATION ROUTES
# =============================================================================

@app.get("/api/negotiations/active")
async def get_active_negotiations():
    """Get all active negotiations"""
    try:
        negotiations = []
        
        if not os.path.exists(config.DATA_DIRECTORY):
            return {"active_negotiations": [], "count": 0}
        
        # Find all negotiation files
        for file in os.listdir(config.DATA_DIRECTORY):
            if file.startswith('negotiation_') and file.endswith('.json'):
                try:
                    with open(os.path.join(config.DATA_DIRECTORY, file), 'r') as f:
                        neg = json.load(f)
                        if neg.get('status') == 'active':
                            negotiations.append({
                                "negotiation_id": neg['negotiation_id'],
                                "seller_id": neg['seller_id'],
                                "company_name": neg['company_name'],
                                "product_id": neg['product_id'],
                                "current_round": neg['current_round'],
                                "last_message": neg['messages'][-1] if neg.get('messages') else None
                            })
                except:
                    continue
        
        return {
            "active_negotiations": negotiations,
            "count": len(negotiations)
        }
    except Exception as e:
        logger.error(f"Error fetching active negotiations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/negotiations/{negotiation_id}")
async def get_negotiation_details(negotiation_id: str):
    """Get detailed negotiation history"""
    try:
        log_file = config.DATA_DIRECTORY + f"negotiation_{negotiation_id}.json"
        
        if not os.path.exists(log_file):
            raise HTTPException(status_code=404, detail="Negotiation not found")
        
        with open(log_file, 'r') as f:
            negotiation = json.load(f)
        
        return negotiation
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching negotiation details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/negotiations/{negotiation_id}/respond")
async def respond_to_negotiation(negotiation_id: str, response: NegotiationMessage):
    """Submit seller response in negotiation"""
    try:
        # Load negotiation
        log_file = config.DATA_DIRECTORY + f"negotiation_{negotiation_id}.json"
        
        if not os.path.exists(log_file):
            raise HTTPException(status_code=404, detail="Negotiation not found")
        
        with open(log_file, 'r') as f:
            negotiation_log = json.load(f)
        
        # Validate seller
        if response.seller_id != negotiation_log.get('seller_id'):
            raise HTTPException(status_code=403, detail="Unauthorized seller")
        
        # Validate negotiation is still active
        if negotiation_log.get('status') != 'active':
            raise HTTPException(status_code=400, detail="Negotiation is not active")
        
        # Add seller response to messages
        new_message = {
            "round": negotiation_log['current_round'],
            "from": "seller",
            "message": response.message,
            "proposed_price": response.counter_offer_price,
            "proposed_delivery": response.counter_offer_delivery,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        negotiation_log['messages'].append(new_message)
        
        # Process the negotiation round
        seller_response_data = {
            "message": response.message,
            "counter_offer_price": response.counter_offer_price,
            "counter_offer_delivery": response.counter_offer_delivery
        }
        
        decision = process_negotiation_round(negotiation_log, seller_response_data)
        
        return {
            "negotiation_id": negotiation_id,
            "decision": decision.get('action', 'continue'),
            "current_round": negotiation_log['current_round'],
            "status": negotiation_log['status'],
            "buyer_response": decision.get('message', ''),
            "message": "Negotiation response submitted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting negotiation response: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# HELPER FUNCTIONS FOR API
# =============================================================================

def get_active_requests_for_seller(seller_id: str):
    """Get all active requests for a seller"""
    active_requests = []
    
    if not os.path.exists(config.DATA_DIRECTORY):
        return active_requests
    
    # Check request files
    for file in os.listdir(config.DATA_DIRECTORY):
        if file.startswith('seller_requests_') and file.endswith('.csv'):
            try:
                df = pd.read_csv(os.path.join(config.DATA_DIRECTORY, file))
                seller_requests = df[df['seller_id'] == seller_id]
                
                for _, request in seller_requests.iterrows():
                    # Check if response exists
                    response_exists = check_response_exists(seller_id, request['product_id'])
                    
                    if not response_exists:
                        active_requests.append({
                            "request_id": request['request_id'],
                            "product_id": request['product_id'],
                            "product_name": request['product_name'],
                            "quantity_requested": request['quantity_requested'],
                            "request_date": request['request_date'],
                            "deadline": request['response_deadline']
                        })
            except:
                continue
    
    return active_requests

def get_seller_negotiation_history(seller_id: str):
    """Get negotiation history for a seller"""
    negotiations = []
    
    if not os.path.exists(config.DATA_DIRECTORY):
        return negotiations
    
    for file in os.listdir(config.DATA_DIRECTORY):
        if file.startswith('negotiation_') and file.endswith('.json'):
            try:
                with open(os.path.join(config.DATA_DIRECTORY, file), 'r') as f:
                    neg = json.load(f)
                    if neg.get('seller_id') == seller_id:
                        negotiations.append({
                            "negotiation_id": neg['negotiation_id'],
                            "product_id": neg['product_id'],
                            "status": neg['status'],
                            "rounds": neg['current_round'],
                            "final_price": neg.get('final_price'),
                            "created_at": neg['created_at']
                        })
            except:
                continue
    
    return negotiations

def save_process_record(process_record: dict):
    """Save procurement process record"""
    try:
        filename = config.DATA_DIRECTORY + f"process_{process_record['process_id']}.json"
        with open(filename, 'w') as f:
            json.dump(process_record, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving process record: {e}")

def load_process_record(process_id: str):
    """Load procurement process record"""
    try:
        filename = config.DATA_DIRECTORY + f"process_{process_id}.json"
        
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                return json.load(f)
        return None
    except Exception as e:
        logger.error(f"Error loading process record: {e}")
        return None

def count_responses_received(product_id: int):
    """Count responses received for a product"""
    try:
        responses_file = config.DATA_DIRECTORY + f"seller_responses_{product_id}.csv"
        
        if os.path.exists(responses_file):
            df = pd.read_csv(responses_file)
            return len(df)
        return 0
    except Exception as e:
        logger.error(f"Error counting responses: {e}")
        return 0

def get_active_negotiations_count(product_id: int):
    """Count active negotiations for a product"""
    count = 0
    
    if not os.path.exists(config.DATA_DIRECTORY):
        return count
    
    for file in os.listdir(config.DATA_DIRECTORY):
        if file.startswith('negotiation_') and file.endswith('.json'):
            try:
                with open(os.path.join(config.DATA_DIRECTORY, file), 'r') as f:
                    neg = json.load(f)
                    if neg.get('product_id') == product_id and neg.get('status') == 'active':
                        count += 1
            except:
                continue
    
    return count

def check_response_exists(seller_id: str, product_id: int):
    """Check if seller has already responded"""
    try:
        responses_file = config.DATA_DIRECTORY + f"seller_responses_{product_id}.csv"
        
        if os.path.exists(responses_file):
            df = pd.read_csv(responses_file)
            return len(df[df['seller_id'] == seller_id]) > 0
        return False
    except Exception as e:
        logger.error(f"Error checking response exists: {e}")
        return False

def save_seller_response(response_data: dict):
    """Save seller response to file"""
    try:
        responses_file = config.DATA_DIRECTORY + f"seller_responses_{response_data['product_id']}.csv"
        response_df = pd.DataFrame([response_data])
        
        if os.path.exists(responses_file):
            response_df.to_csv(responses_file, mode='a', header=False, index=False)
        else:
            response_df.to_csv(responses_file, index=False)
        
        logger.info(f"Response saved for seller {response_data['seller_id']}")
    except Exception as e:
        logger.error(f"Error saving seller response: {e}")
        raise

def process_negotiation_round(negotiation_log: dict, seller_response: dict):
    """Process seller's response and decide next action using LLM"""
    
    try:
        # Get product info
        products_df = load_products_database()
        product_info = products_df[products_df['product_id'] == negotiation_log['product_id']].iloc[0].to_dict()
        
        # Build negotiation history
        history = "\n".join([
            f"Round {msg['round']} - {msg['from']}: {msg['message']}" 
            for msg in negotiation_log.get('messages', [])
        ])
        
        if llm is None:
            # Fallback decision logic without LLM
            if negotiation_log['current_round'] >= config.MAX_NEGOTIATION_ROUNDS:
                negotiation_log['status'] = 'ended'
                decision = {"action": "end", "reason": "Max rounds reached"}
            else:
                decision = {
                    "action": "counter",
                    "reason": "Continuing negotiation",
                    "counter_price": product_info['cost_price'] * 0.98,
                    "counter_delivery": product_info['lead_time_days'],
                    "message": "We appreciate your flexibility. Can we meet at this price point?"
                }
        else:
            # Use LLM for decision making
            prompt = f"""
            Analyze this negotiation and decide next action:
            
            PRODUCT BASELINE:
            - Current Cost: ₹{product_info['cost_price']}
            - Current Lead Time: {product_info['lead_time_days']} days
            
            NEGOTIATION HISTORY:
            {history}
            
            SELLER'S LATEST RESPONSE:
            Message: {seller_response['message']}
            Counter Price: ₹{seller_response.get('counter_offer_price', 'Not specified')}
            Counter Delivery: {seller_response.get('counter_offer_delivery', 'Not specified')} days
            
            Current Round: {negotiation_log['current_round']} of {config.MAX_NEGOTIATION_ROUNDS}
            
            Decide:
            1. Should we accept this offer?
            2. If not, what counter-offer should we make?
            3. Or should we end negotiations?
            
            Return JSON: {{"action": "<accept/counter/end>", "reason": "<explanation>", "counter_price": <number or null>, "counter_delivery": <number or null>, "message": "<response message>"}}
            """
            
            try:
                response = llm.invoke(prompt)
                content = response.content.strip()
                
                # Extract JSON from response
                if '{' in content:
                    json_str = content[content.find('{'):content.rfind('}')+1]
                    decision = json.loads(json_str)
                else:
                    decision = {"action": "counter", "reason": "Continuing negotiation"}
                    
            except Exception as e:
                logger.error(f"Error in LLM decision: {e}")
                # Fallback decision logic
                if negotiation_log['current_round'] >= config.MAX_NEGOTIATION_ROUNDS:
                    decision = {"action": "end", "reason": "Max rounds reached"}
                else:
                    decision = {
                        "action": "counter",
                        "reason": "Continuing negotiation",
                        "counter_price": product_info['cost_price'] * 0.98,
                        "counter_delivery": product_info['lead_time_days'],
                        "message": "We appreciate your flexibility. Can we meet at this price point?"
                    }
        
        # Process the decision
        if decision['action'] == 'accept':
            negotiation_log['status'] = 'accepted'
            negotiation_log['final_price'] = seller_response.get('counter_offer_price')
            negotiation_log['final_delivery'] = seller_response.get('counter_offer_delivery')
            
            # Add acceptance message
            negotiation_log['messages'].append({
                "round": negotiation_log['current_round'] + 1,
                "from": "buyer",
                "message": decision.get('message', "We accept your offer. Looking forward to a successful partnership."),
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "final_deal": True
            })
            
        elif decision['action'] == 'counter' and negotiation_log['current_round'] < config.MAX_NEGOTIATION_ROUNDS:
            negotiation_log['current_round'] += 1
            
            # Add counter message
            negotiation_log['messages'].append({
                "round": negotiation_log['current_round'],
                "from": "buyer",
                "message": decision.get('message', "We appreciate your flexibility. Can we meet at a better price point?"),
                "proposed_price": decision.get('counter_price'),
                "proposed_delivery": decision.get('counter_delivery'),
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
        else:
            negotiation_log['status'] = 'ended'
            negotiation_log['messages'].append({
                "round": negotiation_log['current_round'] + 1,
                "from": "buyer",
                "message": decision.get('message', 'Thank you for your time. We will explore other options.'),
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "negotiation_ended": True
            })
        
        save_negotiation_log(negotiation_log)
        return decision
        
    except Exception as e:
        logger.error(f"Error processing negotiation round: {e}")
        # Fallback decision
        return {"action": "continue", "reason": "Processing error", "message": "We'll review and get back to you."}

def save_negotiation_log(negotiation_log: dict):
    """Save negotiation log to file"""
    try:
        log_file = config.DATA_DIRECTORY + f"negotiation_{negotiation_log['negotiation_id']}.json"
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        # Convert the entire log
        clean_log = convert_numpy_types(negotiation_log)
        
        with open(log_file, 'w') as f:
            json.dump(clean_log, f, indent=2)
            
    except Exception as e:
        logger.error(f"Error saving negotiation log: {e}")

def auto_start_negotiations(product_id: int):
    """Automatically start negotiations when all sellers respond"""
    try:
        logger.info(f"🤖 Starting auto-negotiations for product {product_id}")
        
        # Load all responses
        responses_file = config.DATA_DIRECTORY + f"seller_responses_{product_id}.csv"
        if not os.path.exists(responses_file):
            logger.error(f"No responses file found for product {product_id}")
            return
            
        responses_df = pd.read_csv(responses_file)
        accepted_responses = responses_df[responses_df['response'] == 'yes']
        
        if len(accepted_responses) == 0:
            logger.info("No accepted responses to negotiate with")
            return
            
        # Get product info
        products_df = load_products_database()
        product_info = products_df[products_df['product_id'] == product_id].iloc[0].to_dict()
        
        # Select top 2-3 sellers for negotiation based on price and delivery
        top_sellers = select_top_sellers_for_negotiation(accepted_responses, product_info, top_n=min(3, len(accepted_responses)))
        
        logger.info(f"Selected {len(top_sellers)} sellers for negotiation")
        
        # Start negotiations with selected sellers
        for seller in top_sellers:
            negotiation_id = str(uuid.uuid4())
            
            # Generate negotiation message
            initial_message = generate_negotiation_message_auto(seller, product_info)
            
            negotiation_log = {
                "negotiation_id": negotiation_id,
                "seller_id": seller['seller_id'],
                "company_name": seller.get('company_name', 'Unknown Company'),
                "product_id": product_id,
                "initial_quoted_price": seller['quoted_price'],
                "initial_delivery_days": seller['expected_delivery_days'],
                "current_round": 1,
                "status": "active",
                "created_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "messages": [
                    {
                        "round": 1,
                        "from": "buyer",
                        "message": initial_message['message'],
                        "proposed_price": initial_message.get('proposed_price'),
                        "proposed_delivery": initial_message.get('proposed_delivery'),
                        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                ]
            }
            
            save_negotiation_log(negotiation_log)
            logger.info(f"💬 Started negotiation with {seller.get('company_name', seller['seller_id'])}")
            
        logger.info(f"✅ Auto-negotiations started for {len(top_sellers)} sellers")
        
    except Exception as e:
        logger.error(f"Error in auto_start_negotiations: {e}")

def select_top_sellers_for_negotiation(accepted_responses: pd.DataFrame, product_info: dict, top_n: int = 3):
    """Select top N sellers for negotiation"""
    try:
        # Calculate scores
        accepted_responses = accepted_responses.copy()
        
        # Price score (lower price = higher score)
        min_price = accepted_responses['quoted_price'].min()
        max_price = accepted_responses['quoted_price'].max()
        if max_price > min_price:
            accepted_responses['price_score'] = 100 * (max_price - accepted_responses['quoted_price']) / (max_price - min_price)
        else:
            accepted_responses['price_score'] = 100
            
        # Delivery score (faster delivery = higher score)
        min_delivery = accepted_responses['expected_delivery_days'].min()
        max_delivery = accepted_responses['expected_delivery_days'].max()
        if max_delivery > min_delivery:
            accepted_responses['delivery_score'] = 100 * (max_delivery - accepted_responses['expected_delivery_days']) / (max_delivery - min_delivery)
        else:
            accepted_responses['delivery_score'] = 100
            
        # Reliability score
        if 'reliability_score' in accepted_responses.columns:
            accepted_responses['reliability_score_weighted'] = accepted_responses['reliability_score']
        else:
            accepted_responses['reliability_score_weighted'] = 80  # Default
        
        # Total score
        accepted_responses['total_score'] = (
            accepted_responses['price_score'] * 0.4 +
            accepted_responses['delivery_score'] * 0.3 +
            accepted_responses['reliability_score_weighted'] * 0.3
        )
        
        # Sort by total score and get top N
        top_sellers = accepted_responses.nlargest(min(top_n, len(accepted_responses)), 'total_score')
        
        return top_sellers.to_dict('records')
        
    except Exception as e:
        logger.error(f"Error selecting top sellers: {e}")
        return accepted_responses.to_dict('records')[:top_n]

def generate_negotiation_message_auto(seller: dict, product_info: dict):
    """Generate initial negotiation message automatically"""
    try:
        # Calculate target values
        current_price = seller['quoted_price']
        target_price = min(current_price * 0.9, product_info['cost_price'] * 0.95)
        target_delivery = max(seller['expected_delivery_days'] - 1, product_info.get('lead_time_days', 5))
        
        company_name = seller.get('company_name', 'Supplier')
        
        if llm is None:
            # Fallback message
            message = f"Thank you {company_name} for your offer of ₹{current_price}. For a long-term partnership with regular orders, could you consider ₹{target_price:.2f} with {target_delivery}-day delivery? We value reliability and are looking for a committed supplier."
        else:
            prompt = f"""
            Generate a professional negotiation opening message:
            
            SELLER: {company_name}
            Their Offer: ₹{current_price}, {seller['expected_delivery_days']} days
            
            OUR TARGETS:
            - Price: ₹{target_price:.2f}
            - Delivery: {target_delivery} days
            
            Write a brief, professional message that:
            1. Thanks them for their offer
            2. Proposes better terms
            3. Mentions volume and long-term partnership
            4. Is respectful but business-focused
            
            Keep under 100 words.
            """
            
            try:
                response = llm.invoke(prompt)
                message = response.content.strip()
            except Exception as e:
                logger.error(f"LLM error in negotiation message: {e}")
                message = f"Thank you {company_name} for your offer of ₹{current_price}. For a long-term partnership with regular orders, could you consider ₹{target_price:.2f} with {target_delivery}-day delivery? We value reliability and are looking for a committed supplier."
        
        return {
            "message": message,
            "proposed_price": round(target_price, 2),
            "proposed_delivery": target_delivery
        }
        
    except Exception as e:
        logger.error(f"Error generating negotiation message: {e}")
        return {
            "message": f"Thank you for your offer. We'd like to discuss pricing and delivery terms for potential long-term partnership.",
            "proposed_price": None,
            "proposed_delivery": None
        }

# =============================================================================
# INITIALIZATION AND MAIN
# =============================================================================

def initialize_system():
    """Initialize the system with required data"""
    try:
        # Create data directory
        os.makedirs(config.DATA_DIRECTORY, exist_ok=True)
        
        # Initialize sellers database if not exists
        if not os.path.exists(config.DATA_DIRECTORY + config.SELLERS_CSV):
            logger.info("Initializing sellers database...")
            generate_sellers_database()
        
        # Initialize products database if not exists
        if not os.path.exists(config.DATA_DIRECTORY + config.PRODUCTS_CSV):
            logger.info("Initializing products database...")
            generate_sample_products()
        
        logger.info("✅ System initialization complete")
        
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        raise

# =============================================================================
# STARTUP EVENT
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    logger.info("🚀 Starting Multi-Seller Supply Chain Management API")
    initialize_system()

if __name__ == "__main__":
    logger.info("🌟 MULTI-SELLER SUPPLY CHAIN MANAGEMENT SYSTEM")
    logger.info("=" * 50)
    
    # Initialize system
    initialize_system()
    
    logger.info("\n🚀 Starting Multi-Seller Portal API...")
    logger.info("\n📱 Access Points:")
    logger.info("- Main Portal: http://localhost:8000")
    logger.info("- API Documentation: http://localhost:8000/docs")
    logger.info("- Health Check: http://localhost:8000/health")
    logger.info("\n🛠️ Key API Endpoints:")
    logger.info("\nSELLER ENDPOINTS:")
    logger.info("- GET  /api/sellers - List all sellers")
    logger.info("- GET  /api/sellers/{seller_id} - Seller details")
    logger.info("- GET  /api/sellers/{seller_id}/requests - Active requests")
    logger.info("- POST /api/responses/submit - Submit response")
    logger.info("- POST /api/negotiations/{id}/respond - Negotiation response")
    logger.info("\nBUYER ENDPOINTS:")
    logger.info("- POST /api/procurement/initiate - Start procurement")
    logger.info("- GET  /api/procurement/{id}/status - Process status")
    logger.info("- GET  /api/responses/product/{id} - View responses")
    logger.info("- GET  /api/negotiations/active - Active negotiations")
    logger.info("\nANALYTICS:")
    logger.info("- GET  /api/analytics/overview - System analytics")
    logger.info("- GET  /api/analytics/seller/{id} - Seller analytics")
    logger.info("\n💡 Usage:")
    logger.info("1. Buyer initiates procurement: POST /api/procurement/initiate")
    logger.info("2. Sellers view requests: GET /api/sellers/{seller_id}/requests")
    logger.info("3. Sellers submit responses: POST /api/responses/submit")
    logger.info("4. Monitor progress: GET /api/procurement/{process_id}/status")
    logger.info("\n🌐 Starting server on http://localhost:8000")
    logger.info("   Press Ctrl+C to stop the server")
    logger.info("=" * 50)
    
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")