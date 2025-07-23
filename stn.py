import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from langchain_google_genai import ChatGoogleGenerativeAI

# =============================================================================
# GLOBAL CONFIGURATION - CENTRALIZED SETTINGS
# =============================================================================

class GlobalConfig:
    """Global configuration for supply chain optimization"""

    # API Configuration
    GOOGLE_API_KEY = ""  # Replace with your API key
    LLM_MODEL = "gemini-2.0-flash"
    LLM_TEMPERATURE = 0.01

    # Business Configuration
    CURRENT_DATE = datetime(2025, 7, 15)  # Mid-July for better context
    CURRENT_MONTH = 7
    LOCATION = "Mumbai, India"
    NUMBER_OF_STORES = 8
    TOTAL_INVENTORY_CAPACITY = 1000  # Increased capacity

    # Data Configuration
    DATA_DIRECTORY = "supply_chain_data/"
    PRODUCTS_CSV = "products_inventory.csv"
    DAILY_SALES_CSV = "daily_sales_history.csv"
    MONTHLY_SALES_CSV = "monthly_sales_history.csv"

    # Analysis Configuration
    VELOCITY_ANALYSIS_DAYS = [5, 10, 30]
    SEASONALITY_YEARS = 3
    SAFETY_STOCK_FACTOR = 1.5

# Initialize global configuration
config = GlobalConfig()

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model=config.LLM_MODEL,
    google_api_key=config.GOOGLE_API_KEY,
    temperature=config.LLM_TEMPERATURE,
    convert_system_message_to_human=True
)

# =============================================================================
# REALISTIC DATA GENERATION WITH BALANCED INVENTORY
# =============================================================================

def generate_realistic_supply_chain_data():
    """Generate realistic supply chain data and save as CSV files"""

    # Create data directory if it doesn't exist
    os.makedirs(config.DATA_DIRECTORY, exist_ok=True)

    # Set seed for reproducible results
    np.random.seed(42)

    print("🏭 Generating realistic supply chain data...")

    # =============================================================================
    # 1. PRODUCT INVENTORY DATA - More realistic quantities
    # =============================================================================

    products_data = {
        'product_id': [5001, 5002, 5003, 5004, 5005, 5006],
        'product_name': [
            'Mangoes (per kg)',           # Category 1: Seasonal High + Low Inventory
            'Air Conditioners (1.5 Ton)', # Category 1: Summer peak + adequate stock
            'Winter Jackets',             # Category 2: Seasonal Ending + Adequate Inventory
            'Sweaters',                   # Category 3: Seasonal Ending + Excess Inventory
            'Bread (per loaf)',           # Category 4: Non-Seasonal High Demand
            'Cooking Oil (1L)'            # Category 4: Non-Seasonal Consistent Demand
        ],
        'quantity_in_inventory': [800, 150, 200, 500, 400, 600],  # More realistic quantities
        'current_quantity_in_store': [120, 25, 40, 180, 80, 100], # Better distributed
        'cost_price': [60, 25000, 1200, 600, 20, 120],
        'selling_price': [95, 35000, 1800, 900, 35, 150],
        'shelf_life_days': [7, 1825, 1095, 1095, 3, 730],
        'lead_time_days': [3, 14, 30, 21, 7, 7],  # More realistic lead times
        'category': [
            'Seasonal_High_LowInventory',
            'Seasonal_High_AdequateInventory',
            'Seasonal_Ending_AdequateInventory',
            'Seasonal_Ending_ExcessInventory',
            'NonSeasonal_HighDemand',
            'NonSeasonal_RegularDemand'
        ]
    }

    products_df = pd.DataFrame(products_data)
    products_df.to_csv(config.DATA_DIRECTORY + config.PRODUCTS_CSV, index=False)
    print(f"✅ Created {config.PRODUCTS_CSV}")

    # =============================================================================
    # 2. DAILY SALES DATA - Last 45 days for better analysis
    # =============================================================================

    daily_sales_data = []
    start_date = config.CURRENT_DATE - timedelta(days=44)  # 45 days of data

    for day in range(45):
        date = start_date + timedelta(days=day)
        day_of_week = date.weekday()  # 0=Monday, 6=Sunday
        is_weekend = day_of_week >= 5
        month = date.month

        # Mangoes - peak summer season with realistic sales
        mango_base = 45
        if month in [5, 6, 7]:  # Peak season
            seasonal_mult = 2.2 + (month - 5) * 0.3  # Increasing through season
        elif month in [4, 8]:  # Shoulder season
            seasonal_mult = 1.5
        else:
            seasonal_mult = 0.2
        weekend_mult = 1.4 if is_weekend else 1.0
        random_var = np.random.uniform(0.8, 1.2)
        mango_sales = int(mango_base * seasonal_mult * weekend_mult * random_var)

        # Air Conditioners - summer peak, high-value low-volume
        ac_base = 3
        if month in [4, 5, 6, 7, 8]:  # Summer season
            ac_seasonal = 2.5 + (month - 4) * 0.2
        else:
            ac_seasonal = 0.3
        ac_weekend = 1.6 if is_weekend else 1.0  # People shop for ACs on weekends
        ac_random = np.random.uniform(0.6, 1.4)
        ac_sales = int(ac_base * ac_seasonal * ac_weekend * ac_random)

        # Winter Jackets - off season, very low sales
        jacket_sales = 0 if np.random.random() < 0.7 else int(np.random.uniform(0, 2))

        # Sweaters - off season, minimal sales
        sweater_sales = 0 if np.random.random() < 0.8 else int(np.random.uniform(0, 3))

        # Bread - consistent daily essential with weekend spike
        bread_base = 85
        bread_weekend = 1.3 if is_weekend else 1.0
        bread_random = np.random.uniform(0.9, 1.1)
        bread_sales = max(60, int(bread_base * bread_weekend * bread_random))

        # Cooking Oil - stable demand
        oil_base = 25
        oil_weekend = 1.1 if is_weekend else 1.0
        oil_random = np.random.uniform(0.8, 1.2)
        oil_sales = int(oil_base * oil_weekend * oil_random)

        daily_sales_data.extend([
            {'date': date.strftime('%Y-%m-%d'), 'product_id': 5001, 'product_name': 'Mangoes (per kg)', 'daily_sales': mango_sales},
            {'date': date.strftime('%Y-%m-%d'), 'product_id': 5002, 'product_name': 'Air Conditioners (1.5 Ton)', 'daily_sales': ac_sales},
            {'date': date.strftime('%Y-%m-%d'), 'product_id': 5003, 'product_name': 'Winter Jackets', 'daily_sales': jacket_sales},
            {'date': date.strftime('%Y-%m-%d'), 'product_id': 5004, 'product_name': 'Sweaters', 'daily_sales': sweater_sales},
            {'date': date.strftime('%Y-%m-%d'), 'product_id': 5005, 'product_name': 'Bread (per loaf)', 'daily_sales': bread_sales},
            {'date': date.strftime('%Y-%m-%d'), 'product_id': 5006, 'product_name': 'Cooking Oil (1L)', 'daily_sales': oil_sales}
        ])

    daily_sales_df = pd.DataFrame(daily_sales_data)
    daily_sales_df.to_csv(config.DATA_DIRECTORY + config.DAILY_SALES_CSV, index=False)
    print(f"✅ Created {config.DAILY_SALES_CSV}")

    # =============================================================================
    # 3. MONTHLY SALES DATA - 3 years with clear seasonal patterns
    # =============================================================================

    monthly_sales_data = []

    for year in [2022, 2023, 2024]:
        for month in range(1, 13):

            # Mangoes - clear seasonal pattern
            if month in [5, 6, 7]:  # Peak season
                mango_mult = np.random.uniform(3.5, 4.2) if month == 6 else np.random.uniform(2.8, 3.5)
            elif month in [4, 8]:  # Shoulder season
                mango_mult = np.random.uniform(1.8, 2.2)
            elif month in [3, 9]:  # Transition
                mango_mult = np.random.uniform(0.6, 1.0)
            else:  # Off season
                mango_mult = np.random.uniform(0.1, 0.3)
            mango_year_factor = 1.0 + (year - 2022) * 0.12  # Growth trend
            mango_monthly = int(45 * 30 * mango_mult * mango_year_factor)

            # Air Conditioners - summer seasonal with growth
            if month in [4, 5, 6, 7, 8]:  # Summer season
                ac_mult = np.random.uniform(3.0, 4.0) if month in [5, 6] else np.random.uniform(2.2, 2.8)
            elif month in [3, 9]:  # Shoulder
                ac_mult = np.random.uniform(1.2, 1.6)
            else:  # Off season
                ac_mult = np.random.uniform(0.2, 0.5)
            ac_year_factor = 1.0 + (year - 2022) * 0.18  # Strong growth in AC market
            ac_monthly = int(3 * 30 * ac_mult * ac_year_factor)

            # Winter Jackets - winter seasonal
            if month in [11, 12, 1, 2]:  # Winter season
                jacket_mult = np.random.uniform(3.2, 4.0) if month in [12, 1] else np.random.uniform(2.5, 3.2)
            elif month in [10, 3]:  # Shoulder
                jacket_mult = np.random.uniform(1.5, 2.0)
            else:  # Off season
                jacket_mult = np.random.uniform(0.1, 0.4)
            jacket_year_factor = 1.0 - (year - 2022) * 0.05  # Slight decline
            jacket_monthly = int(2 * 30 * jacket_mult * jacket_year_factor)

            # Sweaters - similar to jackets but declining more
            if month in [11, 12, 1, 2]:  # Winter season
                sweater_mult = np.random.uniform(2.8, 3.5) if month in [12, 1] else np.random.uniform(2.2, 2.8)
            elif month in [10, 3]:  # Shoulder
                sweater_mult = np.random.uniform(1.2, 1.8)
            else:  # Off season
                sweater_mult = np.random.uniform(0.05, 0.3)
            sweater_year_factor = 1.0 - (year - 2022) * 0.08  # Declining trend
            sweater_monthly = int(2 * 30 * sweater_mult * sweater_year_factor)

            # Bread - stable non-seasonal
            bread_mult = np.random.uniform(0.95, 1.05)
            if month in [10, 11, 12]:  # Festival season
                bread_mult *= 1.15
            bread_year_factor = 1.0 + (year - 2022) * 0.04  # Steady growth
            bread_monthly = int(85 * 30 * bread_mult * bread_year_factor)

            # Cooking Oil - stable with slight seasonal variation
            oil_mult = np.random.uniform(0.9, 1.1)
            if month in [10, 11, 12]:  # Festival cooking
                oil_mult *= 1.2
            oil_year_factor = 1.0 + (year - 2022) * 0.06  # Steady growth
            oil_monthly = int(25 * 30 * oil_mult * oil_year_factor)

            monthly_sales_data.extend([
                {'year': year, 'month': month, 'product_id': 5001, 'product_name': 'Mangoes (per kg)', 'monthly_sales': mango_monthly},
                {'year': year, 'month': month, 'product_id': 5002, 'product_name': 'Air Conditioners (1.5 Ton)', 'monthly_sales': ac_monthly},
                {'year': year, 'month': month, 'product_id': 5003, 'product_name': 'Winter Jackets', 'monthly_sales': jacket_monthly},
                {'year': year, 'month': month, 'product_id': 5004, 'product_name': 'Sweaters', 'monthly_sales': sweater_monthly},
                {'year': year, 'month': month, 'product_id': 5005, 'product_name': 'Bread (per loaf)', 'monthly_sales': bread_monthly},
                {'year': year, 'month': month, 'product_id': 5006, 'product_name': 'Cooking Oil (1L)', 'monthly_sales': oil_monthly}
            ])

    monthly_sales_df = pd.DataFrame(monthly_sales_data)
    monthly_sales_df.to_csv(config.DATA_DIRECTORY + config.MONTHLY_SALES_CSV, index=False)
    print(f"✅ Created {config.MONTHLY_SALES_CSV}")

    return products_df, daily_sales_df, monthly_sales_df

# =============================================================================
# DATA LOADING FUNCTIONS - READY FOR DATABASE MIGRATION
# =============================================================================

def load_products_data():
    """Load products inventory data from CSV (future: from database)"""
    try:
        df = pd.read_csv(config.DATA_DIRECTORY + config.PRODUCTS_CSV)
        return df
    except FileNotFoundError:
        print("❌ Products CSV not found. Generating data...")
        generate_realistic_supply_chain_data()
        return pd.read_csv(config.DATA_DIRECTORY + config.PRODUCTS_CSV)

def load_daily_sales_data():
    """Load daily sales data from CSV (future: from database)"""
    try:
        df = pd.read_csv(config.DATA_DIRECTORY + config.DAILY_SALES_CSV)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except FileNotFoundError:
        print("❌ Daily sales CSV not found. Generating data...")
        generate_realistic_supply_chain_data()
        df = pd.read_csv(config.DATA_DIRECTORY + config.DAILY_SALES_CSV)
        df['date'] = pd.to_datetime(df['date'])
        return df

def load_monthly_sales_data():
    """Load monthly sales data from CSV (future: from database)"""
    try:
        df = pd.read_csv(config.DATA_DIRECTORY + config.MONTHLY_SALES_CSV)
        return df
    except FileNotFoundError:
        print("❌ Monthly sales CSV not found. Generating data...")
        generate_realistic_supply_chain_data()
        return pd.read_csv(config.DATA_DIRECTORY + config.MONTHLY_SALES_CSV)

# =============================================================================
# ANALYSIS FUNCTIONS - USING GLOBAL CONFIG
# =============================================================================

def calculate_profit_metrics(product_row):
    """Calculate profit metrics using global configuration"""
    try:
        selling_price = product_row['selling_price']
        cost_price = product_row['cost_price']
        inventory_quantity = product_row['quantity_in_inventory']

        profit_per_unit = selling_price - cost_price
        profit_margin = (profit_per_unit / selling_price) * 100 if selling_price > 0 else 0
        total_inventory_value = inventory_quantity * cost_price

        # Use global config for profitability thresholds
        if profit_margin > 25:
            profitability_status = "Excellent"
        elif profit_margin > 15:
            profitability_status = "High"
        elif profit_margin > 8:
            profitability_status = "Medium"
        else:
            profitability_status = "Low"

        return {
            "profit_per_unit": profit_per_unit,
            "profit_margin_percent": round(profit_margin, 2),
            "total_inventory_value": total_inventory_value,
            "profitability_status": profitability_status
        }
    except Exception as e:
        return {"error": f"Profit calculation failed: {str(e)}"}

def calculate_velocity_metrics(product_id, daily_sales_df):
    """Calculate velocity metrics using global configuration"""
    try:
        # Get product sales data
        product_sales = daily_sales_df[daily_sales_df['product_id'] == product_id].copy()

        if len(product_sales) < max(config.VELOCITY_ANALYSIS_DAYS):
            return {"error": f"Insufficient sales data (have {len(product_sales)} days, need {max(config.VELOCITY_ANALYSIS_DAYS)})"}

        # Sort by date
        product_sales = product_sales.sort_values('date')
        all_sales = product_sales['daily_sales'].values

        # Calculate averages for configured periods
        velocity_metrics = {}
        for period in config.VELOCITY_ANALYSIS_DAYS:
            period_sales = all_sales[-period:] if len(all_sales) >= period else all_sales
            velocity_metrics[f'day_{period}_avg'] = round(float(np.mean(period_sales)), 2)

        # Determine velocity category using global thresholds
        day_5_avg = velocity_metrics['day_5_avg']
        day_30_avg = velocity_metrics['day_30_avg']

        velocity_category = "Regular_Analysis_Needed"

        if day_5_avg > 50 and velocity_metrics['day_10_avg'] > 45 and day_30_avg > 40:
            velocity_category = "Category_1_All_High"
        elif day_5_avg > day_30_avg * 1.5 and day_5_avg > 50:
            velocity_category = "Category_4_Demand_Spike"

        # Calculate suggested order quantity using global config
        lead_time = 3  # Default, will be overridden by product-specific data
        safety_factor = config.TOTAL_INVENTORY_CAPACITY / config.NUMBER_OF_STORES
        safety_buffer_per_store = safety_factor / day_30_avg if day_30_avg > 0 else 1
        demand_during_leadtime = day_5_avg * lead_time
        optimal_order_quantity = int(demand_during_leadtime * safety_buffer_per_store * config.SAFETY_STOCK_FACTOR)

        velocity_metrics.update({
            "velocity_category": velocity_category,
            "demand_spike_ratio": round(day_5_avg / day_30_avg if day_30_avg > 0 else 0, 2),
            "optimal_order_quantity": optimal_order_quantity,
            "total_data_points": len(product_sales)
        })

        return velocity_metrics

    except Exception as e:
        return {"error": f"Velocity calculation failed: {str(e)}"}

def calculate_seasonality_metrics(product_id, monthly_sales_df):
    """Calculate seasonality metrics using global configuration"""
    try:
        # Get product monthly sales data
        product_monthly = monthly_sales_df[monthly_sales_df['product_id'] == product_id].copy()

        if len(product_monthly) < 12:
            return {"error": f"Insufficient monthly data (have {len(product_monthly)} months, need 12)"}

        # Get monthly sales values
        monthly_sales_list = product_monthly['monthly_sales'].tolist()

        # Calculate monthly averages across years using global config
        monthly_avg = {}
        years_of_data = min(len(monthly_sales_list) // 12, config.SEASONALITY_YEARS)

        for month in range(1, 13):
            month_values = []
            for year in range(years_of_data):
                index = year * 12 + (month - 1)
                if index < len(monthly_sales_list):
                    month_values.append(monthly_sales_list[index])
            monthly_avg[month] = sum(month_values) / len(month_values) if month_values else 0

        # Find peak and low months
        max_month = max(monthly_avg, key=monthly_avg.get) if monthly_avg else 1
        min_month = min(monthly_avg, key=monthly_avg.get) if monthly_avg else 1

        # Calculate seasonality strength
        avg_sales = sum(monthly_avg.values()) / 12 if monthly_avg else 0
        max_sales = monthly_avg[max_month]
        min_sales = monthly_avg[min_month]

        seasonality_ratio = max_sales / min_sales if min_sales > 0 else 1

        # Classify seasonality using global thresholds
        if seasonality_ratio > 8:
            classification = "Extremely Seasonal"
            confidence = 95
        elif seasonality_ratio > 4:
            classification = "Highly Seasonal"
            confidence = 90
        elif seasonality_ratio > 2:
            classification = "Moderately Seasonal"
            confidence = 75
        else:
            classification = "Non-Seasonal"
            confidence = 60

        # Determine current phase using global current month
        current_sales = monthly_avg.get(config.CURRENT_MONTH, avg_sales)

        if current_sales > avg_sales * 1.8:
            phase = "Peak"
            multiplier = 2.0
            recommendation = "Order Now - Peak Season"
        elif current_sales > avg_sales * 1.3:
            phase = "High Season"
            multiplier = 1.6
            recommendation = "Order Now - High Demand"
        elif current_sales < avg_sales * 0.4:
            phase = "Off-Season"
            multiplier = 0.4
            recommendation = "Reduce Orders - Off Season"
        elif current_sales < avg_sales * 0.7:
            phase = "Low Season"
            multiplier = 0.7
            recommendation = "Minimal Orders - Low Demand"
        else:
            # Check proximity to peak using global current month
            peak_proximity = min(abs(config.CURRENT_MONTH - max_month),
                               abs(config.CURRENT_MONTH - max_month + 12),
                               abs(config.CURRENT_MONTH - max_month - 12))

            if peak_proximity <= 1:
                if config.CURRENT_MONTH < max_month or (max_month == 1 and config.CURRENT_MONTH == 12):
                    phase = "Rising"
                    multiplier = 1.4
                    recommendation = "Prepare for Peak - Rising Demand"
                else:
                    phase = "Declining"
                    multiplier = 0.8
                    recommendation = "Monitor Closely - Declining Season"
            else:
                phase = "Stable"
                multiplier = 1.0
                recommendation = "Standard Management"

        return {
            "seasonality_classification": classification,
            "current_season_phase": phase,
            "demand_multiplier": round(multiplier, 2),
            "season_confidence": confidence,
            "peak_months": [max_month],
            "low_months": [min_month],
            "recommendation": recommendation,
            "reasoning": f"Peak month {max_month} (avg: {max_sales:.0f}), current month {config.CURRENT_MONTH} (avg: {current_sales:.0f}) shows {phase} pattern",
            "seasonality_ratio": round(seasonality_ratio, 2),
            "total_months_analyzed": len(monthly_sales_list)
        }

    except Exception as e:
        return {"error": f"Seasonality analysis failed: {str(e)}"}

# =============================================================================
# LLM DECISION FUNCTION - ENHANCED WITH GLOBAL CONTEXT
# =============================================================================

def get_llm_quantity_decision(product_data, profit_data, velocity_data, seasonality_data):
    """Get LLM decision for order quantity using global context"""

    prompt = f"""
You are a supply chain optimization expert analyzing inventory for a retail chain in {config.LOCATION}.

BUSINESS CONTEXT:
- Current Date: {config.CURRENT_DATE.strftime('%Y-%m-%d')}
- Location: {config.LOCATION}
- Number of Stores: {config.NUMBER_OF_STORES}
- Total Warehouse Capacity: {config.TOTAL_INVENTORY_CAPACITY} units
- Analysis Period: {max(config.VELOCITY_ANALYSIS_DAYS)} days of sales data

PRODUCT ANALYSIS:
Product: {product_data['product_name']} (ID: {product_data['product_id']})
Category: {product_data.get('category', 'Unknown')}

CURRENT INVENTORY STATUS:
- Warehouse Stock: {product_data['quantity_in_inventory']} units
- Store Display Stock: {product_data['current_quantity_in_store']} units
- Total Available: {product_data['quantity_in_inventory'] + product_data['current_quantity_in_store']} units
- Shelf Life: {product_data['shelf_life_days']} days
- Supplier Lead Time: {product_data['lead_time_days']} days

FINANCIAL METRICS:
- Cost Price: ₹{product_data['cost_price']}
- Selling Price: ₹{product_data['selling_price']}
- Profit per Unit: ₹{profit_data.get('profit_per_unit', 0)}
- Profit Margin: {profit_data.get('profit_margin_percent', 0)}%
- Profitability Status: {profit_data.get('profitability_status', 'Unknown')}
- Current Inventory Value: ₹{profit_data.get('total_inventory_value', 0):,}

DEMAND ANALYSIS:
Recent Sales Velocity:
- Last 5 days average: {velocity_data.get('day_5_avg', 0)} units/day
- Last 10 days average: {velocity_data.get('day_10_avg', 0)} units/day
- Last 30 days average: {velocity_data.get('day_30_avg', 0)} units/day
- Velocity Category: {velocity_data.get('velocity_category', 'Unknown')}
- Demand Trend: {velocity_data.get('demand_spike_ratio', 1.0)}x recent vs baseline

SEASONAL INTELLIGENCE:
- Classification: {seasonality_data.get('seasonality_classification', 'Unknown')}
- Current Phase: {seasonality_data.get('current_season_phase', 'Unknown')}
- Seasonal Multiplier: {seasonality_data.get('demand_multiplier', 1.0)}x
- Confidence Level: {seasonality_data.get('season_confidence', 0)}%
- Peak Months: {seasonality_data.get('peak_months', [])}
- Strategic Recommendation: {seasonality_data.get('recommendation', 'Unknown')}
- Analysis: {seasonality_data.get('reasoning', 'No reasoning available')}

RISK ASSESSMENT:
- Days Until Stockout: {(product_data['quantity_in_inventory'] + product_data['current_quantity_in_store']) / max(velocity_data.get('day_5_avg', 1), 1):.1f} days
- Spoilage Risk: {'HIGH' if product_data['shelf_life_days'] < 7 else 'MEDIUM' if product_data['shelf_life_days'] < 30 else 'LOW'}
- Supply Risk: {'HIGH' if product_data['lead_time_days'] > 14 else 'MEDIUM' if product_data['lead_time_days'] > 7 else 'LOW'}

BUSINESS CONSTRAINTS:
- Distribute across {config.NUMBER_OF_STORES} stores
- Factor in {product_data['lead_time_days']}-day supplier lead time
- Consider {product_data['shelf_life_days']}-day shelf life limit
- Optimize for current {seasonality_data.get('current_season_phase', 'Unknown')} seasonal phase

DECISION FRAMEWORK:
1. Stock Urgency: Will you run out before next delivery?
2. Seasonal Timing: Is now the right time to order based on demand phase?
3. Profitability: Does the profit margin justify the investment?
4. Risk Management: Balance stockout risk vs. spoilage/excess risk
5. Operational Efficiency: Optimize for multi-store distribution

Calculate the optimal order quantity considering:
- Immediate stockout prevention
- Seasonal demand optimization
- Profit margin maximization
- Waste/spoilage minimization
- Working capital efficiency

OUTPUT: Return ONLY a single integer representing the recommended order quantity.
Return 0 if no order is recommended.

If any data shows errors, use your business judgment based on available information.
"""

    try:
        response = llm.invoke(prompt)
        # Extract number from response
        import re
        numbers = re.findall(r'\d+', response.content)
        if numbers:
            return int(numbers[-1])  # Get the last number (likely the final recommendation)
        else:
            return 0
    except Exception as e:
        print(f"   ❌ LLM decision failed: {e}")
        return 0

# =============================================================================
# MAIN WORKFLOW EXECUTION - USING GLOBAL CONFIG
# =============================================================================

def run_global_supply_chain_optimization():
    """Run the global supply chain optimization workflow"""

    print(f"🌍 GLOBAL SUPPLY CHAIN OPTIMIZATION SYSTEM")
    print("=" * 70)
    print(f"📍 Location: {config.LOCATION}")
    print(f"📅 Analysis Date: {config.CURRENT_DATE.strftime('%Y-%m-%d')}")
    print(f"🏪 Store Network: {config.NUMBER_OF_STORES} stores")
    print(f"📦 Warehouse Capacity: {config.TOTAL_INVENTORY_CAPACITY:,} units")
    print()

    # Load data from CSV files (future: database calls)
    print("📊 Loading supply chain data from CSV files...")
    products_df = load_products_data()
    daily_sales_df = load_daily_sales_data()
    monthly_sales_df = load_monthly_sales_data()

    print(f"✅ Loaded {len(products_df)} products")
    print(f"✅ Loaded {len(daily_sales_df)} daily sales records")
    print(f"✅ Loaded {len(monthly_sales_df)} monthly sales records")
    print()

    final_recommendations = {}
    total_order_value = 0
    critical_items = []

    # Process each product using global workflow
    for _, product_row in products_df.iterrows():
        product_id = product_row['product_id']
        product_name = product_row['product_name']

        print(f"📦 Processing Product {product_id}: {product_name}")
        print("-" * 65)

        # Step 1: Financial Analysis
        print("💰 Analyzing profitability...")
        profit_data = calculate_profit_metrics(product_row)
        if 'error' not in profit_data:
            print(f"   💵 Profit Margin: {profit_data['profit_margin_percent']}% ({profit_data['profitability_status']})")
            print(f"   💎 Profit per Unit: ₹{profit_data['profit_per_unit']:,}")
        else:
            print(f"   ❌ Error: {profit_data['error']}")

        # Step 2: Velocity Analysis
        print("⚡ Analyzing demand velocity...")
        velocity_data = calculate_velocity_metrics(product_id, daily_sales_df)
        if 'error' not in velocity_data:
            print(f"   📈 Recent Velocity: {velocity_data['day_5_avg']}/day (5d), {velocity_data['day_30_avg']}/day (30d)")
            print(f"   🎯 Category: {velocity_data['velocity_category']}")
            if velocity_data['demand_spike_ratio'] > 1.2:
                print(f"   📊 Demand Spike: {velocity_data['demand_spike_ratio']}x baseline")
        else:
            print(f"   ❌ Error: {velocity_data['error']}")

        # Step 3: Seasonality Analysis
        print("📅 Analyzing seasonal patterns...")
        seasonality_data = calculate_seasonality_metrics(product_id, monthly_sales_df)
        if 'error' not in seasonality_data:
            print(f"   🌞 Season Phase: {seasonality_data['current_season_phase']} (confidence: {seasonality_data['season_confidence']}%)")
            print(f"   📊 Demand Multiplier: {seasonality_data['demand_multiplier']}x")
            print(f"   💡 Recommendation: {seasonality_data['recommendation']}")
        else:
            print(f"   ❌ Error: {seasonality_data['error']}")

        # Step 4: Stock Risk Assessment
        current_total_stock = product_row['quantity_in_inventory'] + product_row['current_quantity_in_store']
        daily_velocity = velocity_data.get('day_5_avg', 0) if 'error' not in velocity_data else 0
        days_until_stockout = current_total_stock / max(daily_velocity, 0.1)

        print("⚠️  Assessing inventory risk...")
        print(f"   📦 Current Total Stock: {current_total_stock:,} units")
        print(f"   ⏰ Days Until Stockout: {days_until_stockout:.1f} days")
        print(f"   🚚 Supplier Lead Time: {product_row['lead_time_days']} days")

        # Risk classification
        if days_until_stockout <= product_row['lead_time_days']:
            risk_level = "CRITICAL"
            critical_items.append(product_name)
        elif days_until_stockout <= product_row['lead_time_days'] * 1.5:
            risk_level = "HIGH"
        elif days_until_stockout <= product_row['lead_time_days'] * 2:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        print(f"   🚨 Risk Level: {risk_level}")

        # Step 5: LLM Decision Making
        print("🤖 Getting AI recommendation...")
        product_data = {
            'product_id': product_id,
            'product_name': product_name,
            'category': product_row.get('category', 'Unknown'),
            'quantity_in_inventory': product_row['quantity_in_inventory'],
            'current_quantity_in_store': product_row['current_quantity_in_store'],
            'shelf_life_days': product_row['shelf_life_days'],
            'lead_time_days': product_row['lead_time_days'],
            'cost_price': product_row['cost_price'],
            'selling_price': product_row['selling_price']
        }

        recommended_quantity = get_llm_quantity_decision(product_data, profit_data, velocity_data, seasonality_data)
        order_value = recommended_quantity * product_row['cost_price']

        print(f"   🎯 Recommended Order: {recommended_quantity:,} units")
        print(f"   💰 Order Value: ₹{order_value:,}")

        final_recommendations[str(product_id)] = recommended_quantity
        total_order_value += order_value

        print()

    # Summary Report
    print("🏆 OPTIMIZATION SUMMARY")
    print("=" * 50)
    print(f"📊 Products Analyzed: {len(products_df)}")
    print(f"💰 Total Recommended Order Value: ₹{total_order_value:,}")
    print(f"🚨 Critical Items Requiring Immediate Action: {len(critical_items)}")

    if critical_items:
        print(f"   ⚠️  Critical Products: {', '.join(critical_items)}")

    print()
    print("📋 FINAL RECOMMENDATIONS:")
    print("-" * 40)
    for product_id, quantity in final_recommendations.items():
        product_name = products_df[products_df['product_id'] == int(product_id)]['product_name'].iloc[0]
        if quantity > 0:
            cost_price = products_df[products_df['product_id'] == int(product_id)]['cost_price'].iloc[0]
            value = quantity * cost_price
            print(f"Product {product_id} ({product_name}): {quantity:,} units (₹{value:,})")
        else:
            print(f"Product {product_id} ({product_name}): No order recommended")

    # Export recommendations
    recommendations_df = pd.DataFrame([
        {
            'product_id': pid,
            'product_name': products_df[products_df['product_id'] == int(pid)]['product_name'].iloc[0],
            'recommended_quantity': qty,
            'order_value': qty * products_df[products_df['product_id'] == int(pid)]['cost_price'].iloc[0],
            'analysis_date': config.CURRENT_DATE.strftime('%Y-%m-%d')
        }
        for pid, qty in final_recommendations.items()
    ])

    recommendations_file = config.DATA_DIRECTORY + f"recommendations_{config.CURRENT_DATE.strftime('%Y%m%d')}.csv"
    recommendations_df.to_csv(recommendations_file, index=False)
    print(f"\n💾 Recommendations exported to: {recommendations_file}")

    return final_recommendations

# =============================================================================
# SYSTEM HEALTH CHECK & REPORTING
# =============================================================================

def system_health_check():
    """Perform system health check and data validation"""
    print("🔍 SYSTEM HEALTH CHECK")
    print("=" * 30)

    try:
        # Check data files
        products_df = load_products_data()
        daily_sales_df = load_daily_sales_data()
        monthly_sales_df = load_monthly_sales_data()

        print(f"✅ Products Data: {len(products_df)} records")
        print(f"✅ Daily Sales Data: {len(daily_sales_df)} records")
        print(f"✅ Monthly Sales Data: {len(monthly_sales_df)} records")

        # Validate data quality
        print("\n📊 Data Quality Check:")

        # Check for missing values
        missing_products = products_df.isnull().sum().sum()
        missing_daily = daily_sales_df.isnull().sum().sum()
        missing_monthly = monthly_sales_df.isnull().sum().sum()

        print(f"   Missing values in products: {missing_products}")
        print(f"   Missing values in daily sales: {missing_daily}")
        print(f"   Missing values in monthly sales: {missing_monthly}")

        # Check date ranges
        date_range = daily_sales_df['date'].max() - daily_sales_df['date'].min()
        print(f"   Daily sales date range: {date_range.days} days")

        # Check product categories
        categories = products_df['category'].value_counts()
        print(f"   Product categories: {len(categories)} types")

        print("\n✅ System health check completed successfully!")
        return True

    except Exception as e:
        print(f"❌ System health check failed: {e}")
        return False

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("🚀 INITIALIZING GLOBAL SUPPLY CHAIN OPTIMIZATION SYSTEM")
    print("=" * 65)

    # System health check
    if not system_health_check():
        print("❌ System health check failed. Generating fresh data...")
        generate_realistic_supply_chain_data()

    print()

    # Run optimization
    try:
        recommendations = run_global_supply_chain_optimization()

        print(f"\n🎯 SYSTEM OUTPUT (JSON FORMAT):")
        print(json.dumps(recommendations, indent=2))

        print(f"\n✅ Supply chain optimization completed successfully!")
        print(f"📊 Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        print(f"❌ Optimization failed: {e}")

    print(f"\n💡 Ready for database migration - just replace CSV loading functions with database queries!")
