import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types
from langchain_google_genai import ChatGoogleGenerativeAI

# =============================================================================
# GLOBAL CONFIGURATION - CENTRALIZED SETTINGS (Same as original notebook)
# =============================================================================

class GlobalConfig:
    """Global configuration for supply chain optimization"""

    # API Configuration
    GOOGLE_API_KEY = "AIzaSyCPbIA3-rJWEZSZAbMCv0fg7RgadQUa5-I"  # Replace with your API key
    LLM_MODEL = "gemini-2.0-flash"
    LLM_TEMPERATURE = 0.1

    # Business Configuration
    CURRENT_DATE = datetime(2025, 10, 15)  # Mid-July for better context
    CURRENT_MONTH = 10
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

# =============================================================================
# MULTI-LLM WEATHER-AWARE SUPPLY CHAIN SYSTEM
# =============================================================================

@dataclass
class ProductAnalysisResult:
    """Data class for product analysis results"""
    product_id: int
    product_name: str
    category: str
    search_queries: List[str]
    weather_forecast: str
    market_trends: str
    seasonal_analysis: str
    final_recommendation: str
    confidence_score: float
    order_quantity: int
    reasoning: str

class WeatherAwareSupplyChainSystem:
    """
    Multi-LLM system for weather-aware supply chain decisions:
    1. First LLM (Prompt Generator): Analyzes product and creates search queries
    2. Second LLM (Search Agent): Uses grounding search to gather market intelligence
    3. Third LLM (Decision Maker): Makes final purchasing decision based on all data
    """
    
    def __init__(self, gemini_api_key: str = None):
        """Initialize the multi-LLM system"""
        self.api_key = gemini_api_key or config.GOOGLE_API_KEY
        
        # Initialize Gemini client for grounding search
        self.gemini_client = genai.Client(api_key=self.api_key)
        
        # Configure grounding tool for search LLM
        self.grounding_tool = types.Tool(
            google_search=types.GoogleSearch()
        )
        
        self.gemini_grounding_config = types.GenerateContentConfig(
            tools=[self.grounding_tool],
            temperature=0.1
        )
        
        # Config for non-grounding LLMs
        self.gemini_processing_config = types.GenerateContentConfig(
            temperature=0.1
        )
        
        # Initialize LangChain LLM for consistency with original system
        self.langchain_llm = ChatGoogleGenerativeAI(
            model=config.LLM_MODEL,
            google_api_key=self.api_key,
            temperature=config.LLM_TEMPERATURE,
            convert_system_message_to_human=True
        )
    
    def llm1_generate_search_strategy(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        LLM 1: Prompt Generator - Analyzes product and creates intelligent search queries
        """
        prompt = f"""
        You are a supply chain intelligence analyst for a retail chain in {config.LOCATION}.
        
        BUSINESS CONTEXT:
        - Current Date: {config.CURRENT_DATE.strftime('%Y-%m-%d')}
        - Current Month: {config.CURRENT_MONTH}
        - Location: {config.LOCATION}
        - Store Network: {config.NUMBER_OF_STORES} stores
        
        PRODUCT TO ANALYZE:
        - Product: {product_data['product_name']} (ID: {product_data['product_id']})
        - Category: {product_data.get('category', 'Unknown')}
        - Current Inventory: {product_data['quantity_in_inventory']} units
        - Shelf Life: {product_data['shelf_life_days']} days
        - Lead Time: {product_data['lead_time_days']} days
        
        YOUR TASK:
        Analyze this product and determine what market intelligence we need to make an informed purchasing decision.
        
        Consider:
        1. Weather/Climate Impact: How do weather patterns in {config.LOCATION} affect demand?
        2. Seasonal Timing: Given it's {config.CURRENT_DATE.strftime('%B %Y')}, what seasonal factors matter?
        3. Market Trends: What current market conditions could influence demand?
        4. Regional Factors: How do local conditions in {config.LOCATION} impact this product category?
        
        OUTPUT FORMAT (JSON):
        {{
            "product_analysis": {{
                "weather_dependency": "high/medium/low",
                "seasonal_impact": "peak/rising/stable/declining/off-season",
                "market_sensitivity": "high/medium/low",
                "regional_factors": ["factor1", "factor2", "factor3"]
            }},
            "search_queries": [
                "query1 for weather/climate data",
                "query2 for market trends", 
                "query3 for seasonal patterns",
                "query4 for regional demand patterns"
            ],
            "priority_factors": ["most important factor", "second factor", "third factor"],
            "analysis_focus": "detailed explanation of what to focus on"
        }}
        
        Return ONLY valid JSON.
        """
        
        try:
            response = self.gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=self.gemini_processing_config,
            )
            
            # Parse JSON response
            import re
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                result["llm1_success"] = True
                print(f"✅ LLM 1 (Prompt Generator): Created {len(result.get('search_queries', []))} search queries")
                return result
            else:
                print("❌ LLM 1: Failed to generate valid JSON")
                return {"llm1_success": False, "error": "Invalid JSON response"}
                
        except Exception as e:
            print(f"❌ LLM 1 Error: {str(e)}")
            return {"llm1_success": False, "error": str(e)}
    
    def llm2_gather_market_intelligence(self, search_strategy: Dict[str, Any], product_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        LLM 2: Search Agent - Uses grounding search to gather comprehensive market intelligence
        """
        if not search_strategy.get("llm1_success", False):
            return {"llm2_success": False, "error": "No valid search strategy from LLM 1"}
        
        search_queries = search_strategy.get("search_queries", [])
        
        # Combine all search queries into comprehensive prompt
        combined_search_prompt = f"""
        Gather comprehensive market intelligence for product purchasing decision in {config.LOCATION}.
        
        CONTEXT:
        - Location: {config.LOCATION}
        - Current Date: {config.CURRENT_DATE.strftime('%B %d, %Y')}
        - Product: {product_data['product_name']}
        - Category: {product_data.get('category', 'Unknown')}
        
        SEARCH FOCUS AREAS:
        {chr(10).join([f"• {query}" for query in search_queries])}
        
        Please find current information about:
        
        1. WEATHER & CLIMATE:
        - Current weather patterns in {config.LOCATION}
        - Weather forecast for next 30-60 days in {config.LOCATION}
        - How weather affects demand for {product_data['product_name']} in India
        - Climate trends that could impact seasonal patterns
        
        2. MARKET TRENDS:
        - Current market demand for {product_data['product_name']} in India
        - Recent price trends and market conditions
        - Consumer behavior changes in {config.CURRENT_DATE.year}
        - Supply chain disruptions or opportunities
        
        3. SEASONAL PATTERNS:
        - Historical demand patterns for {product_data['product_name']} in {config.LOCATION}
        - When peak/off seasons typically occur for this product category
        - How {config.CURRENT_DATE.strftime('%B')} historically performs for this product
        
        4. REGIONAL FACTORS:
        - Local preferences in {config.LOCATION} for {product_data['product_name']}
        - Regional events, festivals, or factors affecting demand
        - Competition and market saturation in the area
        
        Provide specific, actionable intelligence that can inform purchasing decisions.
        """
        
        try:
            # Use Gemini with grounding search
            response = self.gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=combined_search_prompt,
                config=self.gemini_grounding_config,
            )
            
            # Extract search results and metadata
            result = {
                "raw_intelligence": response.text,
                "search_queries_used": search_queries,
                "sources": [],
                "grounding_metadata": None,
                "llm2_success": True
            }
            
            # Extract grounding metadata if available
            if response.candidates[0].grounding_metadata:
                metadata = response.candidates[0].grounding_metadata
                result["grounding_metadata"] = metadata
                
                # Extract sources
                if hasattr(metadata, 'grounding_chunks'):
                    result["sources"] = [
                        {
                            "title": getattr(chunk.web, 'title', 'Unknown Title'),
                            "uri": getattr(chunk.web, 'uri', 'Unknown URI')
                        } for chunk in metadata.grounding_chunks
                    ]
            
            print(f"✅ LLM 2 (Search Agent): Gathered intelligence from {len(result['sources'])} sources")
            return result
            
        except Exception as e:
            print(f"❌ LLM 2 Error: {str(e)}")
            return {"llm2_success": False, "error": str(e)}
    
    def llm3_make_final_decision(self, product_data: Dict[str, Any], search_strategy: Dict[str, Any], 
                                market_intelligence: Dict[str, Any], 
                                historical_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        LLM 3: Decision Maker - Makes final purchasing decision based on all gathered intelligence
        """
        if not market_intelligence.get("llm2_success", False):
            return {"llm3_success": False, "error": "No valid market intelligence from LLM 2"}
        
        decision_prompt = f"""
        You are the final decision maker for supply chain optimization at a retail chain in {config.LOCATION}.
        
        BUSINESS CONTEXT:
        - Current Date: {config.CURRENT_DATE.strftime('%Y-%m-%d')}
        - Location: {config.LOCATION}
        - Store Network: {config.NUMBER_OF_STORES} stores
        - Warehouse Capacity: {config.TOTAL_INVENTORY_CAPACITY} units
        
        PRODUCT INFORMATION:
        - Product: {product_data['product_name']} (ID: {product_data['product_id']})
        - Category: {product_data.get('category', 'Unknown')}
        - Current Inventory: {product_data['quantity_in_inventory']} units
        - Store Stock: {product_data['current_quantity_in_store']} units
        - Total Available: {product_data['quantity_in_inventory'] + product_data['current_quantity_in_store']} units
        - Cost Price: ₹{product_data['cost_price']}
        - Selling Price: ₹{product_data['selling_price']}
        - Shelf Life: {product_data['shelf_life_days']} days
        - Lead Time: {product_data['lead_time_days']} days
        
        HISTORICAL PERFORMANCE:
        {json.dumps(historical_metrics, indent=2)}
        
        SEARCH STRATEGY ANALYSIS:
        {json.dumps(search_strategy, indent=2)}
        
        CURRENT MARKET INTELLIGENCE:
        {market_intelligence.get('raw_intelligence', 'No intelligence available')}
        
        SOURCES CONSULTED:
        {json.dumps(market_intelligence.get('sources', []), indent=2)}
        
        DECISION FRAMEWORK:
        Based on the current date ({config.CURRENT_DATE.strftime('%B %d, %Y')}) and location ({config.LOCATION}), 
        analyze:
        
        1. WEATHER IMPACT: How do current and forecasted weather conditions affect demand?
        2. SEASONAL TIMING: Is this the right time to order based on seasonal patterns?
        3. MARKET CONDITIONS: What do current market trends suggest?
        4. INVENTORY RISK: Balance stockout risk vs. excess inventory risk
        5. PROFIT OPTIMIZATION: Consider profit margins and working capital efficiency
        
        OUTPUT (JSON FORMAT):
        {{
            "decision_summary": {{
                "weather_forecast_impact": "positive/negative/neutral with explanation",
                "seasonal_timing_assessment": "excellent/good/poor with reasoning",
                "market_conditions": "favorable/challenging/neutral with details",
                "inventory_urgency": "critical/moderate/low with timeline",
                "profit_potential": "high/medium/low with calculation"
            }},
            "final_recommendation": {{
                "action": "order_now/order_later/reduce_order/no_order",
                "quantity": 0,
                "confidence_score": 0.85,
                "reasoning": "detailed explanation of decision logic",
                "risk_factors": ["risk1", "risk2"],
                "opportunities": ["opportunity1", "opportunity2"]
            }},
            "timing_analysis": {{
                "best_order_timing": "immediate/1-2_weeks/next_month/seasonal_timing",
                "demand_forecast": "increasing/stable/decreasing",
                "competitive_landscape": "low/medium/high_competition"
            }}
        }}
        
        Return ONLY valid JSON with specific numeric quantity recommendation.
        """
        
        try:
            response = self.gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=decision_prompt,
                config=self.gemini_processing_config,
            )
            
            # Parse JSON response
            import re
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                result["llm3_success"] = True
                
                # Extract order quantity
                order_quantity = result.get("final_recommendation", {}).get("quantity", 0)
                confidence = result.get("final_recommendation", {}).get("confidence_score", 0.5)
                
                print(f"✅ LLM 3 (Decision Maker): Recommended {order_quantity} units (confidence: {confidence:.2f})")
                return result
            else:
                print("❌ LLM 3: Failed to generate valid JSON")
                return {"llm3_success": False, "error": "Invalid JSON response"}
                
        except Exception as e:
            print(f"❌ LLM 3 Error: {str(e)}")
            return {"llm3_success": False, "error": str(e)}
    
    def analyze_product_with_weather_intelligence(self, product_data: Dict[str, Any], 
                                                 historical_metrics: Dict[str, Any]) -> ProductAnalysisResult:
        """
        Complete multi-LLM workflow for a single product
        """
        product_name = product_data['product_name']
        product_id = product_data['product_id']
        
        print(f"\n🌦️  WEATHER-AWARE ANALYSIS: {product_name}")
        print("=" * 60)
        
        # Step 1: Generate search strategy
        print("🧠 Step 1: Analyzing product and generating search strategy...")
        search_strategy = self.llm1_generate_search_strategy(product_data)
        
        if not search_strategy.get("llm1_success", False):
            return ProductAnalysisResult(
                product_id=product_id,
                product_name=product_name,
                category=product_data.get('category', 'Unknown'),
                search_queries=[],
                weather_forecast="Error in analysis",
                market_trends="Error in analysis", 
                seasonal_analysis="Error in analysis",
                final_recommendation="No recommendation due to analysis error",
                confidence_score=0.0,
                order_quantity=0,
                reasoning=f"LLM 1 failed: {search_strategy.get('error', 'Unknown error')}"
            )
        
        # Step 2: Gather market intelligence
        print("🔍 Step 2: Gathering market intelligence with grounding search...")
        market_intelligence = self.llm2_gather_market_intelligence(search_strategy, product_data)
        
        if not market_intelligence.get("llm2_success", False):
            return ProductAnalysisResult(
                product_id=product_id,
                product_name=product_name,
                category=product_data.get('category', 'Unknown'),
                search_queries=search_strategy.get("search_queries", []),
                weather_forecast="Error in search",
                market_trends="Error in search",
                seasonal_analysis="Error in search", 
                final_recommendation="No recommendation due to search error",
                confidence_score=0.0,
                order_quantity=0,
                reasoning=f"LLM 2 failed: {market_intelligence.get('error', 'Unknown error')}"
            )
        
        # Step 3: Make final decision
        print("⚖️  Step 3: Making final purchasing decision...")
        final_decision = self.llm3_make_final_decision(product_data, search_strategy, 
                                                      market_intelligence, historical_metrics)
        
        if not final_decision.get("llm3_success", False):
            return ProductAnalysisResult(
                product_id=product_id,
                product_name=product_name,
                category=product_data.get('category', 'Unknown'),
                search_queries=search_strategy.get("search_queries", []),
                weather_forecast="Available from search",
                market_trends="Available from search",
                seasonal_analysis="Available from search",
                final_recommendation="No recommendation due to decision error",
                confidence_score=0.0,
                order_quantity=0,
                reasoning=f"LLM 3 failed: {final_decision.get('error', 'Unknown error')}"
            )
        
        # Create final result
        decision_data = final_decision.get("final_recommendation", {})
        decision_summary = final_decision.get("decision_summary", {})
        
        result = ProductAnalysisResult(
            product_id=product_id,
            product_name=product_name,
            category=product_data.get('category', 'Unknown'),
            search_queries=search_strategy.get("search_queries", []),
            weather_forecast=decision_summary.get("weather_forecast_impact", "Not analyzed"),
            market_trends=decision_summary.get("market_conditions", "Not analyzed"),
            seasonal_analysis=decision_summary.get("seasonal_timing_assessment", "Not analyzed"),
            final_recommendation=decision_data.get("action", "no_order"),
            confidence_score=decision_data.get("confidence_score", 0.5),
            order_quantity=decision_data.get("quantity", 0),
            reasoning=decision_data.get("reasoning", "No reasoning provided")
        )
        
        print(f"✅ Complete! Recommended: {result.order_quantity} units (confidence: {result.confidence_score:.2f})")
        return result

# =============================================================================
# INTEGRATION WITH ORIGINAL SUPPLY CHAIN SYSTEM
# =============================================================================

def run_weather_aware_supply_chain_optimization():
    """
    Enhanced version of the original system with weather-aware multi-LLM intelligence
    """
    print(f"🌦️  WEATHER-AWARE SUPPLY CHAIN OPTIMIZATION SYSTEM")
    print("=" * 70)
    print(f"📍 Location: {config.LOCATION}")
    print(f"📅 Analysis Date: {config.CURRENT_DATE.strftime('%Y-%m-%d')}")
    print(f"🏪 Store Network: {config.NUMBER_OF_STORES} stores")
    print(f"🤖 Multi-LLM Intelligence: Prompt Generator → Search Agent → Decision Maker")
    print()
    
    # Initialize weather-aware system
    weather_system = WeatherAwareSupplyChainSystem()
    
    # Load data (reuse functions from original system)
    print("📊 Loading supply chain data...")
    products_df = load_products_data()
    daily_sales_df = load_daily_sales_data()
    monthly_sales_df = load_monthly_sales_data()
    
    print(f"✅ Loaded {len(products_df)} products for weather-aware analysis")
    print()
    
    final_recommendations = {}
    total_order_value = 0
    analysis_results = []
    
    # Process each product with weather intelligence
    for _, product_row in products_df.iterrows():
        product_data = {
            'product_id': product_row['product_id'],
            'product_name': product_row['product_name'],
            'category': product_row.get('category', 'Unknown'),
            'quantity_in_inventory': product_row['quantity_in_inventory'],
            'current_quantity_in_store': product_row['current_quantity_in_store'],
            'shelf_life_days': product_row['shelf_life_days'],
            'lead_time_days': product_row['lead_time_days'],
            'cost_price': product_row['cost_price'],
            'selling_price': product_row['selling_price']
        }
        
        # Get historical metrics (reuse from original system)
        historical_metrics = {
            "profit_margin": ((product_row['selling_price'] - product_row['cost_price']) / product_row['selling_price']) * 100,
            "inventory_value": product_row['quantity_in_inventory'] * product_row['cost_price'],
            "days_of_inventory": (product_row['quantity_in_inventory'] + product_row['current_quantity_in_store']) / max(1, 20),  # Simplified
        }
        
        # Run weather-aware analysis
        result = weather_system.analyze_product_with_weather_intelligence(product_data, historical_metrics)
        analysis_results.append(result)
        
        # Store recommendations
        final_recommendations[str(result.product_id)] = result.order_quantity
        total_order_value += result.order_quantity * product_row['cost_price']
        
        print(f"   📊 Weather Impact: {result.weather_forecast}")
        print(f"   📈 Market Trends: {result.market_trends}")
        print(f"   🗓️  Seasonal Analysis: {result.seasonal_analysis}")
        print(f"   🎯 Final Decision: {result.final_recommendation}")
        print(f"   💰 Order Value: ₹{result.order_quantity * product_row['cost_price']:,}")
        print()
    
    # Summary Report
    print("🏆 WEATHER-AWARE OPTIMIZATION SUMMARY")
    print("=" * 50)
    print(f"📊 Products Analyzed: {len(products_df)}")
    print(f"💰 Total Recommended Order Value: ₹{total_order_value:,}")
    print(f"🤖 Multi-LLM Decisions: {sum(1 for r in analysis_results if r.order_quantity > 0)} orders recommended")
    
    high_confidence_decisions = [r for r in analysis_results if r.confidence_score > 0.8]
    print(f"🎯 High Confidence Decisions: {len(high_confidence_decisions)}")
    
    print()
    print("📋 FINAL WEATHER-AWARE RECOMMENDATIONS:")
    print("-" * 45)
    for result in analysis_results:
        if result.order_quantity > 0:
            product_cost = products_df[products_df['product_id'] == result.product_id]['cost_price'].iloc[0]
            value = result.order_quantity * product_cost
            print(f"• {result.product_name}: {result.order_quantity:,} units (₹{value:,}) - Confidence: {result.confidence_score:.2f}")
        else:
            print(f"• {result.product_name}: No order recommended")
    
    return final_recommendations, analysis_results

# =============================================================================
# HELPER FUNCTIONS (Reused from original system)
# =============================================================================

def load_products_data():
    """Load products inventory data"""
    # Simplified version - in practice, load from your CSV or database
    return pd.DataFrame({
        'product_id': [5001, 5002, 5003, 5004, 5005, 5006],
        'product_name': [
            'Mangoes (per kg)',
            'Air Conditioners (1.5 Ton)', 
            'Winter Jackets',
            'Sweaters',
            'Bread (per loaf)',
            'Cooking Oil (1L)'
        ],
        'quantity_in_inventory': [800, 150, 200, 500, 400, 600],
        'current_quantity_in_store': [120, 25, 40, 180, 80, 100],
        'cost_price': [60, 25000, 1200, 600, 20, 120],
        'selling_price': [95, 35000, 1800, 900, 35, 150],
        'shelf_life_days': [7, 1825, 1095, 1095, 3, 730],
        'lead_time_days': [3, 14, 30, 21, 7, 7],
        'category': [
            'Seasonal_High_LowInventory',
            'Seasonal_High_AdequateInventory',
            'Seasonal_Ending_AdequateInventory',
            'Seasonal_Ending_ExcessInventory',
            'NonSeasonal_HighDemand',
            'NonSeasonal_RegularDemand'
        ]
    })

def load_daily_sales_data():
    """Load daily sales data"""
    # Simplified - return empty DataFrame for this example
    return pd.DataFrame()

def load_monthly_sales_data():
    """Load monthly sales data"""
    # Simplified - return empty DataFrame for this example  
    return pd.DataFrame()

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("🚀 INITIALIZING WEATHER-AWARE SUPPLY CHAIN SYSTEM")
    print("=" * 65)
    
    try:
        recommendations, analysis_results = run_weather_aware_supply_chain_optimization()
        
        print(f"\n🎯 SYSTEM OUTPUT (JSON FORMAT):")
        print(json.dumps(recommendations, indent=2))
        
        # Save detailed analysis
        detailed_results = [
            {
                "product_id": r.product_id,
                "product_name": r.product_name,
                "category": r.category,
                "search_queries": r.search_queries,
                "weather_forecast": r.weather_forecast,
                "market_trends": r.market_trends,
                "seasonal_analysis": r.seasonal_analysis,
                "final_recommendation": r.final_recommendation,
                "confidence_score": r.confidence_score,
                "order_quantity": r.order_quantity,
                "reasoning": r.reasoning
            }
            for r in analysis_results
        ]
        
        with open(f"weather_aware_analysis_{config.CURRENT_DATE.strftime('%Y%m%d')}.json", "w") as f:
            json.dump(detailed_results, f, indent=2)
        
        print(f"✅ Weather-aware supply chain optimization completed successfully!")
        print(f"📊 Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💾 Detailed analysis saved to weather_aware_analysis_{config.CURRENT_DATE.strftime('%Y%m%d')}.json")
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()