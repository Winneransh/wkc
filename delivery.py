import os
import json
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from google import genai
from google.genai import types

# =============================================================================
# DELIVERY PATH OPTIMIZATION CONFIGURATION
# =============================================================================

class DeliveryConfig:
    """Configuration for delivery path optimization"""
    
    # API Configuration
    GOOGLE_API_KEY = "AIzaSyCPbIA3-rJWEZSZAbMCv0fg7RgadQUa5-I"
    
    # System Configuration
    CURRENT_DATE = datetime.now()
    MAX_ROUTES = 5  # Maximum number of alternative routes to analyze
    MAX_STATIONS_PER_ROUTE = 8  # Maximum intermediate stations per route
    SEARCH_RADIUS_DAYS = 7  # Look for events within 7 days of travel date
    
    # Analysis Configuration
    PRIORITY_FACTORS = [
        "travel_time",
        "event_disruptions", 
        "weather_conditions",
        "road_safety",
        "fuel_efficiency"
    ]

config = DeliveryConfig()

# =============================================================================
# DATA CLASSES FOR DELIVERY OPTIMIZATION
# =============================================================================

@dataclass
class RouteStation:
    """Individual station/city in a route"""
    name: str
    latitude: float = 0.0
    longitude: float = 0.0
    estimated_travel_time: str = ""
    distance_from_previous: str = ""
    
@dataclass
class DeliveryRoute:
    """Complete delivery route with all stations"""
    route_id: str
    name: str
    origin: str
    destination: str
    stations: List[RouteStation]
    total_distance: str
    total_time: str
    route_type: str  # highway, state_road, mixed
    
@dataclass
class StationEvent:
    """Event affecting a station/city"""
    station_name: str
    event_type: str  # festival, accident, landslide, construction, weather
    event_description: str
    severity: str  # low, medium, high, critical
    impact_on_travel: str
    date_range: str
    alternative_suggestions: str

@dataclass
class OptimizedPath:
    """Final optimized delivery path recommendation"""
    route: DeliveryRoute
    events_analysis: List[StationEvent]
    risk_score: float
    confidence_score: float
    total_estimated_time: str
    cost_estimate: str
    recommendation_reason: str
    alternative_routes: List[str]

# =============================================================================
# MULTI-LLM DELIVERY PATH OPTIMIZATION SYSTEM
# =============================================================================

class DeliveryPathOptimizer:
    """
    4-LLM system for intelligent delivery path optimization:
    1. Route Discovery LLM: Finds all possible routes and stations
    2. Route Formatter LLM: Standardizes and corrects route data
    3. Event Analyzer LLM: Analyzes events/disruptions at each station
    4. Path Decision LLM: Selects optimal path considering all factors
    """
    
    def __init__(self, api_key: str = None):
        """Initialize the multi-LLM delivery optimization system"""
        self.api_key = api_key or config.GOOGLE_API_KEY
        
        # Initialize Gemini client
        self.gemini_client = genai.Client(api_key=self.api_key)
        
        # Configure grounding tool for search LLMs
        self.grounding_tool = types.Tool(
            google_search=types.GoogleSearch()
        )
        
        self.gemini_grounding_config = types.GenerateContentConfig(
            tools=[self.grounding_tool],
            temperature=0.1
        )
        
        # Config for processing LLMs
        self.gemini_processing_config = types.GenerateContentConfig(
            temperature=0.1
        )
    
    def llm1_discover_routes(self, origin: str, destination: str, travel_date: str) -> Dict[str, Any]:
        """
        LLM 1: Route Discovery Agent - Finds all possible delivery routes with stations
        """
        discovery_prompt = f"""
        You are a logistics route discovery specialist for delivery optimization in India.
        
        ROUTE DISCOVERY TASK:
        Find all major delivery routes from {origin} to {destination} for travel date: {travel_date}
        
        REQUIREMENTS:
        1. Find 4-5 different route options (highway, state roads, alternative paths)
        2. For each route, identify 4-8 major intermediate cities/stations
        3. Get accurate distance and time estimates for each segment
        4. Include different route types (NH highways, state highways, mixed routes)
        
        SEARCH FOCUS:
        - Best delivery routes from {origin} to {destination}
        - Major cities and stations between {origin} and {destination}
        - Highway routes NH (National Highway) options
        - State highway and alternative road options
        - Current traffic conditions and route status
        - Distance and time estimates for each route segment
        - Truck-friendly routes and restrictions
        
        Find comprehensive route information including:
        - Route names and highway numbers
        - Major intermediate cities/stations with distances
        - Estimated travel times for each segment
        - Total route distance and time
        - Route conditions and accessibility
        - Any known construction or permanent diversions
        
        Provide detailed route mapping with specific station names and accurate timing estimates.
        """
        
        try:
            response = self.gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=discovery_prompt,
                config=self.gemini_grounding_config,
            )
            
            # Extract search metadata
            result = {
                "raw_route_data": response.text,
                "sources": [],
                "search_success": True
            }
            
            # Extract grounding metadata
            if response.candidates[0].grounding_metadata:
                metadata = response.candidates[0].grounding_metadata
                if hasattr(metadata, 'grounding_chunks'):
                    result["sources"] = [
                        {
                            "title": getattr(chunk.web, 'title', 'Unknown Title'),
                            "uri": getattr(chunk.web, 'uri', 'Unknown URI')
                        } for chunk in metadata.grounding_chunks
                    ]
            
            print(f"✅ LLM 1 (Route Discovery): Found route data from {len(result['sources'])} sources")
            return result
            
        except Exception as e:
            print(f"❌ LLM 1 Error: {str(e)}")
            return {"search_success": False, "error": str(e)}
    
    def llm2_format_routes(self, raw_route_data: Dict[str, Any], origin: str, destination: str) -> Dict[str, Any]:
        """
        LLM 2: Route Formatter - Standardizes route data into structured format
        """
        if not raw_route_data.get("search_success", False):
            return {"format_success": False, "error": "No valid route data from LLM 1"}
        
        formatting_prompt = f"""
        You are a data formatting specialist for logistics systems.
        
        TASK: Convert raw route discovery data into standardized JSON format.
        
        RAW ROUTE DATA:
        {raw_route_data.get('raw_route_data', '')}
        
        ORIGIN: {origin}
        DESTINATION: {destination}
        
        FORMAT REQUIREMENTS:
        Convert the route information into exactly this JSON structure:
        
        {{
            "routes": [
                {{
                    "route_id": "route_1",
                    "route_name": "Route name (e.g., NH-1 via Delhi)",
                    "route_type": "highway/state_road/mixed",
                    "total_distance": "XXX km",
                    "total_time": "XX hours XX minutes",
                    "stations": [
                        {{
                            "station_name": "City/Station name",
                            "distance_from_previous": "XX km",
                            "estimated_time_from_previous": "XX minutes",
                            "cumulative_distance": "XXX km",
                            "cumulative_time": "XX hours XX minutes"
                        }}
                    ]
                }}
            ],
            "analysis_notes": "Any important observations about routes",
            "data_quality": "high/medium/low based on source reliability"
        }}
        
        FORMATTING RULES:
        1. Extract exactly 4-5 distinct routes
        2. Each route must have 4-8 intermediate stations (not including origin/destination)
        3. Include accurate distance and time estimates
        4. Use consistent naming for cities (proper spelling)
        5. Ensure cumulative calculations are correct
        6. Mark route type accurately
        7. If data is unclear, mark as "estimated" in the values
        
        Return ONLY valid JSON in the exact format specified above.
        """
        
        try:
            response = self.gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=formatting_prompt,
                config=self.gemini_processing_config,
            )
            
            # Parse JSON response
            import re
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                result["format_success"] = True
                
                routes_found = len(result.get("routes", []))
                print(f"✅ LLM 2 (Route Formatter): Structured {routes_found} routes")
                return result
            else:
                print("❌ LLM 2: Failed to generate valid JSON")
                return {"format_success": False, "error": "Invalid JSON response"}
                
        except Exception as e:
            print(f"❌ LLM 2 Error: {str(e)}")
            return {"format_success": False, "error": str(e)}
    
    def llm3_analyze_station_events(self, formatted_routes: Dict[str, Any], travel_date: str) -> Dict[str, Any]:
        """
        LLM 3: Event Analyzer - Checks for events/disruptions at each station
        """
        if not formatted_routes.get("format_success", False):
            return {"analysis_success": False, "error": "No valid formatted routes"}
        
        routes = formatted_routes.get("routes", [])
        all_stations = set()
        
        # Extract all unique stations from all routes
        for route in routes:
            for station in route.get("stations", []):
                all_stations.add(station.get("station_name", ""))
        
        stations_list = list(all_stations)
        travel_datetime = datetime.strptime(travel_date, "%Y-%m-%d")
        date_range_start = (travel_datetime - timedelta(days=3)).strftime("%Y-%m-%d")
        date_range_end = (travel_datetime + timedelta(days=3)).strftime("%Y-%m-%d")
        
        events_prompt = f"""
        You are an event monitoring specialist for logistics and transportation.
        
        TASK: Analyze potential events and disruptions for delivery route stations.
        
        STATIONS TO ANALYZE:
        {', '.join(stations_list)}
        
        TRAVEL DATE: {travel_date}
        MONITORING PERIOD: {date_range_start} to {date_range_end}
        
        SEARCH FOR EACH STATION:
        1. CURRENT EVENTS & DISRUPTIONS:
           - Traffic accidents or road closures
           - Construction work or diversions
           - Weather-related issues (floods, landslides, storms)
           - Political events or bandhs/strikes
           
        2. SCHEDULED EVENTS:
           - Festivals or religious celebrations
           - Political rallies or public events
           - Sports events or large gatherings
           - Market days or agricultural events
           
        3. INFRASTRUCTURE STATUS:
           - Road conditions and maintenance
           - Bridge or highway status
           - Fuel availability and costs
           - Rest stop and facility availability
        
        4. SEASONAL FACTORS:
           - Weather patterns for the time period
           - Regional seasonal challenges
           - Agricultural or harvest-related traffic
        
        Search specifically for:
        - "{travel_date} events [station_name]"
        - "Road conditions [station_name] {travel_date}"
        - "Traffic disruptions [station_name]"
        - "Festivals celebrations [station_name] {datetime.strptime(travel_date, '%Y-%m-%d').strftime('%B %Y')}"
        - "Weather forecast [station_name] {travel_date}"
        
        Provide comprehensive event analysis with specific dates, impact levels, and alternative suggestions.
        """
        
        try:
            response = self.gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=events_prompt,
                config=self.gemini_grounding_config,
            )
            
            result = {
                "raw_events_analysis": response.text,
                "stations_analyzed": stations_list,
                "travel_date": travel_date,
                "monitoring_period": f"{date_range_start} to {date_range_end}",
                "sources": [],
                "analysis_success": True
            }
            
            # Extract sources
            if response.candidates[0].grounding_metadata:
                metadata = response.candidates[0].grounding_metadata
                if hasattr(metadata, 'grounding_chunks'):
                    result["sources"] = [
                        {
                            "title": getattr(chunk.web, 'title', 'Unknown Title'),
                            "uri": getattr(chunk.web, 'uri', 'Unknown URI')
                        } for chunk in metadata.grounding_chunks
                    ]
            
            print(f"✅ LLM 3 (Event Analyzer): Analyzed {len(stations_list)} stations from {len(result['sources'])} sources")
            return result
            
        except Exception as e:
            print(f"❌ LLM 3 Error: {str(e)}")
            return {"analysis_success": False, "error": str(e)}
    
    def llm4_optimize_path(self, formatted_routes: Dict[str, Any], events_analysis: Dict[str, Any], 
                          origin: str, destination: str, travel_date: str) -> Dict[str, Any]:
        """
        LLM 4: Path Decision Maker - Selects optimal delivery path
        """
        if not events_analysis.get("analysis_success", False):
            return {"optimization_success": False, "error": "No valid events analysis"}
        
        optimization_prompt = f"""
        You are the final decision maker for delivery path optimization.
        
        DELIVERY REQUEST:
        - Origin: {origin}
        - Destination: {destination}
        - Travel Date: {travel_date}
        - Current Date: {config.CURRENT_DATE.strftime('%Y-%m-%d')}
        
        ROUTE OPTIONS:
        {json.dumps(formatted_routes.get('routes', []), indent=2)}
        
        EVENTS & DISRUPTIONS ANALYSIS:
        {events_analysis.get('raw_events_analysis', '')}
        
        STATIONS ANALYZED: {', '.join(events_analysis.get('stations_analyzed', []))}
        
        OPTIMIZATION CRITERIA (in priority order):
        1. SAFETY & ACCESSIBILITY: Avoid high-risk areas with accidents, landslides, severe weather
        2. TIME EFFICIENCY: Minimize total travel time considering traffic and delays
        3. EVENT IMPACT: Minimize disruptions from festivals, strikes, construction
        4. ROUTE RELIABILITY: Prefer well-maintained highways with good infrastructure
        5. COST EFFECTIVENESS: Consider fuel efficiency and toll costs
        
        DECISION FRAMEWORK:
        For each route, analyze:
        1. EVENT RISK SCORE (0-10): Based on severity of events at stations
        2. TIME RELIABILITY (0-10): Predictability of travel time
        3. SAFETY SCORE (0-10): Road safety and weather conditions
        4. OVERALL VIABILITY (0-10): Combined assessment
        
        OUTPUT (JSON FORMAT):
        {{
            "route_analysis": [
                {{
                    "route_id": "route_X",
                    "route_name": "Route name",
                    "risk_assessment": {{
                        "event_risk_score": 7,
                        "time_reliability_score": 8,
                        "safety_score": 9,
                        "overall_viability_score": 8.1
                    }},
                    "key_issues": ["issue1", "issue2"],
                    "advantages": ["advantage1", "advantage2"],
                    "estimated_delays": "XX minutes due to YY",
                    "recommendation": "recommended/acceptable/avoid"
                }}
            ],
            "optimal_path": {{
                "selected_route_id": "route_X",
                "confidence_score": 0.92,
                "total_estimated_time": "XX hours XX minutes (including delays)",
                "risk_level": "low/medium/high",
                "key_decision_factors": ["factor1", "factor2", "factor3"],
                "alternative_recommendations": ["route_Y as backup", "route_Z if weather clears"],
                "departure_recommendations": {{
                    "best_departure_time": "HH:MM",
                    "latest_safe_departure": "HH:MM",
                    "contingency_plan": "specific backup plan"
                }}
            }},
            "detailed_reasoning": "Comprehensive explanation of why this route was selected, covering all analysis factors and specific event considerations for {travel_date}."
        }}
        
        CRITICAL: Provide specific, actionable reasoning that addresses the actual events and conditions found for {travel_date}.
        
        Return ONLY valid JSON with detailed analysis.
        """
        
        try:
            response = self.gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=optimization_prompt,
                config=self.gemini_processing_config,
            )
            
            # Parse JSON response
            import re
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                result["optimization_success"] = True
                
                optimal_route = result.get("optimal_path", {}).get("selected_route_id", "Unknown")
                confidence = result.get("optimal_path", {}).get("confidence_score", 0.5)
                
                print(f"✅ LLM 4 (Path Optimizer): Selected {optimal_route} (confidence: {confidence:.2f})")
                return result
            else:
                print("❌ LLM 4: Failed to generate valid JSON")
                return {"optimization_success": False, "error": "Invalid JSON response"}
                
        except Exception as e:
            print(f"❌ LLM 4 Error: {str(e)}")
            return {"optimization_success": False, "error": str(e)}
    
    def optimize_delivery_path(self, origin: str, destination: str, travel_date: str) -> Dict[str, Any]:
        """
        Complete 4-LLM workflow for delivery path optimization
        """
        print(f"🚚 DELIVERY PATH OPTIMIZATION")
        print("=" * 60)
        print(f"📍 Origin: {origin}")
        print(f"🎯 Destination: {destination}")
        print(f"📅 Travel Date: {travel_date}")
        print(f"🤖 Multi-LLM Process: Route Discovery → Formatting → Event Analysis → Path Decision")
        print()
        
        # Step 1: Discover routes
        print("🗺️  Step 1: Discovering all possible delivery routes...")
        route_discovery = self.llm1_discover_routes(origin, destination, travel_date)
        
        if not route_discovery.get("search_success", False):
            return {"error": f"Route discovery failed: {route_discovery.get('error', 'Unknown error')}"}
        
        # Step 2: Format routes
        print("📊 Step 2: Formatting and structuring route data...")
        formatted_routes = self.llm2_format_routes(route_discovery, origin, destination)
        
        if not formatted_routes.get("format_success", False):
            return {"error": f"Route formatting failed: {formatted_routes.get('error', 'Unknown error')}"}
        
        # Step 3: Analyze events
        print("🚨 Step 3: Analyzing events and disruptions at all stations...")
        events_analysis = self.llm3_analyze_station_events(formatted_routes, travel_date)
        
        if not events_analysis.get("analysis_success", False):
            return {"error": f"Events analysis failed: {events_analysis.get('error', 'Unknown error')}"}
        
        # Step 4: Optimize path
        print("⚖️  Step 4: Selecting optimal delivery path...")
        optimization_result = self.llm4_optimize_path(formatted_routes, events_analysis, origin, destination, travel_date)
        
        if not optimization_result.get("optimization_success", False):
            return {"error": f"Path optimization failed: {optimization_result.get('error', 'Unknown error')}"}
        
        # Combine all results
        final_result = {
            "request": {
                "origin": origin,
                "destination": destination,
                "travel_date": travel_date,
                "analysis_timestamp": config.CURRENT_DATE.isoformat()
            },
            "route_discovery": {
                "sources_consulted": len(route_discovery.get("sources", [])),
                "raw_data_available": bool(route_discovery.get("raw_route_data"))
            },
            "route_options": formatted_routes.get("routes", []),
            "events_intelligence": {
                "stations_analyzed": events_analysis.get("stations_analyzed", []),
                "monitoring_period": events_analysis.get("monitoring_period", ""),
                "sources_consulted": len(events_analysis.get("sources", []))
            },
            "optimal_recommendation": optimization_result.get("optimal_path", {}),
            "route_analysis": optimization_result.get("route_analysis", []),
            "detailed_reasoning": optimization_result.get("detailed_reasoning", ""),
            "success": True
        }
        
        print("✅ Complete! Optimal delivery path determined.")
        return final_result
    
    def display_optimization_result(self, result: Dict[str, Any]):
        """
        Display the optimization result in a formatted way
        """
        if "error" in result:
            print(f"❌ Optimization Error: {result['error']}")
            return
        
        if not result.get("success", False):
            print("❌ Optimization was not successful")
            return
        
        print("\n" + "="*80)
        print("🚚 DELIVERY PATH OPTIMIZATION RESULTS")
        print("="*80)
        
        # Request Summary
        request = result["request"]
        print(f"\n📋 DELIVERY REQUEST:")
        print(f"   🚀 From: {request['origin']}")
        print(f"   🎯 To: {request['destination']}")
        print(f"   📅 Travel Date: {request['travel_date']}")
        print(f"   🕐 Analysis Time: {request['analysis_timestamp']}")
        
        # Route Discovery Summary
        discovery = result["route_discovery"]
        print(f"\n🗺️  ROUTE DISCOVERY:")
        print(f"   📊 Routes Found: {len(result.get('route_options', []))}")
        print(f"   📚 Sources Consulted: {discovery['sources_consulted']}")
        
        # Events Analysis Summary
        events = result["events_intelligence"]
        print(f"\n🚨 EVENTS ANALYSIS:")
        print(f"   🏙️  Stations Monitored: {len(events['stations_analyzed'])}")
        print(f"   📅 Monitoring Period: {events['monitoring_period']}")
        print(f"   📰 Intelligence Sources: {events['sources_consulted']}")
        print(f"   🏪 Stations: {', '.join(events['stations_analyzed'][:5])}{'...' if len(events['stations_analyzed']) > 5 else ''}")
        
        # Optimal Recommendation
        optimal = result["optimal_recommendation"]
        print(f"\n🎯 OPTIMAL RECOMMENDATION:")
        print(f"   🛣️  Selected Route: {optimal.get('selected_route_id', 'N/A')}")
        print(f"   ⏱️  Estimated Time: {optimal.get('total_estimated_time', 'N/A')}")
        print(f"   🎲 Confidence: {optimal.get('confidence_score', 0):.1%}")
        print(f"   ⚠️  Risk Level: {optimal.get('risk_level', 'N/A').upper()}")
        
        departure = optimal.get('departure_recommendations', {})
        if departure:
            print(f"   🕐 Best Departure: {departure.get('best_departure_time', 'N/A')}")
            print(f"   🚨 Latest Safe Departure: {departure.get('latest_safe_departure', 'N/A')}")
        
        # Key Decision Factors
        factors = optimal.get('key_decision_factors', [])
        if factors:
            print(f"\n💡 KEY DECISION FACTORS:")
            for i, factor in enumerate(factors, 1):
                print(f"   {i}. {factor}")
        
        # Route Analysis
        route_analysis = result.get("route_analysis", [])
        if route_analysis:
            print(f"\n📊 ROUTE ANALYSIS SUMMARY:")
            for route in route_analysis:
                risk = route.get('risk_assessment', {})
                print(f"   🛣️  {route.get('route_name', 'Unknown Route')}:")
                print(f"      ⚠️  Risk Score: {risk.get('event_risk_score', 'N/A')}/10")
                print(f"      ⏱️  Reliability: {risk.get('time_reliability_score', 'N/A')}/10")
                print(f"      🛡️  Safety: {risk.get('safety_score', 'N/A')}/10")
                print(f"      📈 Overall: {risk.get('overall_viability_score', 'N/A')}/10")
                print(f"      💭 Status: {route.get('recommendation', 'N/A').upper()}")
        
        # Detailed Reasoning
        reasoning = result.get("detailed_reasoning", "")
        if reasoning:
            print(f"\n🧠 DETAILED REASONING:")
            print(f"   {reasoning}")
        
        # Alternative Recommendations
        alternatives = optimal.get('alternative_recommendations', [])
        if alternatives:
            print(f"\n🔄 BACKUP OPTIONS:")
            for alt in alternatives:
                print(f"   • {alt}")

# =============================================================================
# USAGE EXAMPLE AND MAIN EXECUTION
# =============================================================================

def main():
    """
    Example usage of the delivery path optimization system
    """
    # Initialize the system
    optimizer = DeliveryPathOptimizer()
    
    # Example delivery request
    origin = "shamli, uttar pradesh"
    destination = "delhi, Delhi"
    travel_date = "2025-07-21"  # Example future date
    
    print("🚀 INITIALIZING DELIVERY PATH OPTIMIZATION SYSTEM")
    print("=" * 65)
    
    try:
        # Run optimization
        result = optimizer.optimize_delivery_path(origin, destination, travel_date)
        
        # Display results
        optimizer.display_optimization_result(result)
        
        # Save results to file
        filename = f"delivery_optimization_{travel_date}_{origin.replace(' ', '_').replace(',', '')}_to_{destination.replace(' ', '_').replace(',', '')}.json"
        with open(filename, "w") as f:
            json.dump(result, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")
        
    except Exception as e:
        print(f"❌ System Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()