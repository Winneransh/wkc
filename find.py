import os
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types

@dataclass
class MerchandiseItem:
    """Data class for merchandise items"""
    name: str
    movie: str
    category: str
    price_range: str
    availability: str
    store: str
    description: str

class MovieMerchandiseSearchSystem:
    """
    A two-LLM system using Gemini in both stages:
    1. First Gemini (with grounding search) finds latest movie merchandise data
    2. Second Gemini (without grounding) processes and formats results into structured answer
    """
    
    def __init__(self, gemini_api_key: str):
        """Initialize the system with Gemini API key"""
        self.gemini_client = genai.Client(api_key=gemini_api_key)
        
        # Configure Gemini grounding tool for first LLM
        self.grounding_tool = types.Tool(
            google_search=types.GoogleSearch()
        )
        
        self.gemini_grounding_config = types.GenerateContentConfig(
            tools=[self.grounding_tool]
        )
        
        # Config for second LLM (no grounding tools)
        self.gemini_processing_config = types.GenerateContentConfig()
    
    def search_movie_merchandise_with_gemini(self, year: int = 2025) -> Dict[str, Any]:
        """
        Step 1: Use Gemini with grounding search to find latest movie merchandise
        """
        search_prompt = f"""
        Find the latest movie merchandise that can be sold in stores in {year}. 
        I need comprehensive information about:
        - Popular movie merchandise items currently available in retail stores
        - New movie tie-in products launched in {year}
        - Collectibles, toys, clothing, accessories from recent blockbuster movies
        - Which stores are selling these items (Target, Walmart, Hot Topic, GameStop, etc.)
        - Price ranges for different categories
        - Availability status (in stock, limited edition, pre-order, etc.)
        
        Focus on major movie franchises and new releases in {year} like:
        - Marvel/DC superhero movies
        - Disney animated films
        - Popular franchises (Star Wars, Harry Potter, etc.)
        - New blockbuster releases
        - Horror movie merchandise
        - Anime/manga movie adaptations
        
        Please provide detailed, current information with specific examples.
        """
        
        try:
            # First LLM: Gemini with grounding search
            response = self.gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=search_prompt,
                config=self.gemini_grounding_config,
            )
            
            # Extract search results and metadata
            result = {
                "raw_response": response.text,
                "search_queries": [],
                "sources": [],
                "grounding_metadata": None
            }
            
            # Extract grounding metadata if available
            if response.candidates[0].grounding_metadata:
                metadata = response.candidates[0].grounding_metadata
                result["grounding_metadata"] = metadata
                
                # Extract search queries
                if hasattr(metadata, 'web_search_queries'):
                    result["search_queries"] = metadata.web_search_queries
                
                # Extract sources
                if hasattr(metadata, 'grounding_chunks'):
                    result["sources"] = [
                        {
                            "title": chunk.web.title,
                            "uri": chunk.web.uri
                        } for chunk in metadata.grounding_chunks
                    ]
            
            print("✅ First LLM (Gemini with grounding) completed search")
            print(f"📊 Found {len(result['sources'])} sources")
            print(f"🔍 Used {len(result['search_queries'])} search queries")
            
            return result
            
        except Exception as e:
            print(f"❌ Error in first LLM search: {str(e)}")
            return {"error": str(e)}
    
    def process_results_with_gemini(self, search_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 2: Use second Gemini (without grounding) to process and structure the results
        """
        if "error" in search_results:
            return {"error": "Cannot process results due to search error"}
        
        processing_prompt = f"""
        Based on the following search results about movie merchandise, create a well-structured, 
        comprehensive answer that lists specific products available in stores in 2025.
        
        Raw search results:
        {search_results['raw_response']}
        
        Sources used:
        {json.dumps(search_results['sources'], indent=2)}
        
        Please create a structured response that includes:
        
        1. **Executive Summary**: Brief overview of the current movie merchandise market
        
        2. **Popular Categories**: Organize merchandise by categories (toys, clothing, collectibles, etc.)
        
        3. **Detailed Product List**: For each category, provide specific products with:
           - Product name
           - Associated movie/franchise
           - Price range
           - Where to buy (specific stores)
           - Availability status
           - Brief description
        
        4. **Trending Items**: Highlight what's currently most popular or selling well
        
        5. **Store Breakdown**: List which stores have the best selection for different categories
        
        6. **Price Analysis**: Give insights into pricing trends across different product types
        
        Format your response in clear sections with bullet points for easy reading.
        Make sure to include specific, actionable information that someone could use to actually 
        find and purchase these items.
        
        If the search results seem incomplete or unclear, note what additional information 
        would be helpful to get a complete picture.
        """
        
        try:
            # Second LLM: Gemini without grounding for processing
            response = self.gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=processing_prompt,
                config=self.gemini_processing_config,
            )
            
            print("✅ Second LLM (Gemini processing) completed analysis")
            
            return {
                "processed_response": response.text,
                "original_sources": search_results.get("sources", []),
                "search_queries_used": search_results.get("search_queries", []),
                "processing_successful": True
            }
            
        except Exception as e:
            print(f"❌ Error in second LLM processing: {str(e)}")
            return {"error": str(e), "processing_successful": False}
    
    def get_movie_merchandise_report(self, year: int = 2025) -> Dict[str, Any]:
        """
        Complete workflow: Search + Process
        """
        print(f"🎬 Starting movie merchandise search for {year}")
        print("=" * 50)
        
        # Step 1: Search with grounding
        print("🔍 Step 1: Searching with Gemini + Google Search...")
        search_results = self.search_movie_merchandise_with_gemini(year)
        
        if "error" in search_results:
            return {"error": f"Search failed: {search_results['error']}"}
        
        # Step 2: Process results
        print("🧠 Step 2: Processing results with second Gemini...")
        processed_results = self.process_results_with_gemini(search_results)
        
        if "error" in processed_results:
            return {"error": f"Processing failed: {processed_results['error']}"}
        
        # Combine results
        final_result = {
            "year": year,
            "search_metadata": {
                "sources_count": len(search_results.get("sources", [])),
                "search_queries": search_results.get("search_queries", []),
                "sources": search_results.get("sources", [])
            },
            "raw_search_data": search_results.get("raw_response", ""),
            "final_report": processed_results.get("processed_response", ""),
            "success": processed_results.get("processing_successful", False)
        }
        
        print("🎉 Complete! Two-LLM process finished successfully")
        return final_result
    
    def display_report(self, report: Dict[str, Any]):
        """
        Display the final report in a formatted way
        """
        if "error" in report:
            print(f"❌ Error: {report['error']}")
            return
        
        print("\n" + "="*80)
        print(f"🎬 MOVIE MERCHANDISE REPORT - {report['year']}")
        print("="*80)
        
        print(f"\n📊 Search Statistics:")
        print(f"   • Sources consulted: {report['search_metadata']['sources_count']}")
        print(f"   • Search queries used: {len(report['search_metadata']['search_queries'])}")
        
        if report['search_metadata']['search_queries']:
            print(f"   • Queries: {', '.join(report['search_metadata']['search_queries'])}")
        
        print(f"\n📋 FINAL REPORT:")
        print("-" * 40)
        print(report['final_report'])
        
        if report['search_metadata']['sources']:
            print(f"\n🔗 Sources Used:")
            for i, source in enumerate(report['search_metadata']['sources'], 1):
                print(f"   {i}. {source['title']}")
                print(f"      {source['uri']}")

# Usage Example
def main():
    """
    Example usage of the two-LLM movie merchandise search system
    """
    # Initialize the system
    api_key = ""
    if not api_key:
        print("❌ Please set GEMINI_API_KEY environment variable")
        return
    
    system = MovieMerchandiseSearchSystem(api_key)
    
    # Get comprehensive report
    report = system.get_movie_merchandise_report(2025)
    
    # Display results
    system.display_report(report)
    
    # Optionally save to file
    with open("movie_merchandise_report_2025.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\n💾 Report saved to movie_merchandise_report_2025.json")

if __name__ == "__main__":
    main()

    
