import os
import requests
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# 1. Constants and Configuration
# ---------------------------------------------------------------------------
TAVILY_TOKEN = os.getenv("TAVILY_API_KEY")
TAVILY_SEARCH_URL = "https://api.tavily.com/search"
TAVILY_EXTRACT_URL = "https://api.tavily.com/extract"

# Default search configuration
DEFAULT_CONFIG = {
    "CURRENT_UTC": "2025-04-02 05:40:54",
    "CURRENT_USER": "admin",
    "DEFAULT_SEARCH_DEPTH": "advanced",
    "DEFAULT_MAX_RESULTS": 5,
    "DEFAULT_DAYS": 7,
    "RELIABLE_DOMAINS": [
        "nytimes.com", "cbsnews.com", "reuters.com", "apnews.com",
        "bjs.gov", "fbi.gov", "police.gov", "justice.gov",
        "crimereports.com", "abc.com", "nbcnews.com", "cnn.com"
    ],
    "EXCLUDED_DOMAINS": [
        "twitter.com", "reddit.com", "facebook.com", "tiktok.com",
        "instagram.com", "pinterest.com"
    ]
}

# ---------------------------------------------------------------------------
# 2. Utility Functions
# ---------------------------------------------------------------------------

def format_date_range(start_year, end_year):
    """Format date range for query enhancement"""
    if start_year and end_year:
        if start_year == end_year:
            return f"in {start_year}"
        return f"between {start_year} and {end_year}"
    return ""

# ---------------------------------------------------------------------------
# 3. Core Search Functions
# ---------------------------------------------------------------------------
def tavily_search(
    query,
    selected_regions=None,  # Changed from location
    start_year=None,
    end_year=None,
    search_mode="all_years",
    topic="general",
    search_depth=DEFAULT_CONFIG["DEFAULT_SEARCH_DEPTH"],
    max_results=DEFAULT_CONFIG["DEFAULT_MAX_RESULTS"],
    days=DEFAULT_CONFIG["DEFAULT_DAYS"],
    include_domains=None,
    exclude_domains=None
):
    """
    Enhanced Tavily search with support for historical crime data queries.
    
    Args:
        query (str): Main search query
        location (str, optional): Geographic location to focus search
        start_year (int, optional): Start year for historical data
        end_year (int, optional): End year for historical data
        search_mode (str): Either "all_years" or "specific_range"
        topic (str): Search topic category
        search_depth (str): Search depth level
        max_results (int): Maximum number of results to return
        days (int): Number of recent days to search
        include_domains (list): Specific domains to include
        exclude_domains (list): Domains to exclude
    """
    # Build enhanced query
    enhanced_query = query

    # Add date range if specified
    if search_mode == "specific_range" and start_year and end_year:
        date_range = format_date_range(start_year, end_year)
        enhanced_query = f"{enhanced_query} {date_range}"

    # Add regions if specified
    if selected_regions:
        regions_str = " OR ".join(selected_regions)
        enhanced_query = f"{enhanced_query} in ({regions_str})"


    # Use default reliable domains if none specified
    if not include_domains:
        include_domains = DEFAULT_CONFIG["RELIABLE_DOMAINS"]

    # Use default excluded domains if none specified
    if not exclude_domains:
        exclude_domains = DEFAULT_CONFIG["EXCLUDED_DOMAINS"]

    payload = {
        "query": enhanced_query,
        "topic": topic,
        "search_depth": search_depth,
        "chunks_per_source": 3,
        "max_results": max_results,
        "days": days,
        "include_answer": True,
        "include_raw_content": False,
        "include_images": True,
        "include_image_descriptions": True,
        "include_domains": include_domains,
        "exclude_domains": exclude_domains
    }

    headers = {
        "Authorization": f"Bearer {TAVILY_TOKEN}",
        "Content-Type": "application/json"
    }

    try:
        resp = requests.post(TAVILY_SEARCH_URL, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[Error] Tavily search failed: {e}")
        return None

def tavily_extract(urls, extract_depth="basic", include_images=True):
    """
    Extract detailed content from URLs with support for images.
    
    Args:
        urls (str or list): Single URL or list of URLs to extract from
        extract_depth (str): Depth of content extraction
        include_images (bool): Whether to include images in extraction
    """
    if isinstance(urls, str):
        urls = [urls]

    payload = {
        "urls": urls,
        "extract_depth": extract_depth,
        "include_images": include_images
    }
    
    headers = {
        "Authorization": f"Bearer {TAVILY_TOKEN}",
        "Content-Type": "application/json"
    }

    try:
        resp = requests.post(TAVILY_EXTRACT_URL, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[Error] Tavily extract failed: {e}")
        return None

# ---------------------------------------------------------------------------
# 4. Report Generation
# ---------------------------------------------------------------------------
def build_markdown_report(query, search_result, extracts):
    """
    Build a comprehensive markdown report with images and links.
    
    Returns:
        str: JSON string containing markdown_report, images, and links
    """
    all_images = []
    all_links = []
    
    if not search_result:
        return json.dumps({
            "markdown_report": f"# Search Error\nNo valid results for query: {query}",
            "images": all_images,
            "links": all_links
        })

    # Process extracts into URL-keyed dictionary
    extracts_by_url = {}
    if isinstance(extracts, list):
        for item in extracts:
            if isinstance(item, dict) and "url" in item:
                extracts_by_url[item["url"]] = item
    elif isinstance(extracts, dict):
        extracts_by_url = extracts

    # Get main components from search results
    answer = search_result.get("answer", "No summary provided.")
    items = search_result.get("results", [])

    if "images" in search_result:
        for img in search_result["images"]:
            if isinstance(img, dict) and "url" in img:
                img_url = img["url"]
                if img_url and img_url.startswith("http"):
                    all_images.append(img_url)
    for item in items:
        if isinstance(item, dict) and "image_url" in item:
            img_url = item["image_url"]
            if img_url and img_url.startswith("http"):
                all_images.append(img_url)
    
    # Build markdown sections
    md_lines = []
    
    # Header section
    md_lines.extend([
        f"# Crime Report Search Results",
        f"**Query:** `{query}`",
        f"**Search Time:** {DEFAULT_CONFIG['CURRENT_UTC']}",
        f"**Generated by:** {DEFAULT_CONFIG['CURRENT_USER']}",
        f"**Time Period:** Past {search_result.get('days', 7)} days\n"
    ])

    # Add fallback image if needed
    images_in_report = len(all_images) > 0
    if not images_in_report:
        print("No images found in search results.")
    else:
        md_lines.append("## Images Found\n")
        for img_url in all_images:
            md_lines.append(f"![Image]({img_url})\n")
    # Summary section
    md_lines.extend([
        f"## Summary / Answer\n{answer}\n",
        "## Detailed Results\n"
    ])

    # Process search items
    if not items:
        md_lines.append("*No search items found.*")
    else:
        for i, item in enumerate(items, start=1):
            title = item.get("title", "Untitled")
            url = item.get("url", "#")
            snippet = item.get("content", "No snippet.")
            
            # Track link
            all_links.append({
                "title": title,
                "url": url,
                "source": item.get("source", "Unknown"),
                "published_date": item.get("published_date", "Unknown")
            })
            
            # Add item content
            md_lines.append(f"**{i}.** [{title}]({url})")
            
            # Handle images
            if "image_url" in item and item["image_url"]:
                img_url = item["image_url"]
                if img_url and img_url.startswith("http"):
                    md_lines.append(f"\n![Image from {title}]({img_url})")
                    all_images.append(img_url)
            
            md_lines.append(f"\n> {snippet}\n")

            # Add extracted content if available
            if url in extracts_by_url:
                extracted_data = extracts_by_url[url]
                if isinstance(extracted_data, dict):
                    # Text content
                    text_content = extracted_data.get("text", "")
                    if len(text_content) > 500:
                        text_content = text_content[:500] + "..."
                    md_lines.append(f"**Extracted Content**:\n\n{text_content}\n")
                    
                    # Images from extraction
                    if "images" in extracted_data and extracted_data["images"]:
                        md_lines.append("**Images:**\n")
                        for img in extracted_data["images"][:3]:
                            if isinstance(img, dict) and "url" in img:
                                img_url = img["url"]
                                if img_url and img_url.startswith("http"):
                                    img_desc = img.get("description", "Image from article")
                                    md_lines.append(f"![{img_desc}]({img_url})\n")
                                    all_images.append(img_url)

    return json.dumps({
        "markdown_report": "\n".join(md_lines),
        "images": all_images,
        "links": all_links,
        "metadata": {
            "query": query,
            "timestamp": DEFAULT_CONFIG["CURRENT_UTC"],
            "user": DEFAULT_CONFIG["CURRENT_USER"],
            "result_count": len(items),
            "image_count": len(all_images),
            "link_count": len(all_links)
        }
    })

# ---------------------------------------------------------------------------
# 5. Main Function for Testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Test configuration
    test_query = "Recent crime incidents and statistics"
    test_config = {
        "selected_regions": ["Chicago", "New York"],
        "search_mode": "specific_range",
        "start_year": 2000,
        "end_year": 2005,
        "topic": "news",
        "max_results": 5
    }
    
    # Execute search
    search_response = tavily_search(
        query=test_query,
        **test_config
    )
    
    if not search_response:
        print("Search failed. Exiting.")
        exit()

    # Extract content from URLs
    urls = [item["url"] for item in search_response.get("results") 
            if "url" in item]
    extract_response = tavily_extract(urls=urls)
    
    # Build report
    result_json = build_markdown_report(test_query, search_response, extract_response)
    result_data = json.loads(result_json)

    # Save outputs
    with open("crime_report.md", "w", encoding="utf-8") as f:
        f.write(result_data["markdown_report"])

    with open("crime_report_data.json", "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)

    # Print summary
    print("\n=== Crime Report Generated ===")
    print(f"- Report saved to: crime_report.md")
    print(f"- Full data saved to: crime_report_data.json")
    print(f"- Found {result_data['metadata']['image_count']} images")
    print(f"- Found {result_data['metadata']['link_count']} links")
    for image in result_data["images"]:
        print(f"  - Image URL: {image}")
    print("\nDone!")