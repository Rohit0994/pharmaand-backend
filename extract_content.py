import os
import json
from pathlib import Path
from bs4 import BeautifulSoup

# Path to the Pharmaand website folder
# When running in Docker, html files are in the 'website' subfolder
# When running locally, use the full path
import sys
if sys.platform == "win32":
    WEBSITE_PATH = r"C:\Users\RohitSharma\OneDrive - pharmaand GmbH\Rohit\AI Work\Pharmaand Website\Pharmaand"
else:
    WEBSITE_PATH = "website"
OUTPUT_FILE = "content/data.json"
SITE_BASE_URL = "https://rohit0994.github.io/pharmaand_GmBH"

def extract_html_content():
    """Extract text content, title, and URL from all HTML files in the website folder."""
    content_dict = {}
    
    # Create output directory if it doesn't exist
    os.makedirs("content", exist_ok=True)
    
    # Find all HTML files
    html_files = list(Path(WEBSITE_PATH).glob("*.html"))
    
    if not html_files:
        print(f"❌ No HTML files found in {WEBSITE_PATH}")
        return
    
    print(f"📄 Found {len(html_files)} HTML files")
    
    for html_file in html_files:
        try:
            with open(html_file, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f.read(), "html.parser")
            
            # Remove script, style, nav, header, footer
            for tag in soup(["script", "style", "nav", "header", "footer"]):
                tag.decompose()
            
            # Extract title
            title = soup.title.string if soup.title else html_file.stem
            title = title.strip() if title else html_file.stem
            
            # Get text and clean it
            text = soup.get_text(separator=" ", strip=True)
            text = " ".join(text.split())  # Normalize whitespace
            
            # Build full URL
            url = f"{SITE_BASE_URL}/{html_file.name}"
            
            content_dict[html_file.stem] = {
                "title": title,
                "url": url,
                "content": text
            }
            print(f"✓ {html_file.name}: {len(text)} characters extracted")
        
        except Exception as e:
            print(f"✗ Error processing {html_file.name}: {e}")
    
    # Save to JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(content_dict, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Content extracted and saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    extract_html_content()
