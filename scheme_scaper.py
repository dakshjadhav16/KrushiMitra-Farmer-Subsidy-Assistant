import requests
from bs4 import BeautifulSoup
import csv
import re
import os
import time
import random
import argparse
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scraper.log"),
        logging.StreamHandler()
    ]
)

class SchemeWebScraper:
    """
    A web scraper for extracting structured information from government scheme websites
    """
    
    def __init__(self, output_file="scheme_data.csv", max_workers=5, delay=1):
        """
        Initialize the scraper with configuration options
        """
        self.output_file = output_file
        self.max_workers = max_workers
        self.delay = delay
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
        
        # Target sections to extract
        self.target_sections = [
            "Details", 
            "Benefits", 
            "Eligibility", 
            "Application Process", 
            "Documents Required", 
            "Frequently Asked Questions"
        ]
        
        # Variations of section titles to handle different naming conventions
        self.section_variations = {
            "Details": ["details", "about the scheme", "about scheme", "scheme details", "overview", "description", "introduction"],
            "Benefits": ["benefits", "scheme benefits", "advantages", "key benefits", "what you get"],
            "Eligibility": ["eligibility", "who can apply", "eligibility criteria", "who is eligible"],
            "Application Process": ["application process", "how to apply", "application procedure", "process", "steps to apply"],
            "Documents Required": ["documents required", "required documents", "documents needed", "necessary documents"],
            "Frequently Asked Questions": ["frequently asked questions", "faqs", "faq", "common questions"]
        }
        
        # Initialize the CSV file with headers
        self._initialize_csv()
        
    def _initialize_csv(self):
        """Create the CSV file with headers"""
        with open(self.output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Add Scheme Name after URL
            headers = ["URL", "Scheme Name"] + self.target_sections
            writer.writerow(headers)
            
    def _get_scheme_name(self, soup, url):
        """
        Extract the scheme name from the page with special handling for myscheme.gov.in
        """
        logging.info(f"Extracting scheme name from {url}")
        
        # Special handling for myscheme.gov.in based on URL pattern
        parsed_url = urlparse(url)
        if "myscheme.gov.in" in parsed_url.netloc:
            # Extract scheme code from URL path (e.g., /schemes/kcc)
            path_parts = parsed_url.path.split('/')
            if len(path_parts) > 2 and path_parts[-2] == "schemes":
                scheme_code = path_parts[-1].upper()
                
                # Look for the scheme name in the page content
                if scheme_code == "KCC":
                    return "Kisan Credit Card Scheme"
                
                # Try to find the full name on the page
                # Look for headings that might contain the scheme code
                headings = soup.find_all(['h1', 'h2', 'h3', 'h4'])
                for heading in headings:
                    text = heading.get_text(strip=True)
                    if scheme_code in text:
                        return text
                
                # Look for specific sections that might contain the scheme name
                scheme_sections = soup.find_all(['div', 'section'], class_=lambda x: x and ('header' in x or 'title' in x or 'scheme' in x))
                for section in scheme_sections:
                    text = section.get_text(strip=True)
                    if scheme_code in text:
                        # Try to extract the full name
                        for line in text.split('\n'):
                            if scheme_code in line:
                                return line.strip()
                
                # If nothing found, construct a name from the code
                scheme_codes = {
                    "KCC": "Kisan Credit Card Scheme",
                    "PMKISAN": "PM-Kisan Samman Nidhi Yojana",
                    "PMFBY": "Pradhan Mantri Fasal Bima Yojana",
                    "PMJAY": "Pradhan Mantri Jan Arogya Yojana",
                    "PMJDY": "Pradhan Mantri Jan Dhan Yojana",
                    "PMMY": "Pradhan Mantri Mudra Yojana",
                    "PMKVY": "Pradhan Mantri Kaushal Vikas Yojana",
                    "PMAY": "Pradhan Mantri Awas Yojana",
                    # Add more scheme codes and names as needed
                }
                
                if scheme_code in scheme_codes:
                    return scheme_codes[scheme_code]
                return f"{scheme_code} Scheme"
        
        # General strategy for other websites
        # Strategy 1: Look for h1 tags
        h1_tags = soup.find_all('h1')
        if h1_tags:
            scheme_name = h1_tags[0].get_text(strip=True)
            logging.info(f"Found scheme name from h1: {scheme_name}")
            return scheme_name
            
        # Strategy 2: Look for title tags
        title_tag = soup.find('title')
        if title_tag:
            title_text = title_tag.get_text(strip=True)
            # Clean up title text (often contains site name)
            title_parts = title_text.split('|')
            if len(title_parts) > 1:
                scheme_name = title_parts[0].strip()
                logging.info(f"Found scheme name from title (split): {scheme_name}")
                return scheme_name
            
            # Try to extract scheme name from title using common patterns
            title_lower = title_text.lower()
            scheme_indicators = ['scheme', 'yojana', 'programme', 'program', 'initiative']
            
            for indicator in scheme_indicators:
                if indicator in title_lower:
                    # Try to get text before and including the indicator
                    pattern = re.compile(f'(.*?{indicator})', re.IGNORECASE)
                    match = pattern.search(title_text)
                    if match:
                        scheme_name = match.group(1).strip()
                        logging.info(f"Found scheme name from title (pattern): {scheme_name}")
                        return scheme_name
            
            # If no pattern matched, use the full title
            logging.info(f"Found scheme name from title (full): {title_text}")
            return title_text
        
        # Try extracting from URL path as fallback
        path_parts = parsed_url.path.split('/')
        if len(path_parts) > 0:
            scheme_part = path_parts[-1] if path_parts[-1] else (path_parts[-2] if len(path_parts) > 1 else '')
            if scheme_part:
                scheme_name = scheme_part.replace('-', ' ').replace('_', ' ').replace('.html', '').replace('.php', '').title()
                logging.info(f"Found scheme name from URL: {scheme_name}")
                return scheme_name
        
        # Extreme fallback - use domain and path
        domain = parsed_url.netloc
        path = parsed_url.path
        return f"Scheme from {domain}{path}"
    
    def _find_section_content(self, soup, section_name):
        """Find and extract the content of a specific section"""
        # Get variations of the section name for flexible matching
        variations = self.section_variations.get(section_name, [section_name.lower()])
        
        # Strategy 1: Look for headings with the section name
        for heading_tag in ['h2', 'h3', 'h4', 'h5', 'h6', 'h1']:
            headings = soup.find_all(heading_tag)
            
            for heading in headings:
                heading_text = heading.get_text(strip=True).lower()
                
                # Check if the heading matches any of our variations
                if any(variation in heading_text for variation in variations):
                    # Found a matching heading, now extract content
                    content = []
                    
                    # Get the next siblings until next heading or end of parent
                    for sibling in heading.find_next_siblings():
                        if sibling.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                            break
                        if sibling.name in ['p', 'ul', 'ol', 'div']:
                            content.append(sibling.get_text(strip=True))
                    
                    # Join all the content pieces
                    if content:
                        return " ".join(content)
        
        # Strategy 2: Look for div/section with ID or class containing section name
        for variation in variations:
            # Try with ID
            section_div = soup.find(lambda tag: tag.has_attr('id') and variation in tag.get('id', '').lower())
            if section_div:
                return section_div.get_text(strip=True)
                
            # Try with class
            section_div = soup.find(lambda tag: tag.has_attr('class') and 
                                   any(variation in cls.lower() for cls in tag.get('class', [])))
            if section_div:
                return section_div.get_text(strip=True)
        
        # Strategy 3: Look for sections by tab panel structure (common in government sites)
        tab_panels = soup.find_all(class_=lambda cls: cls and ('tab-pane' in cls or 'tab-content' in cls))
        for panel in tab_panels:
            panel_text = panel.get_text(strip=True).lower()
            if any(variation in panel_text for variation in variations):
                return panel.get_text(strip=True)
        
        # Not found
        return ""
    
    def scrape_scheme_page(self, url):
        """Scrape a scheme page and extract the target sections"""
        try:
            # Add a small random delay to avoid rate limiting
            sleep_time = self.delay + random.uniform(0.1, 1.0)
            time.sleep(sleep_time)
            
            logging.info(f"Scraping URL: {url}")
            
            # Send the request
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            # Parse the HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract scheme name and guarantee it's not empty
            scheme_name = self._get_scheme_name(soup, url)
            logging.info(f"Found scheme: {scheme_name}")
            
            # Initialize result dictionary
            result = {
                "URL": url,
                "Scheme Name": scheme_name or "Unknown Scheme"  # Ensure not empty
            }
            
            # Extract each target section
            for section in self.target_sections:
                content = self._find_section_content(soup, section)
                result[section] = content
                logging.debug(f"Section '{section}' extracted: {len(content)} characters")
            
            # Debug print the final result
            logging.info(f"Final scheme name: {result['Scheme Name']}")
            return result
            
        except Exception as e:
            logging.error(f"Error scraping {url}: {str(e)}")
            # Return partial data if available with a non-empty scheme name
            return {
                "URL": url,
                "Scheme Name": "Error: Could not extract scheme name",
                **{section: "" for section in self.target_sections}
            }
    
    def scrape_multiple_urls(self, urls):
        """Scrape multiple URLs in parallel"""
        results = []
        
        logging.info(f"Starting to scrape {len(urls)} URLs")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all scraping tasks
            future_to_url = {executor.submit(self.scrape_scheme_page, url): url for url in urls}
            
            # Process results as they complete
            for future in future_to_url:
                try:
                    result = future.result()
                    if result:
                        # Ensure scheme name is not empty
                        if not result.get("Scheme Name"):
                            url = future_to_url[future]
                            result["Scheme Name"] = f"Scheme from {urlparse(url).netloc}"
                        
                        results.append(result)
                        # Debug print to verify scheme name is being captured
                        logging.info(f"Successfully scraped: {result['Scheme Name']} from {result['URL']}")
                except Exception as e:
                    url = future_to_url[future]
                    logging.error(f"Failed to scrape {url}: {str(e)}")
        
        logging.info(f"Completed scraping {len(results)} URLs successfully")
        return results
    
    def write_results_to_csv(self, results):
        """Write scraped results to CSV file"""
        if not results:
            logging.warning("No results to write to CSV")
            return
            
        headers = ["URL", "Scheme Name"] + self.target_sections
        
        try:
            # Write to CSV file
            with open(self.output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                
                # Debug before writing
                for result in results:
                    # Ensure Scheme Name is not empty
                    if not result.get("Scheme Name"):
                        result["Scheme Name"] = "Unknown Scheme"
                    
                    logging.info(f"Writing to CSV: {result['Scheme Name']} - {result['URL']}")
                    writer.writerow(result)
                    
            logging.info(f"Results written to {self.output_file}")
            
            # Verify CSV content after writing
            self._verify_csv_output()
            
        except Exception as e:
            logging.error(f"Error writing to CSV: {str(e)}")
            
            # Emergency dump to a different file
            try:
                emergency_file = f"emergency_dump_{int(time.time())}.csv"
                with open(emergency_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writeheader()
                    for result in results:
                        if not result.get("Scheme Name"):
                            result["Scheme Name"] = "Unknown Scheme"
                        writer.writerow(result)
                logging.info(f"Emergency dump created at {emergency_file}")
            except Exception as e2:
                logging.error(f"Failed to create emergency dump: {str(e2)}")
    
    def _verify_csv_output(self):
        """Verify that the CSV file contains the expected data"""
        try:
            with open(self.output_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                row_count = 0
                empty_scheme_names = 0
                
                for row in reader:
                    row_count += 1
                    if not row.get("Scheme Name"):
                        empty_scheme_names += 1
                        
                logging.info(f"CSV verification: {row_count} rows, {empty_scheme_names} empty scheme names")
                
                if empty_scheme_names > 0:
                    logging.warning(f"Found {empty_scheme_names} rows with empty scheme names!")
                    
        except Exception as e:
            logging.error(f"Error verifying CSV: {str(e)}")
    
    def run(self, urls):
        """Main method to run the scraper on a list of URLs"""
        logging.info(f"Starting scraper for {len(urls)} URLs")
        
        # Scrape the URLs
        results = self.scrape_multiple_urls(urls)
        
        # Write to CSV
        self.write_results_to_csv(results)
        
        logging.info(f"Scraping completed. Results saved to {self.output_file}")
        return results

# Define a specific mapping of scheme codes to names for myscheme.gov.in
SCHEME_CODE_MAPPING = {
    "kcc": "Kisan Credit Card Scheme",
    "pmkisan": "PM-Kisan Samman Nidhi Yojana",
    "pmfby": "Pradhan Mantri Fasal Bima Yojana",
    "pmjay": "Pradhan Mantri Jan Arogya Yojana",
    "pmjdy": "Pradhan Mantri Jan Dhan Yojana",
    "pmmy": "Pradhan Mantri Mudra Yojana",
    "pmkvy": "Pradhan Mantri Kaushal Vikas Yojana",
    "pmay": "Pradhan Mantri Awas Yojana",
    # Add more scheme codes and names as needed
}

# Function to directly fix existing CSV file
def fix_existing_csv(input_file, output_file=None):
    """
    Fix an existing CSV file by adding scheme names
    
    Parameters:
    -----------
    input_file : str
        Path to the input CSV file
    output_file : str, optional
        Path to the output CSV file. If None, will overwrite the input file.
    """
    if output_file is None:
        output_file = input_file + ".fixed.csv"
        
    logging.info(f"Fixing CSV file: {input_file} -> {output_file}")
    
    try:
        # Read input CSV
        with open(input_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            rows = list(reader)
            
        # Fix scheme names
        for row in rows:
            url = row.get("URL", "")
            if not row.get("Scheme Name") and url:
                parsed_url = urlparse(url)
                path_parts = parsed_url.path.split('/')
                
                # For myscheme.gov.in
                if "myscheme.gov.in" in parsed_url.netloc and len(path_parts) > 2 and path_parts[-2] == "schemes":
                    scheme_code = path_parts[-1].lower()
                    if scheme_code in SCHEME_CODE_MAPPING:
                        row["Scheme Name"] = SCHEME_CODE_MAPPING[scheme_code]
                    else:
                        row["Scheme Name"] = f"{scheme_code.upper()} Scheme"
                else:
                    # Generic extraction from URL
                    scheme_part = path_parts[-1] if path_parts[-1] else (path_parts[-2] if len(path_parts) > 1 else '')
                    if scheme_part:
                        row["Scheme Name"] = scheme_part.replace('-', ' ').replace('_', ' ').replace('.html', '').replace('.php', '').title()
                    else:
                        row["Scheme Name"] = f"Scheme from {parsed_url.netloc}"
        
        # Write output CSV
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
            
        logging.info(f"Fixed CSV saved to {output_file}")
        return True
    except Exception as e:
        logging.error(f"Error fixing CSV: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape government scheme websites for structured information")
    parser.add_argument("--input", "-i", help="Input file with URLs (one per line)", type=str)
    parser.add_argument("--output", "-o", help="Output CSV file", default="scheme_data.csv")
    parser.add_argument("--workers", "-w", help="Number of parallel workers", type=int, default=5)
    parser.add_argument("--delay", "-d", help="Delay between requests (seconds)", type=float, default=1.0)
    parser.add_argument("--fix-csv", help="Fix existing CSV file by adding scheme names", type=str)
    
    args = parser.parse_args()
    
    # Fix existing CSV if requested
    if args.fix_csv:
        fix_existing_csv(args.fix_csv, args.output)
        exit(0)
    
    if args.input:
        # Read URLs from file
        try:
            with open(args.input, 'r') as f:
                urls = [line.strip() for line in f if line.strip()]
            logging.info(f"Loaded {len(urls)} URLs from {args.input}")
        except Exception as e:
            logging.error(f"Error reading input file: {str(e)}")
            urls = []
    else:
        # Example URL for testing
        urls = [
    "https://www.myscheme.gov.in/schemes/kcc",
    "https://www.myscheme.gov.in/schemes/iaris",
    "https://www.myscheme.gov.in/schemes/acandabc",
    "https://www.myscheme.gov.in/schemes/anby",
    "https://www.myscheme.gov.in/schemes/ncrfs",
    "https://www.myscheme.gov.in/schemes/fapllf",
    "https://www.myscheme.gov.in/schemes/spfrpcvwnmpcn-faascdmf",
    "https://www.myscheme.gov.in/schemes/sspftws",
    "https://www.myscheme.gov.in/schemes/anky",
    "https://www.myscheme.gov.in/schemes/ysrjk",
    "https://www.myscheme.gov.in/schemes/mmksy",
    "https://www.myscheme.gov.in/schemes/zbnf",
    "https://www.myscheme.gov.in/schemes/ssyc",
    "https://www.myscheme.gov.in/schemes/ssst-pts",
    "https://www.myscheme.gov.in/schemes/km",
    "https://www.myscheme.gov.in/schemes/fasuccu",
    "https://www.myscheme.gov.in/schemes/kcc",
    "https://www.myscheme.gov.in/schemes/isds",
    "https://www.myscheme.gov.in/schemes/pmgsy",
    "https://www.myscheme.gov.in/schemes/fapfpfoobm",
    "https://www.myscheme.gov.in/schemes/vds",
    "https://www.myscheme.gov.in/schemes/hmneh",
    "https://www.myscheme.gov.in/schemes/bhausaheb-fundkar-horticulture-plantataion-scheme",
    "https://www.myscheme.gov.in/schemes/kt",
    "https://www.myscheme.gov.in/schemes/abn",
    "https://www.myscheme.gov.in/schemes/fmtaspsstc",
    "https://www.myscheme.gov.in/schemes/scdicmd",
    "https://www.myscheme.gov.in/schemes/seh-tmdummapuy",
    "https://www.myscheme.gov.in/schemes/ntkjky",
    "https://www.myscheme.gov.in/schemes/fp",
    "https://www.myscheme.gov.in/schemes/fig",
    "https://www.myscheme.gov.in/schemes/pcwfstfpi",
    "https://www.myscheme.gov.in/schemes/ahtn",
    "https://www.myscheme.gov.in/schemes/omps",
    "https://www.myscheme.gov.in/schemes/gopinath-munde-shetkari-apghat-suraksha-sanugrah-audhan-yojana",
    "https://www.myscheme.gov.in/schemes/chief-minister-agriculture-and-food-processing-scheme",
    "https://www.myscheme.gov.in/schemes/pocra",
    "https://www.myscheme.gov.in/schemes/namo-shetkari-mahasanman-nidhi-yojana",
    "https://www.myscheme.gov.in/schemes/kvdgssy",
    "https://www.myscheme.gov.in/schemes/chief-minister-sustainable-agriculture-irrigation-scheme",
    "https://www.myscheme.gov.in/schemes/dsctaecc",
    "https://www.myscheme.gov.in/schemes/nbm",
    "https://www.myscheme.gov.in/schemes/sagy",
    "https://www.myscheme.gov.in/schemes/eebdsrs",
    "https://www.myscheme.gov.in/schemes/pmmsy",
    "https://www.myscheme.gov.in/schemes/iaris",
    "https://www.myscheme.gov.in/schemes/cpis",
    "https://www.myscheme.gov.in/schemes/sr",
    "https://www.myscheme.gov.in/schemes/nbhm",
    "https://www.myscheme.gov.in/schemes/pmksypdmc",
    "https://www.myscheme.gov.in/schemes/rgsa",
    "https://www.myscheme.gov.in/schemes/ppe",
    "https://www.myscheme.gov.in/schemes/pm-kisan",
    "https://www.myscheme.gov.in/schemes/nswf",
    "https://www.myscheme.gov.in/schemes/pmfby",
    "https://www.myscheme.gov.in/schemes/iepg",
    "https://www.myscheme.gov.in/schemes/pkvy",
    "https://www.myscheme.gov.in/schemes/aif",
    "https://www.myscheme.gov.in/schemes/ksis",
    "https://www.myscheme.gov.in/schemes/smbdlp",
    "https://www.myscheme.gov.in/schemes/pcardah",
    "https://www.myscheme.gov.in/schemes/pcardsia",
    "https://www.myscheme.gov.in/schemes/pcardbif",
    "https://www.myscheme.gov.in/schemes/akckvy",
    "https://www.myscheme.gov.in/schemes/cmbkyii",
    "https://www.myscheme.gov.in/schemes/pby",
    "https://www.myscheme.gov.in/schemes/agfst",
    "https://www.myscheme.gov.in/schemes/caicoloabfabfa",
    "https://www.myscheme.gov.in/schemes/pmmsy",
    "https://www.myscheme.gov.in/schemes/fsss",
    "https://www.myscheme.gov.in/schemes/ciad",
    "https://www.myscheme.gov.in/schemes/eotmaf",
    "https://www.myscheme.gov.in/schemes/bpuk",
    "https://www.myscheme.gov.in/schemes/kgsuk",
    "https://www.myscheme.gov.in/schemes/ssbutsc",
    "https://www.myscheme.gov.in/schemes/gspaa-dfwasaofbc",
    "https://www.myscheme.gov.in/schemes/sseg10scp",
    "https://www.myscheme.gov.in/schemes/mkusy",
    "https://www.myscheme.gov.in/schemes/spwfrpsmb-faascdmf",
    "https://www.myscheme.gov.in/schemes/fds",
    "https://www.myscheme.gov.in/schemes/gopinath-munde-shetkari-apghat-suraksha-sanugrah-audhan-yojana",
    "https://www.myscheme.gov.in/schemes/ky",
    "https://www.myscheme.gov.in/schemes/nfbks",
    "https://www.myscheme.gov.in/schemes/wit",
    "https://www.myscheme.gov.in/schemes/speoepsgug",
    "https://www.myscheme.gov.in/schemes/mbydfvy",
    "https://www.myscheme.gov.in/schemes/kuy-movcdner",
    "https://www.myscheme.gov.in/schemes/emadu",
    "https://www.myscheme.gov.in/schemes/gks",
    "https://www.myscheme.gov.in/schemes/grama-jyothi",
    "https://www.myscheme.gov.in/schemes/dmochn",
    "https://www.myscheme.gov.in/schemes/mkuy",
    "https://www.myscheme.gov.in/schemes/sebpu",
    "https://www.myscheme.gov.in/schemes/mpy-mcpnpky",
    "https://www.myscheme.gov.in/schemes/gstn",
    "https://www.myscheme.gov.in/schemes/npvy"
]

        logging.info("Using default test URLs")
    
    # Initialize and run the scraper
    scraper = SchemeWebScraper(output_file=args.output, max_workers=args.workers, delay=args.delay)
    scraper.run(urls)