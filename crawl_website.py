import requests
import re
from bs4 import BeautifulSoup
import logging
import io

def download_and_clean_html(url):
    """ Reads an HTML file, extracts text, and cleans it for indexing. """

    print("Fetching text from url ...")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.exceptions.HTTPError as http_err:
        return ""
    except requests.exceptions.RequestException as req_err:
        return ""
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")

        selected_tags = soup.find_all(string=True)

        if not selected_tags:
            logging.warning(f"No matching tags found in {url}")
            return ""
        
        # text = " ".join(tag.get_text(separator=" ") for tag in selected_tags if tag.get_text())
        text = soup.get_text(separator=' ', strip=True)
        # text = " ".join(soup.stripped_strings)


        text = text.replace('\xa0', '')
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        text = re.sub(r'\(\s+', '(', text)
        text = re.sub(r'\s+\)', ')', text)
        text = re.sub(r'\[\s+', '[', text)    # remove space after opening square bracket
        text = re.sub(r'\s+\]', ']', text)    # remove space before closing square bracket


        lines = text.splitlines()
        non_blank_lines = [line.strip() for line in lines if line.strip()]
        text = "\n".join(non_blank_lines)

        if len(text) > 10000:
            print(f"Trimming webpage content from {len(text)} to 10000 characters")
            text = text[:10000]
              
        return text
    
    else:
        return ""