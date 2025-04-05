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
        # print(response.text)
        # print('\n')
        soup = BeautifulSoup(response.text, "html.parser")

        # Replace all <br> tags with a literal period + space
        for br in soup.find_all("br"):
            if br.parent:
                br.insert_after(". ")  # insert after instead of replace
                br.decompose()         # remove the <br> tag itself

        selected_tags = soup.find_all(string=True)

        if not selected_tags:
            logging.warning(f"No matching tags found in {url}")
            return ""

        # all_tags = soup.find_all(True)  # Find all tags, not just text nodes
        # END_PUNCTUATION = ('.', '!', '?', '…', '‽')

        # collected_text = []

        # for tag in all_tags:
        #     tag_name = tag.name
        #     print(f'tag name: {tag_name}')
        #     text = tag.get_text(separator=' ', strip=True)
        #     print(text)
        #     print('\n')
        #     # Add period if it's a list item or table cell and missing punctuation
        #     if tag_name in ['li']:
        #         if text and not text.endswith(END_PUNCTUATION):
        #             text += '. '
        #             # print(text)
        #     if text:
        #         # print(text)
        #         collected_text.append(text)
        # print(collected_text)
        text = soup.get_text(separator=' ', strip=True)
        text = " ".join(collected_text)
        text = text.replace('\xa0', '')
        ZERO_WIDTH_CHARACTERS = ['\u200b', '\u200c', '\u200d', '\u2060', '\ufeff']
        for zwc in ZERO_WIDTH_CHARACTERS:
            text = text.replace(zwc, '')
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