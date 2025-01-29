# ODJFS Licensing Reports Webscraper

## Overview

This project aims to compile a dataset of Ohio Department of Jobs and Family Services (ODJFS) licensing reports for childcare centers. We focus primarily on non-compliance data (e.g., which rules were found to be broken during the licensing inspection) but also compile information on each center (e.g., center capacity and age group ratios). The non-compliance data is primarily web-scraped from the ODJFS portal for ease and reliability, including:
- Center ID
- Non-compliance findings
- Non-compliance code
- Non-compliance Webpage Link

Much of the center-specific data is stored in PDF reports available on the ODJFS webpages. These reports lack embedded text, so we use the EasyOCR library to extract the required fields from each PDF to construct the center dataset. 

### Technical Approach

- **Web Scraping:** Extracts non-compliance data efficiently in about 40 minutes using 10 parallel processing cores.
- **PDF Extraction:** Uses EasyOCR to process scanned PDFs, which is currently slow (taking 2-3 days with 2 parallel processes due to VRAM limitations). This can be improved by utilizing machines with more VRAM.
- **Optimization Efforts:** Work is ongoing to enhance PDF processing efficiency (e.g., adjust image processing to lower DPI).

## Features

- **Automated Data Extraction:** Web scraping for non-compliance records and OCR-based PDF parsing.
- **Structured Data Compilation:** Aggregates licensing findings with center-specific details.
- **Scalability Improvements:** Exploring faster PDF processing through hardware optimizations.

## Installation (NOT UP TO DATE. THERE ARE ADDITIONAL REQUIREMENTS)

To set up the project locally, follow these steps:

### Prerequisites
Ensure you have the following installed:
- Python (>=3.8)
- pip
- Virtual Environment (optional but recommended)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repository/ODJFS-Licensing-Report-Webscraper.git
   cd ODJFS-Licensing-Report-Webscraper
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the project:
   ```bash
   python main.py
   ```
   
## Folder Structure

```
ODJFS-Licensing-Report-Webscraper/
│-- data/               # Webscraped data and final results (Last scraped on January 20th, 2025)
│-- text_extraction.py  # Code for processing PDFs and extracting text fields
│-- test_scraper.ipynb  # Jupyter notebook for testing text extraction before scaling to the entire dataset
│-- webscraping.py      # Code for extracting non-compliance, PDF, and center data from the ODJFS website
│-- requirements.txt    # Python dependencies
│-- main.py             # Main execution script (runs both web scraping and PDF extraction). Currently still in progress.
│-- README.md           # Project documentation
```

