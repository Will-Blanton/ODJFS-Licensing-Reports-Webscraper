import argparse
import os
import logging
import multiprocessing as mp
import pandas as pd
import re
import requests
import time

from bs4 import BeautifulSoup
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import TypedDict
from urllib.parse import urlparse, parse_qs

from tqdm import tqdm

# Link Constants
ODCY_LINK = "https://childcaresearch.ohio.gov/search?q=fVLNbhMxEN40v0uapgKhHhAiBy6VQkUR1xwWN1VDIVl1V0gFcXDWk42FY6%2b83pS98Q6ICxdeg1fgyBvwJjB2uhCJqrPS2DOf59tvPPZqnuf9RrOrtd0ddCRhyRXMiVqtlBwO3oDOuZKj46On9hsOSCFMoWEkoTCaiuEgLOaCJ%2bdQxuoDyJEshGhaxkfbREcn8ewoAqqTJdHcgOb0MZ7ZC7VacwZ6WqzmoFtEFdKUDcJN2X7LM6IY7FZH4jKD1gU1XKbdkGrDqZjSFbQnckGlyXuXWJvGijGBknszYUuuIz%2fUkCdLpYQfuSVIoX9Cy3y2mGWgkVPJ%2fTNV6O1E8zVQke%2b%2fgIXSsCkjVEN3vAaJGuy%2bE13x1QqD%2b1EGiRUEwHKy5IJZeC%2fWVOaZ0sYR9oMFNv6PqTdbg5Y8XRob3TnlIFiseZYfjKkWpaPBs2z8MUMaJPDPgLLIYO%2f3Qs3X1MA5l9hmihmQB5tBiPK0wCT7q%2bLuK07nXOCVTmReoKIEmtNgfEma0zEJA9wTEqI%2fJaROZkEjINEEB5FKThsThLxaDce5eSPthnejNTudmrOH%2fw3dXesFlSm8e2%2ffV%2b1Gs8jOk%2bNW3bL59cpVL9MFm51V4DdvloHWt3h7q7TtztrIIo0Htwi0IvyWdZbA71jn2%2b5s7rZCp9cNJub4IseS2bUTgYDEAHM8rkUv%2b%2frymV1%2fPf%2fCnC77p9Z25vvn4JtDOhVynfl0%2bPPwh0P8CqmsYu3%2bAQ%3d%3d"
REL_PATH = "https://childcaresearch.ohio.gov/"

# string constants
ANNUAL = "ANNUAL"
PROGRAM_NAME = "program_name"
PROGRAM_ID = "program_id"

# regex patterns
# FINDINGS_PATTERN = r"(?:number\(s\)?|number|numbers)\s*(.*?)(?=\s*below)"
FINDINGS_PATTERN = r"(?:number\(s\)?|number|numbers)\s*(.*?)\s*(below)"


class PageStats(TypedDict):
    rows_total: int
    centers_kept: int
    filtered_no_program_name_col: int
    filtered_no_program_link: int
    filtered_program_page_missing: int
    filtered_no_inspection_link: int
    filtered_no_annual_pdf: int
    filtered_bad_program_id: int
    row_errors: int
    centers_with_nc_link: int
    centers_without_nc_link: int
    centers_with_nc_rows: int
    centers_with_empty_nc: int
    nc_rows: int


def empty_non_compliance_df() -> pd.DataFrame:
    """
    Create an empty non-compliance dataframe with consistent columns.
    """
    return pd.DataFrame(columns=["rule", "code", "compliance", "findings", "occurrence"])


def extract_html(url: str, retries: int = 3, timeout: int = 30) -> BeautifulSoup | None:
    """
    Parse the html at the given url into a beautiful soup object, for manipulation.

    :param url: Any valid URL
    :param retries: number of attempts to retrieve the page
    :param timeout: request timeout in seconds
    :return: the parsed HTML or None if the request failed
    """
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, timeout=timeout)

            # check if the request was successful (status code 200)
            if response.status_code == 200:

                # parse the HTML content
                soup = BeautifulSoup(response.text, 'html.parser')
                return soup

            logging.warning(
                f"Failed to retrieve page from {url}, status code: {response.status_code} "
                f"(attempt {attempt}/{retries})"
            )

        except requests.RequestException as e:
            logging.warning(f"Request failed for {url} on attempt {attempt}/{retries}: {e}")

        if attempt < retries:
            time.sleep(min(attempt, 3))

    logging.error(f"Giving up on {url} after {retries} attempts")
    return None


def extract_inspection(program_page: BeautifulSoup, rel_path: str) -> str | None:
    """
    Extract the inspection link from the childcare center page

    :param program_page: the child care center page's html
    :param rel_path: relative path to append to the inspection link
    :return: the inspection page link
    """
    inspection_url = None

    inspection_button_span = program_page.find('span', class_='inspectionsButton')

    if inspection_button_span:
        inspection_link_tag = inspection_button_span.find_parent('a')

        if inspection_link_tag and 'href' in inspection_link_tag.attrs:
            inspection_url = rel_path + inspection_link_tag['href']

    return inspection_url


def extract_accreditations(program_page: BeautifulSoup) -> dict[str, bool]:
    """
    Extract the center's accreditations (e.g., NAEYC, NECPA) 

    :param program_page: the child care center page's html
    :return: a dictionary with each accreditation as the key and True or False to indicate presence as the value
    """

    accreditations = {
        "NAEYC": False,
        "NECPA": False,
        "NACCP": False,
        "NAFCC": False
    }

    spans = program_page.find_all("span", class_="checkBoxIcon green")

    pattern = re.compile(r"schk(.*?)_checkBox")

    for span in spans:
        span_id = span.get("id", "")
        match = pattern.search(span_id)
        if match:
            extracted_item = match.group(1)

            if extracted_item in accreditations:

                accreditations[extracted_item] = True

    # acr_df = pd.DataFrame(accreditations, [program_name])

    return accreditations


def extract_center_links(url: str, rel_path: str) -> tuple[str | None, str | None]:
    """
    Extract the pdf link from the inspection page
    :param url: inspection url
    :param rel_path: relative path to append to the pdf link
    :return: the pdf link, the non-compliance link (None, None) if not found
    """

    inspection_page = extract_html(url)
    most_recent_pdf_link = None
    most_recent_nc_link = None
    most_recent_date = None

    if inspection_page is not None:
        rows = inspection_page.find_all('div', class_='resultsListRow')

        for row in rows:
            columns = list(row.find_all('div', class_='resultsListColumn'))
            if len(columns) < 2:
                continue

            date_column = columns[0]

            # get the inspection type
            inspect_type_parts = list(columns[1].stripped_strings)
            if not inspect_type_parts:
                continue
            inspect_type = inspect_type_parts[1] if len(inspect_type_parts) > 1 else inspect_type_parts[0]

            pdf_col = row.find('span', class_='inspectionPDFlink')
            if pdf_col is None:
                continue
            pdf_link_tag = pdf_col.find('a', href=True)

            if pdf_link_tag and date_column:

                date_column = list(date_column.stripped_strings)
                if len(date_column) < 2:
                    continue

                # format into a datetime object for date comparisons
                inspection_date = date_column[1]
                try:
                    inspection_date = datetime.strptime(inspection_date, "%m/%d/%Y")
                except ValueError:
                    continue
                pdf_link = pdf_link_tag['href']

                # only save the most recent date (may not be necessary, since all appear to be listed in order
                if inspect_type == ANNUAL and (most_recent_date is None or inspection_date > most_recent_date):
                    most_recent_date = inspection_date
                    most_recent_pdf_link = rel_path + pdf_link

                    # extract the non-compliance link
                    nc_link = row.find('span', class_='inspectionNonCompliantLink')
                    nc_link = nc_link.find('a', href=True) if nc_link is not None else None
                    if nc_link is not None:
                        most_recent_nc_link = rel_path + nc_link['href']
                    else:
                        most_recent_nc_link = None

    return most_recent_pdf_link, most_recent_nc_link


def extract_numbers_with_letters(input_text: str) -> list[str]:
    """
    Extract numbers and their optional associated letters (e.g., '5 (a &b)' -> ['5a', '5b']).
    Save as strings like '4a', '5', etc.

    Args:
        input_text (str): The text to process.

    Returns:
        list: A list of strings representing numbers and their associated letters.
    """

    # match all numbers and their associated letters
    matches = re.findall(r"(\d+[A-Za-z]?)(?:\s*\(([^()]*?)\))?", input_text)

    findings = []
    for m in matches:
        number = m[0]

        letters = re.findall(r"\b\w+\b", m[1])

        # only add the letter if it is a single character
        if letters and all(len(l) == 1 for l in letters):
            for letter in letters:
                findings.append(f"{number}{letter}")
        else:
            findings.append(number)

    return findings


def extract_non_compliance(nc_link: str, rel_path: str) -> pd.DataFrame:
    """
    Extract non-compliance information from the non-compliance page

    :param nc_link: the non-compliance page link
    :param rel_path: relative path to append to the pdf link
    :return: a dataframe containing the non-compliance information
    """

    try:
        rules: list[pd.DataFrame] = []

        nc_page = extract_html(nc_link)
        if nc_page is None:
            return empty_non_compliance_df()

        results_list = nc_page.find('div', class_='resultsList')
        if not results_list:
            return empty_non_compliance_df()

        rows = results_list.find_all('div', class_='resultsListRow')

        for row in rows:
            rule_columns = list(row.find_all('div', class_='resultsListColumn'))
            if len(rule_columns) < 2:
                continue

            description = rule_columns[1].find('a')
            if description is None:
                continue

            rule = description.get_text(strip=True)

            # indicate no findings but the rule still non-compliant
            findings: list[str] = ['-1']
            code = "Missing NC link"

            if "href" in description.attrs:
                description_link = rel_path + description['href']

                rule_page = extract_html(description_link)

                if rule_page:
                    # the misspelling is intentional... it is in the HTML
                    finding_txt = rule_page.find('span', class_='inspectionFindsingsText')
                    if finding_txt is not None:
                        finding_txt = finding_txt.get_text(strip=True)
                        finding_match = re.search(FINDINGS_PATTERN, finding_txt)
                        if finding_match:
                            findings = extract_numbers_with_letters(finding_match.group(1))

                            # get the code and remove : from the beginning
                            code_idx = finding_match.end()
                            code = finding_txt[code_idx:]
                            code = code[code.find("1"):]
                        else:
                            code = finding_txt

            rules.append(
                pd.DataFrame.from_dict(
                    {
                        "rule": [rule],
                        "code": [code],
                        # "domain": [None],
                        "compliance": [None],
                        "findings": [findings],
                    }
                )
            )

        if not rules:
            return empty_non_compliance_df()

        nc_df = pd.concat(rules, axis=0, ignore_index=True)
        nc_df['occurrence'] = nc_df.groupby(by="rule").cumcount()
        return nc_df

    except Exception as e:
        logging.error(f"Error in extract_non_compliance: {e}")
        logging.error(f"Link: {nc_link}")
        return empty_non_compliance_df()


def extract_all_non_compliances(nc_link: str, rel_path: str) -> pd.DataFrame:
    """
    Extract non-compliances across all paginated pages.

    :param nc_link: Base link to the first non-compliance page
    :param rel_path: Relative path to prepend to discovered links
    :return: DataFrame combining non-compliance info from all pages
    """

    # kinda scuffed that we open the first page twice, but it's fine
    nc_df = extract_non_compliance(nc_link, rel_path)

    # process the rest of the pages
    try:
        first_page = extract_html(nc_link)
        if first_page:

            pager_span = first_page.find('span', id='ContentPlaceHolder1_pagerInspectionDetails')
            if pager_span:

                # should be fine since there shouldn't be more than 3 pages of non-compliances
                page_numbers = pager_span.find('span', class_='PageNumbers')

                if page_numbers:
                    all_dfs: list[pd.DataFrame] = [nc_df] if not nc_df.empty else []

                    # leave out the first page, as we already have it
                    page_links: set[str] = set()

                    anchors = page_numbers.find_all('a', class_='linkToPage', href=True)

                    # If there's any anchor, we handle multi-page
                    # e.g. "2", "3", ...
                    # don't include the first page, since it is not a link (<a ...>
                    for a in anchors:
                        href = a['href']
                        page_link = rel_path + href
                        if page_link not in page_links:
                            page_links.add(page_link)

                            df_page = extract_non_compliance(page_link, rel_path)
                            if not df_page.empty:
                                all_dfs.append(df_page)

                    if all_dfs:
                        nc_df = pd.concat(all_dfs, axis=0, ignore_index=True)
                    else:
                        nc_df = empty_non_compliance_df()
    except Exception as e:
        logging.error(f"Error in extract_all_non_compliances: {e}")
    finally:
        return nc_df


def extract_all_centers(url: str, rel_path: str, start_page: int = 0, end_page: int = -1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract all pdf links and associated center info (e.g., name and address info) into a dataFrame for further parsing.

    Also, non-compliance information is extracted into a separate dataframe for each center.

    :param url: The Ohio childcaresearch website URL (https://childcaresearch.ohio.gov/search for licensed childcare)
    :param rel_path: The relative path to append to the pdf links (e.g., https://childcaresearch.ohio.gov)
    :param start_page: The starting page to process
    :param end_page: The maximum number of pages to process (gets all by default)
    :return: a dataframe containing the center name, address info, and link to the pdf for the most recent center licensing inspection and a dataframe containing the non-compliance information
    """

    # NOTE: could probably automate rel_path extraction from the url

    if end_page < 0: 
        end_page = get_last_page(url)

    pdf_urls: list[pd.DataFrame] = []
    non_compliance_dfs: list[pd.DataFrame] = []
    main_page = None
    page_num = start_page

    """
    This uses a given start and end page to limit the number of pages to process
    
    The 'better' way would probably use a while loop with the "NextLast" tag to determine when to stop. Then, it would
    dynamically determine the number of pages to process.
    
    This would at least be better for checking for the last page. Chunking would have to be done differently, though.
    """
    with tqdm(desc=f"Processing Pages {start_page}-{end_page}", unit="page", total=end_page - start_page + 1) as pbar:
        # loop for all available pages
        while page_num <= end_page:
            page_stats: PageStats = {
                "rows_total": 0,
                "centers_kept": 0,
                "filtered_no_program_name_col": 0,
                "filtered_no_program_link": 0,
                "filtered_program_page_missing": 0,
                "filtered_no_inspection_link": 0,
                "filtered_no_annual_pdf": 0,
                "filtered_bad_program_id": 0,
                "row_errors": 0,
                "centers_with_nc_link": 0,
                "centers_without_nc_link": 0,
                "centers_with_nc_rows": 0,
                "centers_with_empty_nc": 0,
                "nc_rows": 0,
            }

            # get the current page of results
            page_link = f"{url}&p={page_num}"

            # logging.info(f"Processing page {page_link}...")
            main_page = extract_html(page_link)

            if main_page is not None:

                try:
                    # get all results rows for further processing
                    results_list = main_page.find('div', class_='resultsList')
                    if results_list is None:
                        logging.error(f"No resultsList found for page {page_num}")
                        logging.info(
                            f"Page {page_num} summary | results=0 kept=0 filtered=0 "
                            f"nc_rows=0 (resultsList missing)"
                        )
                        pbar.update(1)
                        page_num += 1
                        continue

                    rows = results_list.find_all('div', class_='resultsListRow')
                    page_stats["rows_total"] = len(rows)

                    for row_idx, row in enumerate(rows):
                        try:
                            program_name_column = row.find('div', class_='resultsListColumn programListColumnName')
                            if program_name_column is None:
                                page_stats["filtered_no_program_name_col"] += 1
                                continue

                            program_link_tag = program_name_column.find('a')
                            if program_link_tag is None or 'href' not in program_link_tag.attrs:
                                page_stats["filtered_no_program_link"] += 1
                                continue

                            program_name = program_link_tag.text.strip()
                            program_url = rel_path + program_link_tag['href']

                            # get the html for the program page
                            program_page = extract_html(program_url)
                            if program_page is None:
                                page_stats["filtered_program_page_missing"] += 1
                                continue

                            # extract info from the childcare center page
                            inspection_url = extract_inspection(program_page, rel_path)
                            if inspection_url is None:
                                page_stats["filtered_no_inspection_link"] += 1
                            accreditations = extract_accreditations(program_page)
                            program_pdf_link, nc_link = extract_center_links(inspection_url, rel_path) if inspection_url is not None else (None, None)

                            # only save the pdf link if it is an annual inspection
                            if program_pdf_link is None:
                                page_stats["filtered_no_annual_pdf"] += 1
                                continue

                            match = re.search(r"pdf/(\d+)_", program_pdf_link)
                            if match is None:
                                logging.error(f"Could not parse program ID from PDF link: {program_pdf_link}")
                                page_stats["filtered_bad_program_id"] += 1
                                continue
                            program_ID = match.group(1)

                            program_df = pd.DataFrame(
                                {
                                    PROGRAM_ID: [program_ID],
                                    PROGRAM_NAME: [program_name],
                                    "pdf": [program_pdf_link],
                                }
                            )

                            # add accreditations as columns
                            for k, v in accreditations.items():
                                program_df[k] = [v]

                            address_columns = row.find_all("div", class_="resultsListColumn")
                            if len(address_columns) >= 4:
                                program_df['Address'] = [address_columns[1].get_text(strip=True)]
                                program_df['City'] = [address_columns[2].get_text(strip=True)]
                                program_df['Zip'] = [address_columns[3].get_text(strip=True)]

                            # save the current row information
                            pdf_urls.append(program_df)
                            page_stats["centers_kept"] += 1

                            if nc_link is not None:
                                page_stats["centers_with_nc_link"] += 1
                                rule_df = extract_all_non_compliances(nc_link, rel_path)
                                if not rule_df.empty:
                                    page_stats["centers_with_nc_rows"] += 1
                                    page_stats["nc_rows"] += len(rule_df)
                                    rule_df[PROGRAM_ID] = program_ID
                                    rule_df[PROGRAM_NAME] = program_name
                                    rule_df['nc_link'] = nc_link
                                    non_compliance_dfs.append(rule_df)
                                else:
                                    page_stats["centers_with_empty_nc"] += 1
                            else:
                                page_stats["centers_without_nc_link"] += 1

                        except Exception as e:
                            logging.error(f"Error in page {page_num}, row {row_idx}: {e}")
                            page_stats["row_errors"] += 1
                except Exception as e:
                    logging.error(f"Error in page {page_num}: {e}")
                    logging.error(f"Link: {page_link}")
            else:
                logging.error(f"Failed to retrieve results page {page_num}: {page_link}")

            filtered_total = max(page_stats["rows_total"] - page_stats["centers_kept"], 0)
            logging.info(
                "Page %s summary | results=%s kept=%s filtered=%s nc_rows=%s nc_centers_with_link=%s "
                "nc_centers_with_rows=%s nc_centers_empty=%s nc_centers_missing_link=%s row_errors=%s",
                page_num,
                page_stats["rows_total"],
                page_stats["centers_kept"],
                filtered_total,
                page_stats["nc_rows"],
                page_stats["centers_with_nc_link"],
                page_stats["centers_with_nc_rows"],
                page_stats["centers_with_empty_nc"],
                page_stats["centers_without_nc_link"],
                page_stats["row_errors"],
            )
            logging.info(
                "Page %s filter breakdown | no_program_name_col=%s no_program_link=%s "
                "program_page_missing=%s no_inspection_link=%s no_annual_pdf=%s bad_program_id=%s",
                page_num,
                page_stats["filtered_no_program_name_col"],
                page_stats["filtered_no_program_link"],
                page_stats["filtered_program_page_missing"],
                page_stats["filtered_no_inspection_link"],
                page_stats["filtered_no_annual_pdf"],
                page_stats["filtered_bad_program_id"],
            )

            # TODO: remove this debug code
            # use for debugging to limit the number of pages to scrape
            # if page_num > 1:
            #     break

            pbar.update(1)
            # next page
            page_num += 1

    # combine into a single dataframe
    if pdf_urls:
        url_df = pd.concat(pdf_urls, axis=0)
    else:
        logging.error(f"No PDF URLs extracted for pages {start_page}-{end_page}")
        url_df = pd.DataFrame()

    if non_compliance_dfs:
        nc_df = pd.concat(non_compliance_dfs, axis=0, ignore_index=True)
    else:
        logging.error(f"No non-compliance data extracted for pages {start_page}-{end_page}")
        nc_df = pd.DataFrame()

    if not nc_df.empty:
        nc_df.set_index([PROGRAM_ID, "rule", "occurrence"], inplace=True)
    if not url_df.empty:
        url_df.set_index(PROGRAM_ID, inplace=True)

    # return with the program name as the index
    return url_df, nc_df


def get_last_page(url: str) -> int:
    home_page = extract_html(url)
    if home_page is None:
        raise ValueError(f"Could not retrieve page from {url}")

    # retrieve the link to the last page via the last page button
    last_page = home_page.find("a", id="ContentPlaceHolder1_pagerPrograms_ctl00_PagingFieldForDataPager_lnkLast")
    if not last_page:
        raise ValueError("Could not find the link to the last page")

    # parse the page number from the link to the last page 
    link = last_page.get("href")
    if link is None:
        raise ValueError("Last page link did not include an href")
    query = urlparse(link).query
    page_number = int(parse_qs(query).get("p", [0])[0])
    return page_number



def parallel_extract(
    url: str,
    rel_path: str,
    total_pages: int = -1,
    chunk_size: int = 5,
    processes: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract information from the webpages in parallel chunks.

    :param url: the ODJFS URL for the childcare search results page.
    :param rel_path: the relative path to childcare results
    :param total_pages: if not provided, will process all pages available.
    :param chunk_size: the number of pages processed at a time in a given process
    :param processes: number of parallel processes
    """

    if total_pages < 0:
        # last page is included (zero-indexed pages)
        total_pages = get_last_page(url) + 1
    if total_pages == 0:
        return pd.DataFrame(), pd.DataFrame()

    logging.info(f"Total pages {total_pages}")

    # last page is included
    ranges: list[tuple[str, str, int, int]] = [
        (url, rel_path, start, min(start + chunk_size - 1, total_pages - 1))
        for start in range(0, total_pages, chunk_size)
    ]

    worker_count = max(1, min(processes, len(ranges)))
    logging.info(f"Processing {total_pages} pages in {len(ranges)} chunks of size {chunk_size} with {worker_count} processes...")
    with mp.Pool(worker_count) as pool:
        results = pool.starmap(extract_all_centers, ranges)
    if not results:
        return pd.DataFrame(), pd.DataFrame()

    # Combine results
    url_dfs, nc_dfs = zip(*results)
    combined_url_df = pd.concat(url_dfs, axis=0)
    combined_nc_df = pd.concat(nc_dfs, axis=0)

    if PROGRAM_NAME in combined_url_df.columns:
        combined_url_df.sort_values(by=PROGRAM_NAME, inplace=True)
    if PROGRAM_NAME in combined_nc_df.columns:
        combined_nc_df.sort_values(by=PROGRAM_NAME, inplace=True)

    return combined_url_df, combined_nc_df


def download_pdf(pdf_url: str) -> BytesIO | None:
    """
    Create a temporary pdf file for data extraction
    :param pdf_url: pdf to download
    :return: BytesIO object containing the pdf or None if invalid URL
    """

    response = requests.get(pdf_url)
    if response.status_code == 200:
        return BytesIO(response.content)
    else:
        logging.error(f"Failed to download PDF: {response.status_code}")
        return None

def load_ODJFS_data(
    folder: str | Path = "",
    odcy_link: str = ODCY_LINK,
    rel_path: str = REL_PATH,
    pdf_links_filename: str = "pdf_links.csv",
    nc_df_filename: str = "non_compliances.csv",
    num_jobs: int = 1
) -> pd.DataFrame:
    """
    Load or create ODJFS center data and PDF links.

    This function checks whether the specified output files already exist within the given folder.
    If not, it extracts the data and saves the files for reuse. Creates the folder if it does not exist.

    :param folder: Folder where output files will be stored.
    :param odcy_link: URL to scrape center data from.
    :param rel_path: Relative path for scraping logic.
    :param pdf_links_filename: Filename for the PDF links CSV (inside the folder).
    :param nc_df_filename: Filename for the non-compliances CSV (inside the folder).
    :param num_jobs: Number of parallel jobs to use in data extraction.
    :return: dataframe containing links to licensing PDFs
    """

    folder_path = Path(folder) if folder else Path(".")

    # ensure folder exists
    if folder:
        folder_path.mkdir(parents=True, exist_ok=True)

    # construct full paths to the CSVs
    pdf_links_path = folder_path / pdf_links_filename
    nc_df_path = folder_path / nc_df_filename

    if os.path.exists(pdf_links_path):
        logging.info("Loading existing data...")
        pdf_links = pd.read_csv(pdf_links_path, index_col=0)
    else:
        logging.info("Webscraping Data (make sure the paths are valid)")

        if num_jobs > 1:
            # formerly 212 pages. There are 416 at the time of writing...
            pdf_links, nc_df = parallel_extract(odcy_link, rel_path, total_pages=-1, chunk_size=5, processes=num_jobs)
        else:
            pdf_links, nc_df = extract_all_centers(odcy_link, rel_path)

        pdf_links.to_csv(pdf_links_path, index=True)
        nc_df.to_csv(nc_df_path, index=True)

    return pdf_links


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Script to extract PDF links and non-compliance data.")
    parser.add_argument("pdf_links_output",
                        help="Path to the output CSV file for extracted PDF links.")

    parser.add_argument("nc_output",
                        help="Path to the output CSV file for the non-compliance DataFrame.")

    parser.add_argument("--chunk_size",
                        type=int,
                        default=10,
                        help="Number of pages to handle in each extraction chunk (default: 10).")

    parser.add_argument("--num_workers",
                        type=int,
                        default=1,
                        help="Number of parallel worker processes. For context, 10 workers takes about 40-50 minutes to create the entire dataset. (default: 1).")

    parser.add_argument("--total_pages",
                        type=int,
                        default=-1,
                        help="Total number of result pages to process. Use -1 to process all available pages (default: -1).")

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging (includes sucesses and progress updates as well as failures).")

    args = parser.parse_args()
    
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # log file with timestamp
    log_file = log_dir / f"run_{datetime.now().strftime('%Y-%m-%d')}.log"

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
    )

    pdf_links, nc_df = parallel_extract(
        ODCY_LINK,
        REL_PATH,
        total_pages=args.total_pages,
        chunk_size=args.chunk_size,
        processes=args.num_workers
    )

    pdf_links.to_csv(args.pdf_links_output, index=True)
    nc_df.to_csv(args.nc_output, index=True)

    logging.info(f"Total inspection reports: {len(pdf_links)}")

    pdf_links = pdf_links[pdf_links['pdf'].str.contains("ANNUAL")]
    logging.info(f"Found {len(pdf_links)} annual inspection reports.")
