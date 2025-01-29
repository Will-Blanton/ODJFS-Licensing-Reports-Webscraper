import os
import multiprocessing as mp
import pandas as pd
import re
import requests

from bs4 import BeautifulSoup
from datetime import datetime
from io import BytesIO

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


def extract_html(url):
    """
    Parse the html at the given url into a beautiful soup object, for manipulation.

    :param url: Any valid URL
    :return: the parsed HTML or None if the request failed
    """

    response = requests.get(url)

    # check if the request was successful (status code 200)
    if response.status_code == 200:

        # parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        return soup
    else:

        print(f"Failed to retrieve page from {url}, status code: {response.status_code}")
        return None


def extract_inspection(url, rel_path):
    """
    Extract the inspection link from the childcare center page

    :param url: the child care center page
    :param rel_path: relative path to append to the inspection link
    :return: the inspection page link
    """
    inspection_url = None

    # get the html for the program page
    program_page = extract_html(url)

    if program_page:
        inspection_button_span = program_page.find('span', class_='inspectionsButton')

        if inspection_button_span:
            inspection_link_tag = inspection_button_span.find_parent('a')

            if inspection_link_tag and 'href' in inspection_link_tag.attrs:
                inspection_url = rel_path + inspection_link_tag['href']

    return inspection_url


def extract_center_links(url, rel_path) -> (str, str):
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
            date_column = columns[0]

            # get the inspection type
            inspect_type = list(columns[1].stripped_strings)[1]

            pdf_col = row.find('span', class_='inspectionPDFlink')
            pdf_link_tag = pdf_col.find('a', href=True)

            if pdf_link_tag and date_column:

                date_column = list(date_column.stripped_strings)
                # format into a datetime object for date comparisons
                inspection_date = date_column[1]
                inspection_date = datetime.strptime(inspection_date, "%m/%d/%Y")
                pdf_link = pdf_link_tag['href']

                # only save the most recent date (may not be necessary, since all appear to be listed in order
                if inspect_type == ANNUAL and (most_recent_date is None or inspection_date > most_recent_date):
                    most_recent_date = inspection_date
                    most_recent_pdf_link = rel_path + pdf_link

                    # extract the non-compliance link
                    nc_link = row.find('span', class_='inspectionNonCompliantLink')
                    nc_link = nc_link.find('a', href=True)
                    if nc_link is not None:
                        most_recent_nc_link = rel_path + nc_link['href']
                    else:
                        most_recent_nc_link = None

    return most_recent_pdf_link, most_recent_nc_link


def extract_numbers_with_letters(input_text):
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


def extract_non_compliance(nc_link, rel_path) -> pd.DataFrame:
    """
    Extract non-compliance information from the non-compliance page

    :param nc_link: the non-compliance page link
    :param rel_path: relative path to append to the pdf link
    :return: a dataframe containing the non-compliance information
    """

    try:
        rules = []

        nc_page = extract_html(nc_link)
        results_list = nc_page.find('div', class_='resultsList')

        if results_list:
            rows = results_list.find_all('div', class_='resultsListRow')

            for row in rows:
                rule_columns = list(row.find_all('div', class_='resultsListColumn'))
                description = rule_columns[1].find('a')

                rule = description.get_text(strip=True)

                # indicate no findings but the rule still non-compliant
                findings = ['-1']
                code = "Missing NC link"

                if "href" in description.attrs:
                    description_link = rel_path + description['href']

                    rule_page = extract_html(description_link)

                    # ex
                    if rule_page:
                        # the misspelling is intentional... it is in the HTML
                        finding_txt = rule_page.find('span', class_='inspectionFindsingsText')
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

        nc_df = pd.concat(rules, axis=0)

        nc_df['occurrence'] = nc_df.groupby(by="rule").cumcount()

    except Exception as e:
        print(f"Error in extract_non_compliance: {e}")
        print(f"Link: {nc_link}")
        nc_df = None

    return nc_df


def extract_all_centers(url, rel_path, start_page=0, end_page=211) -> (pd.DataFrame, pd.DataFrame):
    """
    Extract all pdf links and associated center info (e.g., name and address info) into a dataFrame for further parsing.

    Also, non-compliance information is extracted into a separate dataframe for each center.

    :param url: The Ohio childcaresearch website URL (https://childcaresearch.ohio.gov/search for licensed childcare)
    :param rel_path: The relative path to append to the pdf links (e.g., https://childcaresearch.ohio.gov)
    :param start_page: The starting page to process
    :param end_page: The maximum number of pages to process
    :return: a dataframe containing the center name, address info, and link to the pdf for the most recent center licensing inspection and a dataframe containing the non-compliance information
    """

    # NOTE: could probably automate rel_path extraction from the url

    pdf_urls = []
    non_compliance_dfs = []
    main_page = None
    page_num = start_page

    with tqdm(desc=f"Processing Pages {start_page}-{end_page}", unit="page", total=end_page - start_page + 1) as pbar:
        # loop for all available pages
        while page_num <= end_page and not (pdf_urls and main_page is None):

            # get the current page of results
            page_link = f"{url}&p={page_num}"
            # print(f"Processing page {page_link}...")
            main_page = extract_html(page_link)

            if main_page is not None:

                try:
                    # get all results rows for further processing
                    results_list = main_page.find('div', class_='resultsList')
                    rows = results_list.find_all('div', class_='resultsListRow')

                    for row in rows:
                        program_name_column = row.find('div', class_='resultsListColumn programListColumnName')

                        program_df = pd.DataFrame()
                        if program_name_column:

                            program_link_tag = program_name_column.find('a')

                            if program_link_tag:
                                program_name = program_link_tag.text.strip()
                                program_url = rel_path + program_link_tag['href']
                                inspection_url = extract_inspection(program_url, rel_path)
                                program_pdf_link, nc_link = extract_center_links(inspection_url, rel_path) if inspection_url is not None else None

                                # only save the pdf link if it is an annual inspection
                                if program_pdf_link is not None:

                                    program_ID = re.search(r"pdf/(\d+)_", program_pdf_link).group(1)

                                    program_df[PROGRAM_ID] = [program_ID]
                                    program_df[PROGRAM_NAME] = [program_name]
                                    program_df['pdf'] = [program_pdf_link]

                                    if nc_link is not None:
                                        rule_df = extract_non_compliance(nc_link, rel_path)
                                        rule_df[PROGRAM_ID] = program_ID
                                        rule_df[PROGRAM_NAME] = program_name
                                        rule_df['nc_link'] = nc_link
                                        non_compliance_dfs.append(rule_df)
                                else:
                                    continue

                        address_columns = row.findAll("div", class_="resultsListColumn")
                        if address_columns:
                            program_df['Address'] = [address_columns[1].get_text(strip=True)]
                            program_df['City'] = [address_columns[2].get_text(strip=True)]
                            program_df['Zip'] = [address_columns[3].get_text(strip=True)]

                        # save the current row  information
                        pdf_urls.append(program_df)
                except Exception as e:
                    print(f"Error in page {page_num}: {e}")
                    print(f"Link: {page_link}")

                # TODO: remove this debug code
                # use for debugging to limit the number of pages to scrape
                # if page_num > 1:
                #     break

            pbar.update(1)
            # next page
            page_num += 1

    # combine into a single dataframe
    if pdf_urls and non_compliance_dfs:
        url_df = pd.concat(pdf_urls, axis=0)
        nc_df = pd.concat(non_compliance_dfs, axis=0)

        nc_df.set_index([PROGRAM_ID, "rule", "occurrence"], inplace=True)
        url_df.set_index(PROGRAM_ID, inplace=True)
    else:
        if not pdf_urls:
            print(f"No PDF URLs extracted for pages {start_page}-{end_page}")
        if not non_compliance_dfs:
            print(f"No non-compliance data extracted for pages {start_page}-{end_page}")

        url_df = pd.DataFrame()
        nc_df = pd.DataFrame()

    # return with the program name as the index
    return url_df, nc_df


def parallel_extract(url, rel_path, total_pages, chunk_size=5, processes=4):
    ranges = [(url, rel_path, start, min(start + chunk_size - 1, total_pages))
              for start in range(0, total_pages, chunk_size)]

    print(f"Processing {total_pages} pages in {len(ranges)} chunks of size {chunk_size} with {processes} processes...")

    with mp.Pool(processes) as pool:
        results = pool.starmap(extract_all_centers, ranges)

    # Combine results
    url_dfs, nc_dfs = zip(*results)
    combined_url_df = pd.concat(url_dfs, axis=0)
    combined_nc_df = pd.concat(nc_dfs, axis=0)

    combined_url_df.sort_values(by=PROGRAM_NAME, inplace=True)
    combined_nc_df.sort_values(by=PROGRAM_NAME, inplace=True)

    return combined_url_df, combined_nc_df


def download_pdf(pdf_url) -> BytesIO | None:
    """
    Create a temporary pdf file for data extraction
    :param pdf_url: pdf to download
    :return: BytesIO object containing the pdf or None if invalid URL
    """

    response = requests.get(pdf_url)
    if response.status_code == 200:
        return BytesIO(response.content)
    else:
        print(f"Failed to download PDF: {response.status_code}")
        return None


def load_ODJFS_data(odcy_link, rel_path, pdf_links_path="pdf_links.csv", nc_df_path="non_compliance.csv", num_jobs=1):
    """
    Checks if specified files already exist. If not, generates them using extract_all_centers,
    saves them, and returns the corresponding DataFrames.

    Args:
        odcy_link (str): The link used to extract center data.
        rel_path (str): The relative path used in the extraction process.
        pdf_links_path (str): File path to save or load the pdf_links DataFrame.
        nc_df_path (str): File path to save or load the non_compliance DataFrame.

    Returns:
        tuple: A tuple of two DataFrames (pdf_links, nc_df).
    """

    if os.path.exists(pdf_links_path) and os.path.exists(nc_df_path):
        print("Loading existing data...")
        pdf_links = pd.read_csv(pdf_links_path, index_col=0)
        nc_df = pd.read_csv(nc_df_path)
        nc_df.reset_index(inplace=True, drop=True)
        nc_df.set_index(["program_id", "rule", "occurrence"], inplace=True)
    else:
        print("Webscraping Data (make sure the paths are valid)")
        if num_jobs > 1:
            pdf_links, nc_df = parallel_extract(odcy_link, rel_path, total_pages=212, chunk_size=5, processes=num_jobs)
        else:
            pdf_links, nc_df = extract_all_centers(odcy_link, rel_path)

        pdf_links.to_csv(pdf_links_path, index=True)
        nc_df.to_csv(nc_df_path, index=True)

    return pdf_links, nc_df


if __name__ == "__main__":
    # extract all pdf links
    pdf_links, nc_df = parallel_extract(ODCY_LINK, REL_PATH, total_pages=212, chunk_size=10, processes=10)

    pdf_links.to_csv("pdf_links_final.csv", index=True)
    nc_df.to_csv("nc_df_final.csv", index=True)
    print(f"Total inspection reports: {len(pdf_links)}")

    # ensure all pdf links are annual
    pdf_links = pdf_links[pdf_links['pdf'].str.contains("ANNUAL")]
    print(f"Found {len(pdf_links)} annual inspection reports.")

