import pandas as pd
import requests

from bs4 import BeautifulSoup
from datetime import datetime
from io import BytesIO

# Link Constants
ODCY_LINK = "https://childcaresearch.ohio.gov/search?q=fVLNbhMxEN40v0uapgKhHhAiBy6VQkUR1xwWN1VDIVl1V0gFcXDWk42FY6%2b83pS98Q6ICxdeg1fgyBvwJjB2uhCJqrPS2DOf59tvPPZqnuf9RrOrtd0ddCRhyRXMiVqtlBwO3oDOuZKj46On9hsOSCFMoWEkoTCaiuEgLOaCJ%2bdQxuoDyJEshGhaxkfbREcn8ewoAqqTJdHcgOb0MZ7ZC7VacwZ6WqzmoFtEFdKUDcJN2X7LM6IY7FZH4jKD1gU1XKbdkGrDqZjSFbQnckGlyXuXWJvGijGBknszYUuuIz%2fUkCdLpYQfuSVIoX9Cy3y2mGWgkVPJ%2fTNV6O1E8zVQke%2b%2fgIXSsCkjVEN3vAaJGuy%2bE13x1QqD%2b1EGiRUEwHKy5IJZeC%2fWVOaZ0sYR9oMFNv6PqTdbg5Y8XRob3TnlIFiseZYfjKkWpaPBs2z8MUMaJPDPgLLIYO%2f3Qs3X1MA5l9hmihmQB5tBiPK0wCT7q%2bLuK07nXOCVTmReoKIEmtNgfEma0zEJA9wTEqI%2fJaROZkEjINEEB5FKThsThLxaDce5eSPthnejNTudmrOH%2fw3dXesFlSm8e2%2ffV%2b1Gs8jOk%2bNW3bL59cpVL9MFm51V4DdvloHWt3h7q7TtztrIIo0Htwi0IvyWdZbA71jn2%2b5s7rZCp9cNJub4IseS2bUTgYDEAHM8rkUv%2b%2frymV1%2fPf%2fCnC77p9Z25vvn4JtDOhVynfl0%2bPPwh0P8CqmsYu3%2bAQ%3d%3d"
REL_PATH = "https://childcaresearch.ohio.gov/"

# string constants
ANNUAL = "ANNUAL"

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
        print(f"Failed to retrieve page, status code: {response.status_code}")
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


def extract_pdf(url, rel_path) -> str:
    """
    Extract the pdf link from the inspection page
    :param url: inspection url
    :param rel_path: relative path to append to the pdf link
    :return: the pdf link
    """

    inspection_page = extract_html(url)
    most_recent_pdf_link = None
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

    return most_recent_pdf_link


def extract_all_pdfs(url, rel_path) -> pd.DataFrame:
    """
    Extract all pdf links and associated center info (e.g., name and address info) into a dataFrame for further parsing.

    :param url: The Ohio childcaresearch website URL (https://childcaresearch.ohio.gov/search for licensed childcare)
    :param rel_path: The relative path to append to the pdf links (e.g., https://childcaresearch.ohio.gov)
    :return: a dataframe containing the center name, address info, and link to the pdf for the most recent center licensing inspection
    """

    # NOTE: could probably automate rel_path extraction from the url

    pdf_urls = []
    main_page = None
    page_num = 0

    # loop for all available pages
    while not (pdf_urls and main_page is None):

        # get the current page of results
        page_link = f"{url}&p={page_num}"
        print(f"Page link: {page_link}")
        main_page = extract_html(page_link)

        if main_page is not None:
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
                        program_pdf_link = extract_pdf(inspection_url, rel_path) if inspection_url is not None else None

                        # only save the pdf link if it is an annual inspection
                        if program_pdf_link is not None:
                            program_df['program_name'] = [program_name]
                            program_df['pdf'] = [program_pdf_link]
                        else:
                            continue

                address_columns = row.findAll("div", class_="resultsListColumn")
                if address_columns:
                    program_df['Address'] = [address_columns[1].get_text(strip=True)]
                    program_df['City'] = [address_columns[2].get_text(strip=True)]
                    program_df['Zip'] = [address_columns[3].get_text(strip=True)]

                # save the current row  information
                pdf_urls.append(program_df)

            # use for debugging to limit the number of pages to scrape
            if page_num > 5:
                break

        # next page
        page_num += 1

    # combine into a single dataframe
    url_df = pd.concat(pdf_urls, axis=0)

    # return with the program name as the index
    return url_df.set_index("program_name")


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


if __name__ == "__main__":
    # extract all pdf links
    pdf_links = extract_all_pdfs(ODCY_LINK, REL_PATH)

    print(f"Total inspection reports: {len(pdf_links)}")

    # ensure all pdf links are annual
    pdf_links = pdf_links[pdf_links['pdf'].str.contains("ANNUAL")]
    print(f"Found {len(pdf_links)} annual inspection reports.")

