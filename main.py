import argparse
import re
from time import time

import easyocr
import multiprocessing as mp
import pandas as pd
import os
import urllib.request

from fuzzywuzzy import process, fuzz
from pdf2image import convert_from_path
from collections import defaultdict

from tqdm import tqdm

from webscraping import REL_PATH, ODCY_LINK, load_ODJFS_data
from text_extraction import process_image, DPI

# --------------------------------------------------
# Constants
# --------------------------------------------------

# field names
PROGRAM_NAME = "program_name"

# section names
PROGRAM_DETAILS = "Program Details"
LICENSE_CAPACITY = "License Capacity and Enrollment at the Time of Inspection"
RATIO_OBSERVED = "Staff-Child Ratios at the Time of Inspection"
SERIOUS_NC = "Serious Risk Non-Compliances"
MODERATE_NC = "Moderate Risk Non-Compliances"
LOW_NC = "Low Risk Non-Compliances"
IN_COMPLIANCE = "Rules in Compliance/Not Verified"


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def group_into_rows(extracted_text, threshold=5):
    """
    Group extracted_text entries into rows based on their y-coordinates.

    :param extracted_text: A list of tuples of the form:
                          [ ( (x, y, w, h), [ (text, conf), ... ] ), ... ]
                          assumed to be sorted top-to-bottom, left-to-right.
    :param threshold:    The distance in pixels to decide when to start a new row.
    :return:               A list (rows) of lists, each inner list is one row,
                           containing the sub-rectangle data.
    """
    rows = []
    current_row = []
    if not extracted_text:
        return rows

    # start the first row’s baseline from the very first rectangle's y
    _, first_data = extracted_text[0]
    current_row_y = extracted_text[0][0][1]
    current_row_h = extracted_text[0][0][3]

    for ((x, y, w, h), text_data) in extracted_text:

        if abs(y - current_row_y) > threshold or abs(h - current_row_h) > threshold:
            # push the old row into rows
            rows.append(current_row)
            # start a new row
            current_row = []
            current_row_y = y
            current_row_h = h

        current_row.append(((x, y, w, h), text_data))

    if current_row:
        rows.append(current_row)

    # sort each row by x-coordinate and remove empty rows
    rows = [sorted(r, key=lambda x: x[0][0]) for r in rows if not (len(r) == 1 and len(r[0][1]) == 0)]

    # remove rectangles coords from the rows
    rows = [[(data[1]) for data in row] for row in rows]

    return rows


def find_field_in_rows(
        rows,
        field,
        end_field=None,
        start_idx=0,
        thresh=95,
        check_all_columns=False
):
    """
    Find a field in a list of rows.
    Params:
        - `rows`: A list of rows, where each row is a list of columns.
        - `field`: The field to search for.
        - `end_field`: The field to stop searching at.
        - `start_idx`: The row index to start searching at.
        - `thresh`: The threshold for fuzzy matching.
        - `check_all_columns`: Whether to check all columns in each row. O.w., only the first column is checked.
    Search through `rows` for `field`.
      - If `check_all_columns` = False, only the first column is checked.
      - If `check_all_columns` = True, all columns in each row are checked.
    Stops early if `end_field` is encountered in a row/column (using the same fuzzy logic).

    Returns:
      The row index where `field` is found, or -1 if not found.
    """

    row_idx = start_idx
    before_end_field = True
    while row_idx < len(rows) and before_end_field:
        row = rows[row_idx]

        # decide if we check just the first column or all columns
        columns_to_check = row if check_all_columns else row[:1]

        for col_idx, col in enumerate(columns_to_check):
            if col:
                text_in_col = col[0][0]

                # check if we should stop early
                before_end_field = (
                        end_field is None or
                        len(columns_to_check) > 1 or  # the end fields should only be in the first column
                        fuzz.partial_ratio(end_field, text_in_col) <= thresh
                )

                # check if the field is in the column
                if before_end_field and fuzz.partial_ratio(field, text_in_col) > thresh:
                    return (row_idx, col_idx) if check_all_columns else row_idx

        row_idx += 1

    return (row_idx, -1) if check_all_columns else -1


def flatten_rows(rows):
    """
    Flatten a nested list of tuples while keeping only unique items.

    :param rows: List of nested tuples with text and confidence values
    :return: List of tuples with unique items and their highest confidence
    """

    # use a fuzzy set
    # seen = []
    flattened = []
    for col in rows:
        for textbox in col:
            for item, confidence in textbox:
                # if process.extractOne(item, seen, score_cutoff=thresh) is None:
                #     seen.append(item)
                flattened.append((item, confidence))

    return flattened


def count_compliances(program_df, rule_df):
    """
    Aggregates rule and compliance counts from the data DataFrame and merges them into the program DataFrame.

    :param program_df: DataFrame containing program details
    :param rule_df: DataFrame containing rule and compliance data
    :return: program_df with rule and compliance counts
    """
    rules_count_df = (
        rule_df.groupby([PROGRAM_NAME, 'rule'])
        .nunique()
        .groupby(PROGRAM_NAME)['domain']
        .count()
        .rename({'domain': 'No. Rules with Non-Compliances'})
    )

    compliance_count_df = (
        rule_df.groupby([PROGRAM_NAME, 'compliance'])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    compliance_count_df.rename(columns={'Serious': 'No. Serious Risk',
                                        'Moderate': 'No. Moderate Risk',
                                        'Low': 'No. Low Risk'}, inplace=True)

    merged_df = program_df.copy()
    merged_df = merged_df.merge(rules_count_df, on=PROGRAM_NAME, how='left')
    merged_df = merged_df.merge(compliance_count_df, on=PROGRAM_NAME, how='left')

    # Step 4: Fill NaN values (if necessary, e.g., programs with no data in rules or compliance)
    merged_df.fillna(0, inplace=True)

    return merged_df


# --------------------------------------------------
# Data Extraction Functions
# --------------------------------------------------
def process_program_details(rows, thresh=95):
    """
    Process the OCR results for the first page (or first few pages) of the PDF.

    Output a dataframe containing the partial results and the index of the last row processed.

    Includes:
    - Program Details
    - Inspection Information
    - Summary of Findings

    :param rows the OCR results grouped by rows
    :param thresh the threshold for fuzzy matching
    :return: partial_df, last_row_idx
    """

    p1_fields = [
        "Program Number",
        "Program Type",
        "County",
        "Building Approval Date",
        "Use Group/Code",
        "Occupancy Limit",
        "Maximum Under 2",  # under 2 1/2 but idk how this will be read...
        "Fire Inspection Approval Date",
        "Food Service Risk Level",
        "Inspection Type",
        "Inspection Scope",
        "Inspection Notice",
        "Inspection Date",
        "Begin Time",
        "End Time",
        "Reviewer",
        "No. Rules Verified",
        # "No. Rules with Non-compliance", Just use the non-compliance dataframe instead
        # "No. Serious Risk",
        # "No. Moderate Risk",
        # "No. Low Risk",
    ]

    extracted_data = {field: None for field in p1_fields}

    row_idx = 0

    # TODO: refactor to handle mutiple Reviewers, for example
    for field_name in p1_fields:
        row_idx, col_idx = find_field_in_rows(
            rows=rows,
            field=field_name,
            end_field=LICENSE_CAPACITY,
            start_idx=row_idx,
            thresh=thresh,
            check_all_columns=True  # check all columns in each row, return the column index
        )

        if col_idx == -1 or len(rows[row_idx][col_idx]) <= 1:
            if col_idx == -1:
                print(f"Field '{field_name}' not found in rows.")
            extracted_data[field_name] = None
        else:
            extracted_data[field_name] = rows[row_idx][col_idx][-1]

    p1_df = pd.DataFrame([extracted_data])

    return p1_df, row_idx + 1


def process_license_table(rows):
    table_rows = [
        "Infant",
        "Young Toddler",
        "Total Under 2",
        "Older Toddler",
        "Preschool",
        "School",
        "Total Capacity/Enrollment"]

    columns = [
        "Full Time",
        "Part Time",
        "Total"
    ]

    table = {"License Capacity": {}}

    row_idx = 0

    for i, t_row in enumerate(table_rows):

        prev = row_idx
        # row_idx = get_row(rows, t_row, start_idx=row_idx, end_field=RATIO_OBSERVED)
        row_idx = find_field_in_rows(
            rows,
            t_row,
            end_field=RATIO_OBSERVED,
            start_idx=row_idx,
        )

        if row_idx == -1:
            row_idx = prev
            continue

        current_row = rows[row_idx]

        # save license capacity totals
        if len(current_row) > len(columns) + 1:
            table["License Capacity"][t_row] = current_row[1][0]
            current_row = current_row[2:]
        else:
            current_row = current_row[1:]

        # TODO: remove debug
        # print(f"Current Row: {current_row}")

        table[t_row] = {columns[i]: field[0] if len(field) > 0 else (None, None) for i, field in enumerate(current_row)}

    df = pd.DataFrame({0: table}).T
    table_df = pd.concat([pd.json_normalize(df[col]).add_prefix(f"{col} ") for col in df.columns], axis=1)

    return table_df, row_idx + 1


def process_ratio_table(rows):

    table_rows = [
        # "Infant/Toddler",
        # "Preschool"
        "0 to < 12 months",
        "12 months to < 18 months",
        "18 months to < 30 months",
        "30 months to < 36 months",
        "3 years to < 4 years",
        "4 years to < 5 years",
        "School Age to < 11 years"
    ]

    row_idx = 0
    table = defaultdict(list)

    end_row = find_field_in_rows(
        rows,
        "Summary of Non-Compliances",
        start_idx=row_idx,
    )

    if end_row == -1:
        end_row = 0
    else:
        rows = rows[:end_row]

    flattened = []
    for col in rows:
        for textbox in col:
            if len(textbox) > 0:
                it, conf = textbox[0]
                for item, confidence in textbox[1:]:
                    it += f" {item}"
                    conf += confidence
                conf /= len(textbox)
                flattened.append((it, conf))

    rows = flattened

    for i, r in enumerate(rows):

        field = process.extractOne(r[0], table_rows, score_cutoff=85, scorer=fuzz.ratio)

        if field is not None and i + 1 < len(rows):
            ratio = rows[i + 1]
            match = re.search(r"(\d+)\s*\w+\s*(\d+)", ratio[0])
            if match:
                ratio = f"{match.group(1)}:{match.group(2)}", ratio[1]
            table[field[0]].append(ratio)

    df = pd.DataFrame({category: [values] for category, values in table.items()})

    return df, end_row


def process_rules_in_compliance(rows, rule_df, compliance_level):
    """
    Process the OCR results for the last page of the PDF.

    Output a dataframe containing the partial results and the index of the last row processed.

    Includes:
    - Rules in Compliance/Not Verified

    :param rows the OCR results grouped by rows
    :param rule_df the dataframe containing the rules for the center along with partial results
    :param compliance_level the compliance level to process
    :return: partial_df, last_row_idx
    """

    compliance_dict = {
        "Serious": MODERATE_NC,
        "Moderate": LOW_NC,
        "Low": IN_COMPLIANCE
    }

    next_section = compliance_dict[compliance_level]

    end_idx = find_field_in_rows(
        rows,
        next_section,
        start_idx=0,
        thresh=95,
    )

    if end_idx == -1:
        end_idx = len(rows)

    # flatten the rows and remove duplicates
    rows = flatten_rows(rows[:end_idx])

    domain = None
    rule = None
    rule_counts = defaultdict(int)

    for c, rule, _ in rule_df.index:
        rule_counts[rule] = 0
        center = c

    row_idx = 0
    while row_idx < len(rows):
        row, conf = rows[row_idx]

        if fuzz.partial_ratio("Domain", row) > 95:
            domain = row, conf
        elif fuzz.partial_ratio("Rule", row) > 95:
            # extract the rule
            rule = row[6:]

            # find the closest rule in the rule_df
            rule = process.extractOne(rule, rule_counts.keys(), score_cutoff=85)

            if rule is not None:
                rule = rule[0]

                rule_idx = (center, rule, rule_counts[rule])

                if rule_idx in rule_df.index:
                    rule_df.loc[rule_idx, "compliance"] = compliance_level

                    # add the
                    if domain is not None:
                        rule_df.loc[rule_idx, "domain"] = domain[0]

        elif fuzz.partial_ratio("Code", row) > 95:
            code = row

            row_idx += 1

            # go until the row containss Finding to indicate the end of the rule
            while row_idx < len(rows) and fuzz.partial_ratio("Finding", rows[row_idx]) < 95:
                code += rows[row_idx][0]
                row_idx += 1

            idx = (center, rule, rule_counts[rule])
            if idx in rule_df.index:
                rule_df.loc[idx, "code"] = code[6:]

            # update the rule count
            rule_counts[rule] += 1

        row_idx += 1

    return None, end_idx


# --------------------------------------------------
# Main Extraction Functions
# --------------------------------------------------

def process_ocr_results(center_df: pd.DataFrame, rule_df: pd.DataFrame, extracted_texts: list,
                        section_methods: list[tuple[str, callable]]):
    """
    Process the extracted text from the PDFs and return a DataFrame with the processed data.
    :param center_df: a DataFrame containing the center's information (e.g., name, address, etc.)
    :param rule_df: a DataFrame containing the rules for the center along with partial results
    :param extracted_texts: a list of extracted text from the PDFs
    :param section_methods: a list of tuples containing the section name, the method to process the section, and any additional arguments
    :return: a tuple containing the processed DataFrame and the finding/code dataframe
    """
    rows = []

    # group the extracted text into rows to avoid page breaks
    for extracted_text in extracted_texts:
        rows += group_into_rows(extracted_text)

    print(f"Rows: {len(rows)}")
    # print(rows)

    # use a separate dataframe for the rules
    processed_dfs = [center_df.reset_index(drop=False)]

    # try:
    for i, (field, method, kwargs) in enumerate(section_methods):

        # print(f"Processing Section: {field}")
        df, row_idx = method(rows, **kwargs)

        if df is not None:
            processed_dfs.append(df)

        # ensure at end of the section
        row_idx, find_field_in_rows(rows,
                                    field,
                                    start_idx=row_idx,
                                    end_field=None if i == len(section_methods) - 1 else section_methods[i + 1][0]
                                    )
        rows = rows[row_idx:]

    program_df = pd.concat(processed_dfs, axis=1)
    # program_df.set_index(PROGRAM_NAME, inplace=True)

    # rule_df[PROGRAM_NAME] = center_df.index[0]
    # rule_df.set_index(PROGRAM_NAME, inplace=True)

    return program_df, rule_df
    # except Exception as e:
    #     print(f"Error processing rows: {e}")
    #     print(f"Row Index: {row_idx}")
    #     print(f"Section: {section_methods[i][0]}")
    #     return None


def process_df_chunk(center_df, rule_df):
    '''
    Process a chunk of the center_df and rule_df DataFrames.
    :return: a list of tuples containing the processed DataFrames for each center
    '''
    SECTION_METHODS = [
        (PROGRAM_DETAILS, process_program_details, {}),
        (LICENSE_CAPACITY, process_license_table, {}),
        (RATIO_OBSERVED, process_ratio_table, {}),
        (SERIOUS_NC, process_rules_in_compliance, {"compliance_level": "Serious", "rule_df": rule_df}),
        (MODERATE_NC, process_rules_in_compliance, {"compliance_level": "Moderate", "rule_df": rule_df}),
        (LOW_NC, process_rules_in_compliance, {"compliance_level": "Low", "rule_df": rule_df}),
    ]

    results = []

    ocr_kwargs = {
        "width_ths": 1,
        "batch_size": 10,
    }

    ocr = easyocr.Reader(['en'], gpu=True)

    for program_name, center in tqdm(center_df.iterrows(), total=center_df.shape[0], desc="Processing PDFs"):
        # try:
        local_file, _ = urllib.request.urlretrieve(center['pdf'])
        images = convert_from_path(local_file, dpi=DPI)

        extracted_texts = [process_image(image, ocr, ocr_kwargs=ocr_kwargs, verbose=True) for image in images]

        # clean up the buffered image
        os.remove(local_file)
        del images

        start_time = time()

        center_df_row = center.to_frame().T
        rules = rule_df[rule_df.index.get_level_values(PROGRAM_NAME) == program_name].copy()
        program_df, rules = process_ocr_results(center_df_row, rules, extracted_texts, SECTION_METHODS)

        print(f"Processing {program_name} results took {time() - start_time:.2f} seconds.")
        results.append((program_df, rules))
        # except Exception as e:
        #     print(f"Error processing center {center['pdf']}: {e}")

    return results


# --------------------------------------------------
# Main
# --------------------------------------------------
def main(args):
    if os.path.exists(args.center_path) and os.path.exists(args.non_compliance_path):
        partial_center_df = pd.read_csv(args.center_path, index_col=0)
        partial_rule_df = pd.read_csv(args.non_compliance_path, index_col=0)
    else:
        partial_center_df = pd.DataFrame()
        partial_rule_df = pd.DataFrame()

    print(f"Pdf Links Path: {args.pdf_links_path}")
    print(f"NC DF Path: {args.nc_df_path}")

    center_df, rule_df = load_ODJFS_data(
        ODCY_LINK, REL_PATH, pdf_links_path=args.pdf_links_path, nc_df_path=args.nc_df_path, num_jobs=args.num_jobs
    )

    center_df = center_df.iloc[:20].copy()
    rule_df = rule_df[rule_df.index.get_level_values(PROGRAM_NAME).isin(center_df.index)].copy()

    if not partial_center_df.empty:
        # remove already processed centers from the data
        center_df = center_df[~center_df.index.isin(partial_center_df.index)]
        rule_df = rule_df[~rule_df.index.get_level_values(PROGRAM_NAME).isin(partial_center_df.index)]

    # split the center_df into chunks
    chunk_size = len(center_df) // args.batch_size
    center_chunks = [center_df.iloc[i:i + chunk_size] for i in range(0, len(center_df), chunk_size)]
    unique_centers = [center_chunk.index.unique() for center_chunk in center_chunks]
    rule_chunks = [rule_df[rule_df.index.get_level_values(PROGRAM_NAME).isin(c)] for c in unique_centers]
    all_args = [(center_chunk, rule_chunk) for center_chunk, rule_chunk in zip(center_chunks, rule_chunks)]

    print(f"Processing {len(center_df)} centers in {len(all_args)} chunks.")

    # process in batches of num_jobs with tqdm progress bar
    num_batches = (len(all_args) + args.num_jobs - 1) // args.num_jobs
    print(f"Processing in {num_batches} batches.")

    batch_num = 1
    for batch_start in range(0, len(all_args), args.num_jobs):
        batch_args = all_args[batch_start:batch_start + args.num_jobs]

        with mp.Pool(processes=args.num_jobs) as pool:
            batch_results = pool.starmap(process_df_chunk, batch_args)

        batch_center_df = pd.concat([result[0] for result in batch_results], axis=0)
        batch_rule_df = pd.concat([result[1] for result in batch_results], axis=0)

        partial_center_df = pd.concat([partial_center_df, batch_center_df], axis=0)
        partial_rule_df = pd.concat([partial_rule_df, batch_rule_df], axis=0)

        # save checkpoint after each batch
        partial_center_df.to_csv(args.center_path, index=True)
        partial_rule_df.to_csv(args.non_compliance_path, index=True)

        print(f"Checkpoint saved after processing batch {batch_num}.")
        batch_num += 1

    print("All batches processed and results saved.")

    # TODO: order the column names
    final_center_df = count_compliances(partial_center_df, partial_rule_df)
    final_center_df.sort_index(axis=1, inplace=True)
    final_center_df.to_csv(args.center_path, index=True)
    print(f"Final center results saved to {args.center_path}")

    # sort the rule dataframe by program name
    final_rule_df = partial_rule_df.sort_index(axis=0, level=0)
    final_rule_df.to_csv(args.non_compliance_path, index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract data from a PDF file.")
    parser.add_argument("center_path", help="Path to the output CSV file for program center info.")
    parser.add_argument("non_compliance_path", help="Path to the output CSV file for non-compliance info.")
    parser.add_argument("pdf_links_path", help="Path to the CSV file containing the PDF links.")
    parser.add_argument("nc_df_path", help="Path to the CSV file containing the web-scraped non-compliance data.")
    parser.add_argument("--num_jobs", type=int, help="Number of jobs to run in parallel.")
    parser.add_argument("--batch_size",
                        type=int,
                        help="Number of PDFS to process in a chunk. Default is 16.",
                        default=16)
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    args = parser.parse_args()

    main(args)
