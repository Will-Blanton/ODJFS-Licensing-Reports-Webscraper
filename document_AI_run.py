import argparse
import cv2
import fitz
import io
import logging
import numpy as np
import pandas as pd
import re
import requests

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import InternalServerError
from google.api_core.exceptions import RetryError
from google.cloud import documentai
from google.cloud import storage
from google.cloud.documentai_toolbox import gcs_utilities
from pathlib import Path
from PIL import Image
from typing import Optional

from webscraping import load_ODJFS_data
from config import DOC_PROCESSOR_ID, PROJECT_ID, GCS_BUCKET

# NOTE: max number of concurrent batches for the Google API is 5
MAX_API_CALLS = 5

# Given unprocessed documents in path gs://bucket/path/to/folder
# gcs_bucket_name = "bucket"
# gcs_prefix = "path/to/folder"
# batch_size = 50

def create_gcs_bucket(bucket_name, delete=False):
    """Deletes the bucket if it exists, then creates a new bucket in Google Cloud Storage."""
    storage_client = storage.Client()

    # Check if the bucket already exists
    bucket = storage_client.lookup_bucket(bucket_name)
    if bucket and delete:
        logging.info(f"Bucket {bucket_name} already exists. Deleting it.")
        bucket.delete(force=True)

        # create a new bucket
        new_bucket = storage_client.create_bucket(bucket_name)
        logging.info(f"Bucket {new_bucket.name} created.")

    return storage_client.bucket(bucket_name)


def preprocess_image(pix: fitz.Pixmap, sharpen: bool = True) -> Image.Image:
    """
    Convert a PyMuPDF pixmap to a PIL.Image that is:
      • grayscale (smaller file, consistent contrast)
      • Otsu‑binarised (keeps faint strokes, kills background noise)
      • lightly sharpened (optional) to recover thin text lines

    Needed due to issues with Google Doc AI's native image preprocessing and certain documents.

    Returns a 3‑channel RGB image ready for PDF embedding.
    """
    # pixmap → ndarray (BGR, no alpha)
    arr = np.frombuffer(pix.samples, np.uint8)\
            .reshape(pix.height, pix.width, 4 if pix.alpha else 3)

    if pix.alpha:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)

    gray   = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

    # Otsu gives per‑page threshold; safer than “fixed 180”
    _, bin_ = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    if sharpen:
        blur  = cv2.GaussianBlur(bin_, (0, 0), 3)
        bin_  = cv2.addWeighted(bin_, 1.5, blur, -0.5, 0)

    # back to RGB so PIL→PDF works
    rgb = cv2.cvtColor(bin_, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb)


def download_and_upload_pdf(bucket, blob_name, pdf_url, page_count=1, dpi=300):
    """
    Download a PDF from a URL and upload it to GCS for processing.

    :param bucket: GCS bucket object
    :param blob_name: name of the PDF file (e.g., the PDF's ID)
    :param pdf_url: URL of the PDF file
    :param page_count: number of pages to download
    :param dpi: resolution of the PDF (300 by default. This works, so use unless major issues with pricing)
    """
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
        pdf_content = response.content

        # retrieve the first page_count pages
        input_pdf = fitz.open(stream=pdf_content, filetype='pdf')

        # extract the first page(s) and preprocess for field extraction/OCR
        images = []
        for page_num in range(min(page_count, len(input_pdf))):
            pix = input_pdf[page_num].get_pixmap(dpi=dpi)
            img = preprocess_image(pix, sharpen=True)
            images.append(img)

        pdf_buffer = io.BytesIO()
        images[0].save(pdf_buffer, format="PDF", save_all=True, append_images=images[1:])
        pdf_buffer.seek(0)

        # upload to GCS
        blob = bucket.blob(blob_name)
        blob.upload_from_file(pdf_buffer, content_type='application/pdf')

        logging.info(f"Uploaded {blob_name} to GCS.")
    except Exception as e:
        logging.error(f"Failed to download and upload {pdf_url} to GCS: {e}")


def process_pdf_batch(bucket, gcs_prefix: str, pdf_links: pd.Series, num_pages):
    """Process a batch of PDF links: download and upload each one."""
    for idx, link in pdf_links.items():
        blob_name = f"{gcs_prefix}/{idx}.pdf"
        download_and_upload_pdf(bucket, blob_name, link, page_count=num_pages)


def upload_files_to_gcs(bucket, gcs_prefix: str, pdf_links: pd.Series, batch_size, num_workers, num_pages=1):
    """
    Uploads PDF files to Google Cloud Storage in parallel batches. PDFs need to be uploaded to GCS for processing.

    WARNING: too many workers may lead to issues with uploading PDF files to GCS. Keep num_workers <= 5 for safety or add backoff/retry mechanisms
    """

    batches = [pdf_links[i:i + batch_size] for i in range(0, len(pdf_links), batch_size)]

    # NOTE: Could revisit with exponential backoff if issues with rate limits (no issues with 8 threads)
    # uses threads since this is primarily an IO/network operation
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for batch_index, batch_links in enumerate(batches):
            futures.append(
                executor.submit(process_pdf_batch, bucket, gcs_prefix, batch_links, num_pages)
            )

        # wait for all batches to complete
        for future in futures:
            future.result()

    logging.info("All batches uploaded to GCS.")


def get_fields(document):
    """
    Extract document fields from an OCR output JSON doc
    """

    fields = {}
    
    for ent in document.entities:

        # process the table (listed as grouped fields/entities)
        if ent.properties:

            for ep in ent.properties:

                ep_label = ep.type_

                if ep.properties:
                    # handle two levels of nesting
                    fields.update({f"{ep_label}-{prop.type_}": prop.mention_text for prop in ep.properties}) 
                else:
                    fields[ep_label] = ep.mention_text

        else:
            fields[ent.type_] = ent.mention_text
            
    return fields


def process_ocr_batch(client: documentai.DocumentProcessorServiceClient, request, timeout, verbose=False):
    """
    Process a batch of OCR documents and return their fields in a DataFrame.
    """
    
    operation = client.batch_process_documents(request)
    
    # wait until the batch process is complete
    try:
        logging.info(f"Waiting for operation {operation.operation.name} to complete...")
        operation.result(timeout=timeout)

    # catch exception when operation doesn't finish before timeout
    except (RetryError, InternalServerError) as e:
        logging.error(e.message)
            
    # after the operation is complete, get output document information from operation metadata
    metadata = documentai.BatchProcessMetadata(operation.metadata)

    if metadata.state != documentai.BatchProcessMetadata.State.SUCCEEDED:
        raise ValueError(f"Batch Process Failed: {metadata.state_message}")

    storage_client = storage.Client()
    
    documents = []

    # one process per Input Document
    for process in list(metadata.individual_process_statuses):

        # output_gcs_destination format: gs://BUCKET/PREFIX/OPERATION_NUMBER/INPUT_FILE_NUMBER/
        # The Cloud Storage API requires the bucket name and URI prefix separately
        matches = re.match(r"gs://(.*?)/(.*)", process.output_gcs_destination)
        if not matches:
            logging.error(
                "Could not parse output GCS destination:",
                process.output_gcs_destination,
            )
            continue

        output_bucket, output_prefix = matches.groups()

        # Get List of Document Objects from the Output Bucket
        output_blobs = storage_client.list_blobs(output_bucket, prefix=output_prefix)

        # Document AI may output multiple JSON files per source file
        current_doc_fields = {}
        doc_ID = None
        for blob in output_blobs:
            # Document AI should only output JSON files to GCS
            if blob.content_type != "application/json":
                logging.error(
                    f"Skipping non-supported file: {blob.name} - Mimetype: {blob.content_type}"
                )
                continue
                
            if doc_ID is None:
                doc_ID = blob.name.split("/")[-1].split(".")[0].split("-")[0]

            # download JSON File as bytes object and convert to Document Object
            logging.info(f"Fetching {blob.name}")

            document = documentai.Document.from_json(
                blob.download_as_bytes(), ignore_unknown_fields=True
            )
            
            fields = get_fields(document)
            current_doc_fields.update(fields)
            
        documents.append(pd.DataFrame(current_doc_fields, index=[doc_ID]))

    return pd.concat(documents)

def extract_from_pdfs(
        gcs_bucket_name: str,
        gcs_prefix: str,
        batch_size: int = 50,
        location='us',
        field_mask: Optional[str] = None,
        timeout: int = 400,
        threads: Optional[int] = None,
        verbose=True
): 
    """
    Use Google Document AI to extract the fields from all PDF files in the given bucket and output as a DataFrame.
    """
    def process_doc_batch(batch):

        request = documentai.BatchProcessRequest(
            name=proc_name,
            input_documents=batch,
            document_output_config=output_config
        )

        return process_ocr_batch(client, request, timeout, verbose)

    opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")

    client = documentai.DocumentProcessorServiceClient(client_options=opts)
    
    # Cloud Storage URI for the Output Directory
    gcs_output_config = documentai.DocumentOutputConfig.GcsOutputConfig(
        gcs_uri=f"gs://{GCS_BUCKET}/output/", field_mask=field_mask
    )

    # Where to write results
    output_config = documentai.DocumentOutputConfig(gcs_output_config=gcs_output_config)

    batches = gcs_utilities.create_batches(
        gcs_bucket_name=gcs_bucket_name, 
        gcs_prefix=gcs_prefix, 
        batch_size=batch_size
    )

    proc_name = client.processor_path(PROJECT_ID, location="us", processor=DOC_PROCESSOR_ID)

    logging.info(f"{len(batches)} batch(es) created.")
    
    processed_docs = [] 

    # submit batches in parallel   
    with ThreadPoolExecutor(max_workers=min(threads, MAX_API_CALLS)) as executor:   
        futures = [executor.submit(process_doc_batch, batch) for batch in batches]

        for future in as_completed(futures):
            try:
                response = future.result()
                processed_docs.append(response)
            except Exception as e:
                logging.error(f"Batch failed: {e}")
        
    return pd.concat(processed_docs)


def main(args):

    # links = pd.read_csv("output/2024-2025/pdf_links.csv", index_col=0).iloc[0]

    # webscrape the links or load from an existing file (don't need the rules)
    links, _ = load_ODJFS_data(
        folder=args.output_folder,
        num_jobs=args.num_jobs
    )

    logging.info(f"Bucket name: {GCS_BUCKET}")

    # upload the pdfs to gcs before processing
    bucket = create_gcs_bucket(GCS_BUCKET)
    upload_files_to_gcs(bucket, args.gcs_prefix, links['pdf'], args.batch_size, args.num_jobs)

    # Perform OCR on the pdfs
    processed = extract_from_pdfs(
            gcs_bucket_name=GCS_BUCKET,
            gcs_prefix=args.gcs_prefix,
            batch_size=args.batch_size,
            field_mask="entities", # only extract entities/fields
            threads=args.num_jobs,
            timeout=900
    )

    # ensure indices are strings for matching
    links.index = links.index.astype(str)

    final = links.merge(processed, how='outer', left_index=True, right_index=True)

    final.to_csv(f"{args.output_folder}/center_data.csv")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Extract licensing data from the ODJFS website and licensing PDFs.")
    # parser.add_argument("output_folder", help="Path to the folder that will contain the output CSV file for program center info, pdf links, and non-compliances.")
    parser.add_argument("--output_folder", default="output/2024-2025")
    parser.add_argument("--num_jobs", 
                        type=int, 
                        help="Number of jobs to run in parallel.",
                        default=8
                        )
    parser.add_argument("--batch_size",
                        type=int,
                        help="Number of PDFS to process in a chunk. Default is 50.",
                        default=50)
    parser.add_argument("--gcs_prefix", 
                        type=str, 
                        help="Folder path to store the pdfs in Google Cloud. Default = 'input'",
                        default="input")
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

    main(args)
