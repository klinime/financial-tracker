import os
import shutil
import zipfile

from adobe.pdfservices.operation.auth.service_principal_credentials import (
    ServicePrincipalCredentials,
)
from adobe.pdfservices.operation.pdf_services import PDFServices
from adobe.pdfservices.operation.pdf_services_media_type import PDFServicesMediaType
from adobe.pdfservices.operation.pdfjobs.jobs.extract_pdf_job import ExtractPDFJob
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_element_type import (
    ExtractElementType,
)
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_pdf_params import (
    ExtractPDFParams,
)
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.table_structure_type import (
    TableStructureType,
)
from adobe.pdfservices.operation.pdfjobs.result.extract_pdf_result import (
    ExtractPDFResult,
)


class PDFTextTableExtractor:
    def __init__(self) -> None:
        credentials = ServicePrincipalCredentials(
            client_id=os.getenv("PDF_SERVICES_CLIENT_ID"),
            client_secret=os.getenv("PDF_SERVICES_CLIENT_SECRET"),
        )
        self.pdf_services = PDFServices(credentials=credentials)

    def extract_text_table(self, pdf_path: str, extract_dest: str) -> None:
        input_stream = open(pdf_path, "rb").read()
        input_asset = self.pdf_services.upload(
            input_stream=input_stream, mime_type=PDFServicesMediaType.PDF
        )
        extract_pdf_params = ExtractPDFParams(
            elements_to_extract=[ExtractElementType.TEXT, ExtractElementType.TABLES],
            table_structure_type=TableStructureType.CSV,
        )
        extract_pdf_job = ExtractPDFJob(
            input_asset=input_asset, extract_pdf_params=extract_pdf_params
        )
        location = self.pdf_services.submit(extract_pdf_job)
        pdf_services_response = self.pdf_services.get_job_result(
            location, ExtractPDFResult
        )
        result_asset = pdf_services_response.get_result().get_resource()
        stream_asset = self.pdf_services.get_content(result_asset)
        with open(f"{extract_dest}.zip", "wb") as file:
            file.write(stream_asset.get_input_stream())
        shutil.rmtree(extract_dest)
        with zipfile.ZipFile(f"{extract_dest}.zip", "r") as zip_ref:
            zip_ref.extractall(extract_dest)
        os.remove(f"{extract_dest}.zip")
