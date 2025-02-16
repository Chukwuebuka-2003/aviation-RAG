import os
import pymupdf  # Explicitly using pymupdf
from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field, ConfigDict
from groundx import Document, GroundX
from dotenv import load_dotenv

load_dotenv()


class DocumentSearchToolInput(BaseModel):
    """Input schema for DocumentSearchTool."""
    query: str = Field(..., description="Query to search the document.")


class DocumentSearchTool(BaseTool):
    name: str = "DocumentSearchTool"
    description: str = "Search the document for the given query."
    args_schema: Type[BaseModel] = DocumentSearchToolInput

    model_config = ConfigDict(extra="allow")

    def __init__(self, file_path: str):
        """Initialize the searcher with a PDF file path and set up the GroundX collection."""
        super().__init__()
        self.file_path = file_path
        self.client = GroundX(api_key=os.getenv("GROUNDX_API_KEY"))
        self.bucket_id = self._create_bucket()

        # Check file size before uploading. If >10MB, split into smaller parts.
        if os.path.getsize(file_path) > 10_485_760:  # 10MB
            print("PDF is too large! Splitting into smaller parts...")
            # Use a lower max_pages (e.g., 10 pages per file) to ensure each part is under 10MB
            self.pdf_parts = split_pdf(file_path, max_pages=10)
        else:
            self.pdf_parts = [file_path]

        # Upload each PDF part separately
        self.process_ids = [self._upload_document(part) for part in self.pdf_parts]

    def _upload_document(self, file_path):
        """Uploads a PDF document to GroundX."""
        ingest = self.client.ingest(
            documents=[
                Document(
                    bucket_id=self.bucket_id,
                    file_name=os.path.basename(file_path),
                    file_path=file_path,
                    file_type="pdf",
                    search_data={"key": "value"},
                )
            ]
        )
        return ingest.ingest.process_id

    def _create_bucket(self):
        """Creates a storage bucket in GroundX for the PDFs."""
        response = self.client.buckets.create(name="agentic_rag")
        return response.bucket.bucket_id

    def _run(self, query: str) -> str:
        """Performs a document search across all PDF parts."""
        results = []
        for process_id in self.process_ids:
            status_response = self.client.documents.get_processing_status_by_id(
                process_id=process_id
            )

            if status_response.ingest.status != "complete":
                return "Document is still being processed..."

            search_response = self.client.search.content(
                id=self.bucket_id, query=query, n=10, verbosity=2
            )

            for result in search_response.search.results:
                results.append(result.text)

        return "\n____\n".join(results) if results else "No relevant information found."


def split_pdf(input_path, max_pages=10):
    """
    Splits a large PDF into smaller parts if it exceeds size limits.
    Each part will contain up to max_pages pages.
    """
    doc = pymupdf.open(input_path)
    parts = []
    for i in range(0, len(doc), max_pages):
        part_path = f"split_{i}.pdf"
        part_doc = pymupdf.open()  # Create a new empty document
        for j in range(i, min(i + max_pages, len(doc))):
            part_doc.insert_pdf(doc, from_page=j, to_page=j)
        part_doc.save(part_path)
        part_doc.close()
        parts.append(part_path)
    doc.close()
    return parts
