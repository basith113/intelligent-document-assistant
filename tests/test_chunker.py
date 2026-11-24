"""Document extraction and chunking tests."""

from io import BytesIO

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app.core.errors import AppError
from app.rag.chunker import Chunker
from app.rag.loader import Page, load_document


def test_text_markdown_and_normalization():
    assert load_document(r"folder\notes.MD", b"\xef\xbb\xbf# Heading\r\n\r\nwords\t here") == [
        Page("# Heading\n\nwords here")
    ]
    assert load_document("notes.txt", b"plain text") == [Page("plain text")]


@pytest.mark.parametrize(("name", "data", "status"), [
    ("no.exe", b"text", 415),
    ("wrong.pdf", b"not a PDF", 422),
    ("binary.txt", b"\x00\xff", 422),
    ("empty.md", b" \n", 422),
])
def test_invalid_files(name, data, status):
    with pytest.raises(AppError) as error:
        load_document(name, data)
    assert error.value.status == status


def test_pdf_page_numbers_skip_empty_pages():
    writer = PdfWriter()
    writer.add_blank_page(200, 200)
    page = writer.add_blank_page(200, 200)
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 20 100 Td (Page two) Tj ET")
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
    })
    page[NameObject("/Contents")] = stream
    output = BytesIO()
    writer.write(output)
    assert load_document("policy.pdf", output.getvalue()) == [Page("Page two", 2)]


def test_chunk_overlap_and_metadata():
    chunker = Chunker(10, 3, "text-embedding-3-small")
    a = " alpha" * 21
    chunks = chunker.split("key", [Page(a, 1), Page("Other page", 4)])
    assert [chunk.chunk_id for chunk in chunks] == [f"key:{index}" for index in range(4)]
    assert [chunk.page for chunk in chunks] == [1, 1, 1, 4]
    assert [len(chunk.text.split()) for chunk in chunks] == [10, 10, 7, 2]


def test_chunker_preserves_unicode_and_special_token_text():
    chunker = Chunker(16, 6, "text-embedding-3-small")
    chunks = chunker.split("key", [Page("👩‍💻 café 中文 <|endoftext|> " * 12)])
    assert len(chunks) > 1
    assert all("\ufffd" not in chunk.text for chunk in chunks)
    assert all(len(chunker.encoding.encode_ordinary(chunk.text)) <= 16 for chunk in chunks)
