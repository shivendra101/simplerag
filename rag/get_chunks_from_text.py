
def get_chunks_from_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    """
    Splits the input text into chunks of a specified size with a specified overlap.
    
    """

    words = text.split()

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = start + chunk_size - overlap

    return chunks