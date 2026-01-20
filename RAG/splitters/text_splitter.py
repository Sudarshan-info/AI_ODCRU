from config.settings import CHUNK_SIZE, CHUNK_OVERLAP


def split_documents(documents):
    chunks = []

    for doc in documents:
        start = 0
        text_length = len(doc)

        while start < text_length:
            end = start + CHUNK_SIZE
            chunk = doc[start:end]
            chunks.append(chunk)

            # move start pointer with overlap
            start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks
