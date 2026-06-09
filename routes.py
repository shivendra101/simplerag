from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from ai_service import AIservice
from dependencies import get_ai_service, get_ingest_data_service
import asyncio
import time

from rag.ingest_data_service import IngestDataService

router = APIRouter()

REQUEST_SEMAPHORE = asyncio.Semaphore(10)
REQUEST_TIMEOUT = 30

class QuestionRequest(BaseModel):
    questions: list[str]

class ResponseItem(BaseModel):
    question: str
    answer: str | None = None
    error: str | None = None

class GetResponseOutput(BaseModel):
    status: str
    data: list[ResponseItem]
    metadata: dict

# not data returned for this endpoint, just a status message
class IngestResponseOutput(BaseModel):
    status: str
    data: dict
    metadata: dict

class IngestRequest(BaseModel):
    raw_string: str

async def process_question(question: str, ai_service: AIservice):
    async with REQUEST_SEMAPHORE:
        try:
            answer = await asyncio.wait_for(
                ai_service.ask_question(question),
                timeout=REQUEST_TIMEOUT
            )
            return {"question": question, "answer": answer}
        except asyncio.TimeoutError:
            return {"question": question, "error": f"timeout after {REQUEST_TIMEOUT}s"}
        except Exception as e:
            return {"question": question, "error": str(e)}



@router.post("/get_response", response_model=GetResponseOutput)
async def get_response(
    request: QuestionRequest,
    ai_service: AIservice = Depends(get_ai_service)
):
    if not request.questions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="questions list cannot be empty"
        )

    start_time = time.perf_counter()

    tasks = [process_question(q, ai_service) for q in request.questions]
    results = await asyncio.gather(*tasks)

    elapsed = time.perf_counter() - start_time

    success_count = sum(1 for r in results if r.get("answer") is not None)
    error_count = len(results) - success_count

    return GetResponseOutput(
        status="success" if error_count == 0 else "partial",
        data=[ResponseItem(**r) for r in results],
        metadata={
            "total_questions": len(results),
            "successful": success_count,
            "failed": error_count,
            "elapsed_seconds": round(elapsed, 3),
        }
    )

@router.post("/ingest_data")
async def ingest_data(
    request: IngestRequest,
    ingest_data_service: IngestDataService = Depends(get_ingest_data_service),
):
    await ingest_data_service.ingest_text(request.raw_string)
    pass

@router.post("/get_chunk")
async def get_chunk(
    request: IngestRequest,
    ingest_data_service: IngestDataService = Depends(get_ingest_data_service),
):
    return await ingest_data_service.search_chunks(request.raw_string)