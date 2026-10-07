import asyncio

from app.db.session import session_factory
from app.services.document_processing_service import document_processing_service


async def run_forever() -> None:
    factory = session_factory()
    while True:
        async with factory() as session:
            processed = await document_processing_service.process_next_pending(session)
        await asyncio.sleep(1 if processed else 5)


def main() -> None:
    asyncio.run(run_forever())


if __name__ == "__main__":
    main()
