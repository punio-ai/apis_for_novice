from fastapi import Query, Header, HTTPException, status


def pagination(skip: int = Query(0, ge=0), limit: int = Query(10, ge=1, le=100)):
    """
    Reusable dependency for pagination.
    """
    return {"skip": skip, "limit": limit}


def get_api_key(x_api_key: str | None = Header(None)):
    """
    Mock API Key dependency. 
    Header(None) tells FastAPI to look for 'x-api-key' in the HTTP headers.
    """
    if x_api_key != "super-secret-key-2026":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key"
        )
    return x_api_key
