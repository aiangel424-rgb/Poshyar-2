import io, os
from functools import lru_cache
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image, ImageOps, ImageFilter
from transformers import pipeline

MODEL_ID = os.getenv("SEGMENTATION_MODEL", "fashn-ai/fashn-human-parser")
app = FastAPI(title="Poshyar AI Clothing Mask API", version="1.0")
origins = [x.strip() for x in os.getenv("ALLOWED_ORIGINS", "*").split(",") if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False,
                   allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"])

@lru_cache(maxsize=1)
def model():
    return pipeline("image-segmentation", model=MODEL_ID, device=-1)

@app.get("/")
def root():
    return {"service":"poshyar-clothing-mask","status":"ready",
            "model":MODEL_ID,"health":"/health","endpoint":"/segment",
            "response":"PNG mask; white=clothing, black=protected"}

@app.get("/health")
def health():
    return {"ok":True,"model":MODEL_ID}

@app.post("/segment")
async def segment(image: UploadFile = File(...), format: str = Form("binary-mask-png")):
    if format != "binary-mask-png":
        raise HTTPException(400, "format must be binary-mask-png")
    raw = await image.read(12 * 1024 * 1024 + 1)
    if len(raw) > 12 * 1024 * 1024:
        raise HTTPException(413, "Image exceeds 12 MB")
    try:
        src = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not decode image")
    if src.width * src.height > 16_000_000:
        raise HTTPException(413, "Image exceeds 16 megapixels")
    try:
        results = model()(src)
    except Exception as exc:
        raise HTTPException(503, f"Model failed ({type(exc).__name__})")
    # Review model labels and license before production use.
    garment_labels = {"top", "dress", "skirt", "pants", "belt", "scarf"}
    combined = Image.new("L", src.size, 0)
    for item in results:
        label = str(item.get("label", "")).lower().strip().replace("-", "_")
        if label not in garment_labels or item.get("mask") is None:
            continue
        m = item["mask"]
        if not isinstance(m, Image.Image):
            try:
                import numpy as np
                m = Image.fromarray(np.asarray(m))
            except Exception:
                continue
        m = m.convert("L").resize(src.size, Image.Resampling.NEAREST)
        b = m.point(lambda p: 255 if p >= 128 else 0)
        combined = Image.frombytes("L", src.size,
            bytes(max(a, c) for a, c in zip(combined.tobytes(), b.tobytes())))
    combined = combined.filter(ImageFilter.MedianFilter(size=3))
    out = io.BytesIO()
    combined.save(out, format="PNG")
    return Response(out.getvalue(), media_type="image/png",
                    headers={"Cache-Control":"no-store",
                             "X-Mask-Convention":"white-clothing-black-protected"})
