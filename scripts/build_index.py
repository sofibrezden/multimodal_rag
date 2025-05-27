import json
import argparse
import numpy as np
import faiss
from tqdm import tqdm
import torch
from io import BytesIO
from PIL import Image
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration
from scripts.utils import load_model, load_jsonl

# configs
TEXT_MODEL_NAME = "all-MiniLM-L6-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# load BLIP model for image captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(DEVICE)


# generate captions for images using BLIP
def blip_caption_from_url(image_url: str) -> str:
    try:
        response = requests.get(image_url, timeout=15)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        return f"[Error loading image: {e}]"

    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    out = model.generate(
        **inputs,
        max_length=100,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


# build the FAISS index with text and image data
def build_textonly_index(input_jsonl: str, index_path: str, meta_path: str):
    articles = load_jsonl(input_jsonl)

    all_entries = []
    for art in articles:
        entry = {
            "type": "text",
            "article_id": art.get("id"),
            "source": "article",
            "raw_text": f"{art.get('title', '')}\n{art.get('subtitle', '')}\n{art.get('text', '')}",
            "meta": {"title": art.get("title", ""), "date": art.get("date", ""), "url": art.get("url", "")}
        }
        all_entries.append(entry)

    for art in articles:
        for url in art.get("media_urls", []):
            if not url.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".gif")):
                continue
            entry = {
                "type": "image",
                "article_id": art.get("id"),
                "source": "image",
                "raw_text": url,
                "meta": {"image_url": url}
            }
            all_entries.append(entry)

    for entry in tqdm(all_entries, desc="Summarizing images"):
        if entry["type"] == "image":
            try:
                summary = blip_caption_from_url(entry["raw_text"])
                entry["raw_text"] = summary
            except Exception as ex:
                entry["raw_text"] = ""
                print("Error summarizing image:", ex)

    # generate embeddings for all entries (text and image captions)
    model_text = load_model(TEXT_MODEL_NAME, device=DEVICE)
    embeddings = model_text.encode([e["raw_text"] for e in all_entries], show_progress_bar=True)

    emb_matrix = np.stack(embeddings)
    dim = emb_matrix.shape[1]
    # create FAISS index
    index = faiss.IndexFlatL2(dim)
    index.add(emb_matrix)
    faiss.write_index(index, index_path)

    # save metadata
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(all_entries)} entries: {index_path}, {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS index using BLIP captions for images")
    parser.add_argument("--input", required=True, help="Path to the JSONL articles file")
    parser.add_argument("--index_path", required=True, help="Path to save FAISS index")
    parser.add_argument("--meta_path", required=True, help="Path to save metadata JSON")
    args = parser.parse_args()

    build_textonly_index(args.input, args.index_path, args.meta_path)
