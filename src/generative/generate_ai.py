"""
src/generative/generate_ai.py
Generative AI features for graph-imgrag.

Uses Claude (Anthropic) to:
  1. Generate rich image captions from OCR text + category metadata
  2. Answer questions about retrieved image sets
  3. Suggest related search queries
  4. Summarise a collection of retrieved results

Requires: pip install anthropic
API key:  set ANTHROPIC_API_KEY environment variable
"""

import os
import sys
import base64
import json
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../../..")))

from src.utils.helpers import get_logger

log = get_logger("GenerativeAI")

_client = None


def _get_client():
    global _client
    if _client is None:
        try:
            import anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY environment variable not set.\n"
                    "  Set it with:  export ANTHROPIC_API_KEY=sk-ant-..."
                )
            _client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError(
                "anthropic package not installed.\n"
                "  Install it with:  pip install anthropic"
            )
    return _client


# ── Image → base64 helper ─────────────────────────────────────────────────────

def _image_to_b64(path: str) -> tuple[str, str]:
    """Return (base64_data, media_type) for an image file."""
    ext = Path(path).suffix.lower()
    media_type_map = {
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png":  "image/png",
        ".gif":  "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_type_map.get(ext, "image/jpeg")
    with open(path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    return data, media_type


# ── Core generation functions ─────────────────────────────────────────────────

def generate_image_caption(
    image_path: str,
    ocr_text: str = "",
    category: str = "",
    style: str = "descriptive",
) -> str:
    """
    Generate a rich natural-language caption for a single image.

    Args:
        image_path : path to the image file
        ocr_text   : OCR text already extracted from the image
        category   : COCO supercategory label
        style      : 'descriptive' | 'short' | 'poetic' | 'technical'

    Returns:
        Generated caption string
    """
    client = _get_client()

    style_instructions = {
        "descriptive": "Write a clear, detailed 2-3 sentence description of the image. Mention key objects, actions, and setting.",
        "short":       "Write a single concise sentence caption, like a newspaper photo caption.",
        "poetic":      "Write a 2-3 sentence lyrical, evocative description that captures the mood and atmosphere.",
        "technical":   "Write a technical description noting objects, their spatial relationships, lighting conditions, and any text visible.",
    }
    instruction = style_instructions.get(style, style_instructions["descriptive"])

    context_parts = []
    if ocr_text:
        context_parts.append(f"Text visible in image: \"{ocr_text}\"")
    if category:
        context_parts.append(f"Image category: {category}")
    context = "\n".join(context_parts)

    system_prompt = (
        f"You are an expert image analyst. {instruction} "
        f"Be specific and grounded in what is actually visible. "
        f"Do not hallucinate details that are not present."
    )

    user_content = []

    # Add image if it exists and is readable
    if os.path.exists(image_path):
        try:
            img_data, media_type = _image_to_b64(image_path)
            user_content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": img_data,
                },
            })
        except Exception as e:
            log.warning(f"Could not load image for captioning: {e}")

    text_prompt = "Please generate a caption for this image."
    if context:
        text_prompt += f"\n\nAdditional context:\n{context}"

    user_content.append({"type": "text", "text": text_prompt})

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=300,
        system=system_prompt,
        messages=[{"role": "user", "content": user_content}],
    )

    return response.content[0].text.strip()


def generate_collection_summary(
    results: list,
    query: str,
    ocr_results: dict = None,
) -> str:
    """
    Generate a natural-language summary of a set of retrieved images.

    Args:
        results    : list of search result dicts from search()
        query      : the original search query
        ocr_results: optional dict {path: ocr_text}

    Returns:
        Summary paragraph string
    """
    client = _get_client()

    # Build context from results
    items = []
    for r in results[:5]:  # limit context
        ocr = ""
        if ocr_results:
            ocr = ocr_results.get(r["path"], "").strip()
        item_str = (
            f"  - Rank #{r['rank']}: {r['file']} "
            f"(category: {r['category']}, similarity: {r['similarity']:.3f})"
        )
        if ocr:
            item_str += f"\n    Visible text: \"{ocr[:100]}\""
        items.append(item_str)

    context = "\n".join(items)

    prompt = (
        f"A user searched for: \"{query}\"\n\n"
        f"The top retrieved images are:\n{context}\n\n"
        f"Write a 2-3 sentence natural language summary of what these images "
        f"show and why they are relevant to the query. Be concise and informative."
    )

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.content[0].text.strip()


def suggest_related_queries(
    query: str,
    results: list,
    categories: list = None,
) -> list[str]:
    """
    Suggest 5 related search queries based on the current query and results.

    Returns:
        list of 5 query strings
    """
    client = _get_client()

    cat_context = ""
    if categories:
        cat_context = f"\nAvailable categories: {', '.join(categories)}"

    result_context = ""
    if results:
        cats = list({r["category"] for r in results[:5]})
        result_context = f"\nTop result categories: {', '.join(cats)}"

    prompt = (
        f"A user searched for: \"{query}\"\n"
        f"{result_context}{cat_context}\n\n"
        f"Suggest 5 related search queries that the user might find interesting. "
        f"Make them specific and varied. "
        f"Return ONLY a JSON array of 5 strings, nothing else.\n"
        f"Example: [\"query 1\", \"query 2\", \"query 3\", \"query 4\", \"query 5\"]"
    )

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()
    try:
        # Strip markdown fences if present
        text = text.replace("```json", "").replace("```", "").strip()
        suggestions = json.loads(text)
        return suggestions[:5]
    except Exception:
        # Fallback: extract quoted strings
        import re
        matches = re.findall(r'"([^"]+)"', text)
        return matches[:5] if matches else [query + " outdoor", query + " close up"]


def answer_question_about_image(
    image_path: str,
    question: str,
    ocr_text: str = "",
    category: str = "",
) -> str:
    """
    Answer a free-form question about a specific image.

    Args:
        image_path : path to the image
        question   : user's question
        ocr_text   : OCR text from the image
        category   : image category

    Returns:
        Answer string
    """
    client = _get_client()

    context_parts = []
    if ocr_text:
        context_parts.append(f"Text in image: \"{ocr_text}\"")
    if category:
        context_parts.append(f"Category: {category}")
    context = "\n".join(context_parts)

    user_content = []

    if os.path.exists(image_path):
        try:
            img_data, media_type = _image_to_b64(image_path)
            user_content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": img_data,
                },
            })
        except Exception as e:
            log.warning(f"Could not load image for Q&A: {e}")

    text_prompt = f"Question: {question}"
    if context:
        text_prompt = f"{context}\n\n{text_prompt}"

    user_content.append({"type": "text", "text": text_prompt})

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=300,
        system="You are a helpful assistant answering questions about images. Be concise and accurate. Only describe what you can actually see.",
        messages=[{"role": "user", "content": user_content}],
    )

    return response.content[0].text.strip()


def generate_dataset_insights(stats: dict, ocr_sample: dict = None) -> str:
    """
    Generate high-level insights about the indexed image dataset.

    Args:
        stats      : dict from database.get_stats()
        ocr_sample : optional sample of {path: ocr_text} for analysis

    Returns:
        Insights paragraph
    """
    client = _get_client()

    cats_str = "\n".join(
        f"  - {cat}: {n} images"
        for cat, n in (stats.get("categories") or {}).items()
    )

    sample_text = ""
    if ocr_sample:
        samples = list(ocr_sample.items())[:10]
        sample_lines = [f"  - {Path(p).name}: \"{t[:60]}\"" for p, t in samples if t.strip()]
        if sample_lines:
            sample_text = "\nSample OCR text from images:\n" + "\n".join(sample_lines[:5])

    prompt = (
        f"Here is information about an indexed image dataset:\n\n"
        f"Total images: {stats.get('total_images', 0)}\n"
        f"Total graph edges: {stats.get('total_edges', 0)}\n"
        f"Categories:\n{cats_str}"
        f"{sample_text}\n\n"
        f"Write 3-4 sentences of interesting insights about this dataset. "
        f"Comment on the distribution, connectivity, and what kinds of searches "
        f"would work best. Be analytical and helpful."
    )

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=250,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.content[0].text.strip()


def check_api_key() -> tuple[bool, str]:
    """
    Check if the Anthropic API key is configured and valid.
    Returns (is_valid, message).
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return False, "ANTHROPIC_API_KEY environment variable not set."
    if not api_key.startswith("sk-ant-"):
        return False, "API key format looks incorrect (should start with sk-ant-)."
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        # Quick test call
        client.messages.create(
            model="claude-opus-4-6",
            max_tokens=5,
            messages=[{"role": "user", "content": "hi"}],
        )
        return True, "API key is valid."
    except ImportError:
        return False, "anthropic package not installed. Run: pip install anthropic"
    except Exception as e:
        return False, f"API error: {str(e)[:100]}"