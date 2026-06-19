import base64
import io
import json
import os
from streamlit_drawable_canvas import st_canvas
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Literal

import streamlit as st
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from litellm import completion

# Optional detector dependency
try:
    from ultralytics import YOLOWorld
except Exception:
    YOLOWorld = None


# ====================================================
# CONFIG
# ====================================================
load_dotenv()
st.set_page_config(layout="wide")
st.title("🛰️ تحلیلگر تصاویر ماهواره‌ای")

MAX_DISPLAY_SIZE = 512

ONTOLOGY = [
    "building",
    "farmland",
    "road",
    "water",
    "tree",
    "vehicle",
    "bridge",
    "river",
    "field",
    "crop_land",
    "bare_soil",
    "airport",
    "ship",
]


# ====================================================
# ROUTER SCHEMA
# ====================================================
class RouterDecision(BaseModel):
    task_type: Literal["counting", "grounding", "yes_no", "description", "mixed", "other"] = Field(
        description="Primary task type of the user's question."
    )
    needs_detection: bool = Field(
        description="True if object detection / grounding should run."
    )
    needs_visualization: bool = Field(
        default=False,
        description="True if the user asks to show, draw, highlight, mark, or outline objects."
    )
    needs_grounding: bool = Field(
        description="True if the answer depends on spatial location."
    )
    needs_yes_no: bool = Field(
        description="True if the question can be answered with yes/no."
    )
    needs_description: bool = Field(
        description="True if the user mainly wants a scene description."
    )
    target_classes: List[str] = Field(
        default_factory=list,
        description="Canonical object classes from the fixed ontology."
    )
    spatial_precision: Literal["high", "medium", "low", "none"] = Field(
        default="none",
        description="Needed spatial precision."
    )
    answer_format: Literal["json_internal", "text"] = Field(
        default="json_internal",
        description="Internal format choice for the pipeline."
    )


ROUTER_SYSTEM_PROMPT = f"""
You are a strict question router for satellite-image VQA.

Return only structured output matching the schema.

Your job:
1) classify the user's question into one primary task_type
2) decide whether detection/grounding/counting is needed
3) normalize the target object classes into canonical labels
4) return only data that matches the schema

Allowed canonical target classes:
{", ".join(ONTOLOGY)}

Routing rules:
- counting: when the user asks how many / number of / count
- grounding: when the user asks where / location / which part / direction / show / draw / highlight / mark / outline
- yes_no: when the user asks is there / are there / does it contain / exist
- description: when the user asks describe / what is in the image
- mixed: when more than one task is clearly needed
- other: only when none fits well

Important:
- If the question mentions an object class, place it in target_classes.
- If the question is about building/farmland/road/water/etc., set needs_detection=true.
- If the answer depends on location, set needs_grounding=true.
- If it is a yes/no question, set needs_yes_no=true.
- If it is a counting question, set task_type=counting and needs_detection=true.
- If the user asks to show, draw, mark, highlight, or outline objects, set needs_visualization=true.
- Use the fixed ontology only.
"""


# ====================================================
# DATA STRUCTURES
# ====================================================
@dataclass
class Detection:
    target_class: str
    bbox: List[float]   # [x1, y1, x2, y2] normalized
    score: float


# ====================================================
# HELPERS
# ====================================================
def encode_image(image: Image.Image, fmt="JPEG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def resize_for_display(image: Image.Image):
    w, h = image.size
    if max(w, h) <= MAX_DISPLAY_SIZE:
        return image, 1.0
    scale = MAX_DISPLAY_SIZE / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = image.resize((new_w, new_h), Image.LANCZOS)
    return resized, scale


def box_area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih

    a1 = box_area(box1)
    a2 = box_area(box2)
    return inter / (a1 + a2 - inter + 1e-9)


def classwise_nms(detections: List[Detection], iou_threshold: float = 0.3) -> List[Detection]:
    if not detections:
        return []

    output: List[Detection] = []
    by_class: Dict[str, List[Detection]] = {}
    for det in detections:
        by_class.setdefault(det.target_class, []).append(det)

    for cls, items in by_class.items():
        items = sorted(items, key=lambda d: d.score, reverse=True)
        kept = []
        while items:
            best = items.pop(0)
            kept.append(best)
            items = [d for d in items if iou(best.bbox, d.bbox) <= iou_threshold]
        output.extend(kept)

    return output


def normalize_target_classes(classes: List[str]) -> List[str]:
    out = []
    for c in classes:
        c = c.strip().lower().replace(" ", "_")
        if c in ONTOLOGY and c not in out:
            out.append(c)
    return out


def detect_visual_intent(question: str) -> bool:
    q = question.lower()
    keywords = [
        "show", "draw", "highlight", "mark",
        "outline", "visualize", "display"
    ]
    return any(k in q for k in keywords)


@st.cache_resource
def get_router_llm():
    return ChatOpenAI(model="gpt-4.1-mini", temperature=0)


@st.cache_resource
def get_detector():
    if YOLOWorld is None:
        return None
    return YOLOWorld("yolov8x-world.pt")


def fallback_router(question: str) -> RouterDecision:
    q = question.lower()
    target_classes = []

    if any(w in q for w in ["building", "house", "apartment", "roof", "residential", "home", "dwelling"]):
        target_classes.append("building")
    if any(w in q for w in ["farmland", "farm", "field", "crop", "agricultural"]):
        target_classes.append("farmland")
    if any(w in q for w in ["road", "street", "highway", "path"]):
        target_classes.append("road")
    if any(w in q for w in ["water", "river", "lake", "sea", "ocean"]):
        target_classes.append("water")
    if any(w in q for w in ["tree", "forest", "vegetation", "garden", "orchard"]):
        target_classes.append("tree")
    if any(w in q for w in ["vehicle", "car", "truck", "bus", "van"]):
        target_classes.append("vehicle")

    target_classes = normalize_target_classes(target_classes)
    wants_visual = any(w in q for w in ["show", "draw", "highlight", "mark", "outline", "visualize", "display"])

    if any(w in q for w in ["how many", "count", "number of", "total"]):
        return RouterDecision(
            task_type="counting",
            needs_detection=True,
            needs_visualization=wants_visual,
            needs_grounding=True,
            needs_yes_no=False,
            needs_description=False,
            target_classes=target_classes,
            spatial_precision="high",
            answer_format="json_internal",
        )

    if any(w in q for w in ["where", "location", "which part", "direction", "north", "south", "east", "west"]):
        return RouterDecision(
            task_type="grounding",
            needs_detection=True,
            needs_visualization=wants_visual,
            needs_grounding=True,
            needs_yes_no=False,
            needs_description=False,
            target_classes=target_classes,
            spatial_precision="high",
            answer_format="json_internal",
        )

    if any(w in q for w in ["is there", "are there", "does it contain", "any ", "exist"]):
        return RouterDecision(
            task_type="yes_no",
            needs_detection=bool(target_classes),
            needs_visualization=wants_visual,
            needs_grounding=False,
            needs_yes_no=True,
            needs_description=False,
            target_classes=target_classes,
            spatial_precision="high",
            answer_format="json_internal",
        )

    return RouterDecision(
        task_type="description",
        needs_detection=False,
        needs_visualization=wants_visual,
        needs_grounding=False,
        needs_yes_no=False,
        needs_description=True,
        target_classes=target_classes,
        spatial_precision="none",
        answer_format="text",
    )


def route_question(question: str) -> RouterDecision:
    try:
        llm = get_router_llm()
        structured = llm.with_structured_output(RouterDecision, method="json_schema")
        result = structured.invoke([
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=question),
        ])
        result.target_classes = normalize_target_classes(result.target_classes)
        return result
    except Exception:
        return fallback_router(question)


def build_vlm_system_prompt(route: RouterDecision) -> str:
    base = (
        "You are an expert remote sensing image interpreter. "
        "You will receive a satellite image, a user question, and optional detector metadata. "
        "Detector metadata is auxiliary evidence, not ground truth. "
        "Detector outputs may contain false positives, false negatives, duplicate detections, and localization errors. "
        "Always inspect the image directly. "
        "Use detector outputs only to assist counting, localization, and object referencing. "
        "If detector outputs disagree with the image, trust the image more."
    )

    if route.task_type == "counting":
        base += " The user wants a count. Give the best final count clearly and mention uncertainty if needed."
    elif route.task_type == "grounding":
        base += " The user wants spatial location. Use directional terms such as north, northeast, center, southwest, etc."
    elif route.task_type == "yes_no":
        base += " The user wants a yes/no answer. Start with yes or no, then add one short justification."
    elif route.task_type == "description":
        base += " The user wants a short scene description."
    elif route.task_type == "mixed":
        base += " The user likely needs both localization and semantic reasoning."

    return base


def detections_to_context(detections: List[Detection], top_k: int = 20):
    items = []
    for i, det in enumerate(sorted(detections, key=lambda d: d.score, reverse=True)[:top_k], start=1):
        items.append({
            "id": i,
            "target_class": det.target_class,
            "bbox": [round(v, 4) for v in det.bbox],
            "score": round(float(det.score), 3),
        })
    return items


def draw_detections(image: Image.Image, detections: List[Detection], label_mode: str = "short") -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    colors = {
        "building": (255, 0, 0),
        "farmland": (0, 128, 0),
        "road": (255, 165, 0),
        "water": (0, 0, 255),
        "tree": (34, 139, 34),
        "vehicle": (255, 255, 0),
        "bridge": (255, 0, 255),
        "river": (0, 128, 255),
        "field": (0, 180, 0),
        "crop_land": (0, 200, 100),
        "bare_soil": (150, 100, 50),
        "airport": (120, 120, 255),
        "ship": (80, 80, 200),
    }

    w, h = img.size
    for det in detections:
        x1 = int(det.bbox[0] * w)
        y1 = int(det.bbox[1] * h)
        x2 = int(det.bbox[2] * w)
        y2 = int(det.bbox[3] * h)
        color = colors.get(det.target_class, (255, 0, 0))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        if label_mode == "short":
            label = f"{det.target_class}:{det.score:.2f}"
        else:
            label = det.target_class

        tb = draw.textbbox((x1, max(0, y1 - 18)), label, font=font)
        text_w = tb[2] - tb[0]
        text_h = tb[3] - tb[1]
        y_text = max(0, y1 - text_h - 2)
        draw.rectangle([x1, y_text, x1 + text_w + 4, y_text + text_h + 2], fill=color)
        draw.text((x1 + 2, y_text + 1), label, fill=(0, 0, 0), font=font)

    return img


def choose_target_classes(route: RouterDecision) -> List[str]:
    return normalize_target_classes(route.target_classes)


def run_detector_on_image(image: Image.Image, classes: List[str]) -> List[Detection]:
    detector = get_detector()
    if detector is None:
        raise RuntimeError(
            "Ultralytics is not installed or YOLOWorld could not be imported. "
            "Install ultralytics and make sure YOLO-World weights are available."
        )

    classes = normalize_target_classes(classes)
    if not classes:
        return []

    detector.set_classes(classes)

    results = detector.predict(
        source=image,
        conf=0.20,
        iou=0.45,
        imgsz=640,
        verbose=False,
    )

    detections: List[Detection] = []
    w, h = image.size

    for result in results:
        if result.boxes is None:
            continue

        boxes_xyxy = result.boxes.xyxy.cpu().tolist()
        scores = result.boxes.conf.cpu().tolist()
        cls_ids = result.boxes.cls.cpu().tolist()

        for box, score, cls_idx in zip(boxes_xyxy, scores, cls_ids):
            cls_idx = int(cls_idx)
            target_class = classes[cls_idx] if 0 <= cls_idx < len(classes) else classes[0]
            x1, y1, x2, y2 = box

            bbox = [
                max(0.0, min(1.0, x1 / w)),
                max(0.0, min(1.0, y1 / h)),
                max(0.0, min(1.0, x2 / w)),
                max(0.0, min(1.0, y2 / h)),
            ]

            detections.append(
                Detection(
                    target_class=target_class,
                    bbox=bbox,
                    score=float(score),
                )
            )

    return detections


def summarize_detections(detections: List[Detection]) -> str:
    if not detections:
        return "No detections were found."

    counts = Counter(d.target_class for d in detections)
    return " | ".join(f"{cls}: {count}" for cls, count in sorted(counts.items(), key=lambda x: x[0]))


def build_detection_context_block(route: RouterDecision, detections: List[Detection]) -> str:
    payload = {
        "router": route.model_dump(),
        "detection_summary": summarize_detections(detections),
        "detections": detections_to_context(detections),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_vlm_messages(image: Image.Image, question: str, route: RouterDecision, detections: List[Detection]) -> list:
    image_b64 = encode_image(image)
    context = build_detection_context_block(route, detections)
    system_prompt = build_vlm_system_prompt(route)

    user_text = f"""
Question:
{question}

Detector metadata:
{context}

Instructions:
- Use the image as the primary evidence.
- Use detector outputs as helper metadata.
- Verify detector results against the image before answering.
- If the detector is uncertain or noisy, mention approximate language.
- For counting, give the best final count and mention if it is approximate.
- For grounding, mention spatial directions if useful.
"""

    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
                {"type": "text", "text": user_text},
            ],
        },
    ]


def answer_with_vlm(image: Image.Image, question: str, route: RouterDecision, detections: List[Detection]) -> str:
    messages = build_vlm_messages(image, question, route, detections)
    response = completion(
        model="openai/claude-opus-4-7",
        messages=messages,
        max_tokens=600,
    )
    return response.choices[0].message.content.strip()


# ====================================================
# APP
# ====================================================
uploaded_file = st.file_uploader(
    "Upload Satellite Image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    original_image = Image.open(uploaded_file).convert("RGB")
    orig_w, orig_h = original_image.size

    display_image, scale = resize_for_display(original_image)
    disp_w, disp_h = display_image.size

    st.subheader("ناحیه مورد نظرتان را انتخاب کنید")

    canvas_result = st_canvas(
        fill_color="rgba(255,0,0,0.2)",
        stroke_width=2,
        stroke_color="#ff0000",
        background_image=display_image,
        update_streamlit=True,
        height=disp_h,
        width=disp_w,
        drawing_mode="rect",
        key="canvas",
    )

    selected_box = None

    if canvas_result.json_data:
        objects = canvas_result.json_data.get("objects", [])
        if len(objects) > 0:
            rect = objects[0]
            left = rect["left"]
            top = rect["top"]
            width = rect["width"] * rect.get("scaleX", 1)
            height = rect["height"] * rect.get("scaleY", 1)

            x1 = int(left / scale)
            y1 = int(top / scale)
            x2 = int((left + width) / scale)
            y2 = int((top + height) / scale)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(orig_w, x2)
            y2 = min(orig_h, y2)

            selected_box = (x1, y1, x2, y2)
            st.success(f"Selected Area: ({x1}, {y1}) → ({x2}, {y2})")

    st.subheader("سوالتان را بپرسید")
    question = st.text_input(
        "Question",
        placeholder="What can you see in this region?"
    )

    if st.button("تحلیل منطقه"):
        if selected_box is None:
            st.error("Please draw a rectangle.")
            st.stop()

        if not question.strip():
            st.error("Please enter a question.")
            st.stop()

        route = route_question(question)
        crop = original_image.crop(selected_box)

        detections: List[Detection] = []
        detector_error = None

        if route.needs_detection:
            target_classes = choose_target_classes(route)
            if target_classes:
                try:
                    detections = run_detector_on_image(crop, target_classes)
                    detections = classwise_nms(detections, iou_threshold=0.30)
                    detections = sorted(detections, key=lambda d: d.score, reverse=True)
                except Exception as e:
                    detector_error = str(e)
                    detections = []

        if route.needs_visualization and detections:
            response_image = draw_detections(crop, detections)
        else:
            response_image = crop

        with st.spinner("در حال انجام تحلیل"):
            try:
                answer = answer_with_vlm(response_image, question, route, detections)
            except Exception as e:
                answer = f"API Error: {e}"

        st.image(response_image, caption="ناحیه انتخاب شده", )

        with st.expander("Router / Detector debug", expanded=False):
            st.subheader("Router")
            st.json(route.model_dump())

            if detector_error:
                st.warning(f"Detector error: {detector_error}")

            if detections:
                st.subheader("Detector output")
                st.json([
                    {
                        "target_class": d.target_class,
                        "bbox": [round(v, 4) for v in d.bbox],
                        "score": round(d.score, 3),
                    }
                    for d in detections
                ])

        st.subheader("پاسخ")
        st.write(answer)