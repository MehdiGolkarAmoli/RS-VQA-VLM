import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
from litellm import completion
from dotenv import load_dotenv
import base64
import io
import os
from typing import List, Literal
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------
load_dotenv()

st.set_page_config(layout="wide")
st.title("🛰️ تحلیلگر تصاویر ماهواره‌ای")

MAX_DISPLAY_SIZE = 512

# ----------------------------------------------------
# ROUTER SCHEMA
# ----------------------------------------------------
class RouterDecision(BaseModel):
    task_type: Literal["counting", "grounding", "yes_no", "description", "mixed", "other"] = Field(
        description="Primary task type of the user's question."
    )
    needs_detection: bool = Field(
        description="True if the next stage should run object detection or localization."
    )
    needs_grounding: bool = Field(
        description="True if the answer depends on spatial location or object grounding."
    )
    needs_yes_no: bool = Field(
        description="True if the question can be answered with yes/no."
    )
    needs_description: bool = Field(
        description="True if the user mainly just wants a scene description."
    )
    target_classes: List[str] = Field(
        default_factory=list,
        description="Canonical object classes that should be detected in the next phase."
    )
    spatial_precision: Literal["high"] = Field(
        description="How precise the spatial answer needs to be."
    )
    answer_format: Literal["json_internal", "text"] = Field(
        default="json_internal",
        description="Internal answer format choice for later pipeline steps."
    )

ROUTER_SYSTEM_PROMPT = """
You are a strict question router for satellite-image VQA.

Your job:
1) classify the user's question into one primary task_type
2) decide whether detection/grounding/counting is needed
3) normalize the target object classes into canonical labels
4) return only data that matches the schema

Allowed canonical target classes should be short and practical, such as:
building, farmland, road, water, tree, vehicle, bridge, river, field, crop_land, bare_soil, industrial_area, residential_area, airport, ship

Routing rules:
- counting: when the user asks how many / number of / count
- grounding: when the user asks where / location / which part / direction
- yes_no: when the user asks is there / are there / does it contain
- description: when the user asks describe / what is in the image
- mixed: when more than one task is clearly needed
- other: only when none fits well

Important:
- If the question mentions an object class, place it in target_classes.
- If the question is about building/farmland/road/water/etc., set needs_detection=true.
- If the answer depends on location, set needs_grounding=true.
- If it is a yes/no question, set needs_yes_no=true.
- If it is a counting question, set task_type=counting and needs_detection=true.
- Return a clean, minimal structured output.
"""

# ----------------------------------------------------
# HELPERS
# ----------------------------------------------------
def encode_image(image: Image.Image, fmt="JPEG"):
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def resize_for_display(image):
    w, h = image.size

    if max(w, h) <= MAX_DISPLAY_SIZE:
        return image, 1.0

    scale = MAX_DISPLAY_SIZE / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = image.resize((new_w, new_h), Image.LANCZOS)

    return resized, scale


@st.cache_resource
def get_router_llm():
    return ChatOpenAI(
        model="gpt-4.1",
        temperature=0,
    )


def fallback_router(question: str) -> RouterDecision:
    q = question.lower()

    target_classes = []
    if any(w in q for w in ["building", "house", "apartment", "roof", "residential"]):
        target_classes.append("building")
    if any(w in q for w in ["farmland", "farm", "field", "crop", "agricultural"]):
        target_classes.append("farmland")
    if any(w in q for w in ["road", "street", "highway", "path"]):
        target_classes.append("road")
    if any(w in q for w in ["water", "river", "lake", "sea", "ocean"]):
        target_classes.append("water")
    if any(w in q for w in ["tree", "forest", "vegetation", 'garden', 'orchard']):
        target_classes.append("tree")
    if any(w in q for w in ["vehicle", "car", "truck", "bus", "van"]):
        target_classes.append("vehicle")

    if any(w in q for w in ["how many", "count", "number of", "total"]):
        return RouterDecision(
            task_type="counting",
            needs_detection=True,
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
            needs_grounding=True,
            needs_yes_no=False,
            needs_description=False,
            target_classes=target_classes ,
            spatial_precision="high",
            answer_format="json_internal",
        )

    if any(w in q for w in ["is there", "are there", "does it contain", "any ", "exist"]):
        return RouterDecision(
            task_type="yes_no",
            needs_detection=True if target_classes else False,
            needs_grounding=False,
            needs_yes_no=True,
            needs_description=False,
            target_classes=target_classes,
            spatial_precision= "high",
            answer_format="json_internal",
        )

    return RouterDecision(
        task_type="description",
        needs_detection=False,
        needs_grounding=False,
        needs_yes_no=False,
        needs_description=True,
        target_classes=target_classes,
        spatial_precision="high",
        answer_format="text",
    )


def route_question(question: str) -> RouterDecision:
    try:
        llm = get_router_llm()
        structured_router = llm.with_structured_output(RouterDecision, method="json_schema")
        return structured_router.invoke([
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=question),
        ])
    except Exception:
        return fallback_router(question)


def build_vlm_system_prompt(route: RouterDecision) -> str:
    base = (
        "You are an expert satellite image analyst."
        "Answer only using what is visible inside the image."
    )

    if route.task_type == "counting":
        base += " The user wants a count. Count carefully and state the number first."
    elif route.task_type == "grounding":
        base += " The user wants spatial location. Use directional terms such as north, northeast, center, southwest, etc."
    elif route.task_type == "yes_no":
        base += " The user wants a yes/no answer. Start with yes or no, then add one short justification."
    elif route.task_type == "description":
        base += " The user wants a short scene description."
    elif route.task_type == "mixed":
        base += " The user likely needs both localization and semantic understanding."

    return base


# ----------------------------------------------------
# IMAGE UPLOAD
# ----------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Satellite Image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:

    original_image = Image.open(uploaded_file).convert("RGB")
    orig_w, orig_h = original_image.size

    display_image, scale = resize_for_display(original_image)
    disp_w, disp_h = display_image.size

    st.subheader("ناحیه مورد نظرتان راانتخاب کنید")

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

    # ------------------------------------------------
    # READ RECTANGLE
    # ------------------------------------------------
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

    # ------------------------------------------------
    # QUESTION
    # ------------------------------------------------
    st.subheader("سوالتان را بپرسید")

    question = st.text_input(
        "Question",
        placeholder="What can you see in this region?"
    )

    # ------------------------------------------------
    # ANALYZE
    # ------------------------------------------------
    if st.button("تحلیل منطقه"):

        if selected_box is None:
            st.error("Please draw a rectangle.")
            st.stop()

        if not question.strip():
            st.error("Please enter a question.")
            st.stop()

        # 1) ROUTE THE QUESTION FIRST
        route = route_question(question)

        with st.expander("Router output", expanded=True):
            st.json(route.model_dump())

        crop = original_image.crop(selected_box)
        crop_b64 = encode_image(crop)

        # 2) BUILD A BETTER PROMPT FOR THE VLM BASED ON THE ROUTER
        vlm_system_prompt = build_vlm_system_prompt(route)

        target_text = ", ".join(route.target_classes) if route.target_classes else "none"

        user_text = f"""
Question: {question}

Router task: {route.task_type}
Need detection: {route.needs_detection}
Need grounding: {route.needs_grounding}
Need yes/no: {route.needs_yes_no}
Need description: {route.needs_description}
Target classes: {target_text}
Spatial precision: {route.spatial_precision}

Answer the user's question using only what is visible in the image.
"""

        messages = [
            {
                "role": "system",
                "content": vlm_system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{crop_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": user_text
                    }
                ]
            }
        ]

        with st.spinner("در حال انجام تحلیل"):
            try:
                response = completion(
                    model="openai/claude-opus-4-7",
                    messages=messages,
                    max_tokens=500,
                )
                answer = response.choices[0].message.content
            except Exception as e:
                answer = f"API Error: {e}"

        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(
                crop,
                caption="ناحیه انتخاب شده",
            )

        with col2:
            st.subheader("پاسخ")
            st.write(answer)