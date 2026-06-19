import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
from litellm import completion
from dotenv import load_dotenv
import base64
import io

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------
load_dotenv()

st.set_page_config(layout="wide")
st.title("🛰️ تحلیلگر تصاویر ماهواره‌ای")

MAX_DISPLAY_SIZE = 512

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
        background_image=display_image,  # PIL image
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

            st.success(
                f"Selected Area: ({x1}, {y1}) → ({x2}, {y2})"
            )

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

        crop = original_image.crop(selected_box)

        crop_b64 = encode_image(crop)

        messages = [
            {
                "role": "system",
                "content":
                (
                    "You are an expert satellite image analyst. "
                    "Answer only using what is visible "
                    "inside the image."
                )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url":
                            f"data:image/jpeg;base64,{crop_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": question
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