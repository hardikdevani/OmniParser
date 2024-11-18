import pyautogui
import time
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from PIL import Image
import os
from utils import get_som_labeled_img, check_ocr_box, get_caption_model_processor, get_yolo_model
import torch
from ultralytics import YOLO
from PIL import Image
from rich import print
device = 'cuda'

som_model = get_yolo_model(model_path='weights/icon_detect/best.pt')

som_model.to(device)
# print('model to {}'.format(device))

# two choices for caption model: fine-tuned blip2 or florence2

# caption_model_processor = get_caption_model_processor(model_name="blip2", model_name_or_path="weights/icon_caption_blip2", device=device)
caption_model_processor = get_caption_model_processor(model_name="florence2", model_name_or_path="weights/icon_caption_florence", device=device)


# cnt = 0
# image_path = 'imgs/image.png'
# # image_path = 'imgs/windows_home.png'
# # image_path = 'imgs/windows_multitab.png'

# BOX_TRESHOLD = 0.03
# def label_parcer(image_path):
#     image = Image.open(image_path)
#     image_rgb = image.convert('RGB')
#     box_overlay_ratio = image.size[0] / 3200
#     draw_bbox_config = {
#         'text_scale': 0.8 * box_overlay_ratio,
#         'text_thickness': max(int(2 * box_overlay_ratio), 1),
#         'text_padding': max(int(3 * box_overlay_ratio), 1),
#         'thickness': max(int(3 * box_overlay_ratio), 1),
#     }

#     ocr_bbox_rslt, is_goal_filtered = check_ocr_box(image_path, display_img = False, output_bb_format='xyxy', goal_filtering=None, easyocr_args={'paragraph': False, 'text_threshold':0.9}, use_paddleocr=True)
#     text, ocr_bbox = ocr_bbox_rslt

#     dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(image_path, som_model, BOX_TRESHOLD = BOX_TRESHOLD, output_coord_in_ratio=False, ocr_bbox=ocr_bbox,draw_bbox_config=draw_bbox_config, caption_model_processor=caption_model_processor, ocr_text=text,use_local_semantics=True, iou_threshold=0.1, imgsz=640)
#     return dino_labled_img, label_coordinates, parsed_content_list

 
# Screenshot and save path
image_folder = 'imgs'
if not os.path.exists(image_folder):
    os.makedirs(image_folder)
 
# Data model for structured output
class ElementExtractor(BaseModel):
    """Element extractor for the icon on the desktop."""
    binary_score: str = Field(description="Gives the name of the element on the screen of a Desktop.")
 
# Initialize the LLM with structured output
llm = ChatOllama(model="mistral-nemo:latest", temperature=0)
structured_llm_grader = llm.with_structured_output(ElementExtractor)
 
# Prompt template
system = """ You are a great element name extractor from a text. 
You can find the name of the element/tool/page/any button that the user is asking you to click or go to.
For example: 'Can you go to start', here you will find 'start' as the name of the element or 'Can you open app folder.', you will see 'app'.
Give a single word or just 2-3 word outputs."""
grade_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "This is the content of the whole Desktop page {page_elements}.\n\n User question: {question}"),
])
retrieval_grader = grade_prompt | structured_llm_grader
 
# Functions for finding index and moving/clicking
def find_content_index(parsed_content_list, target_content):
    for idx, content in enumerate(parsed_content_list):
        if target_content in content:
            return idx
    return None
 
def move_and_click(parsed_content_list, label_coordinates, target_content):
    index = find_content_index(parsed_content_list, target_content)
    if index is None:
        print(f"Content '{target_content}' not found.")
        return
 
    # Get the coordinates
    coordinates = label_coordinates.get(str(index))
    if coordinates is None:
        print(f"No coordinates found for content '{target_content}'.")
        return
 
    # Extract x, y from coordinates and move the cursor
    x, y = coordinates[0], coordinates[1]
    pyautogui.moveTo(x, y, duration=0.5)  # Smoothly move the cursor
    pyautogui.click()  # Perform a click action
    print(f"Clicked on '{target_content}' at coordinates: ({x}, {y})")
 
# Main loop to keep capturing and processing until stopped
try:
    while True:
        # Capture screenshot
        timestamp = int(time.time())
        image_path = os.path.join(image_folder, f"screenshot_{timestamp}.png")
        pyautogui.screenshot(image_path)
        # Load the screenshot
        image = Image.open(image_path)
        image_rgb = image.convert('RGB')
        box_overlay_ratio = image.size[0] / 3200
        draw_bbox_config = {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }
 
        # Run OCR and SOM labeling
        ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
            image_path,
            display_img=False,
            output_bb_format='xyxy',
            goal_filtering=None,
            easyocr_args={'paragraph': False, 'text_threshold': 0.9},
            use_paddleocr=True
        )
        text, ocr_bbox = ocr_bbox_rslt
        dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
            image_path, som_model,
            BOX_TRESHOLD=0.03,
            output_coord_in_ratio=False,
            ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_bbox_config,
            caption_model_processor=caption_model_processor,
            ocr_text=text,
            use_local_semantics=True,
            iou_threshold=0.1,
            imgsz=640
        )
        # print(parsed_content_list, type(parsed_content_list))

        import re
        cleaned_content_list = [item.split(": ", 1)[1] if ": " in item else item for item in parsed_content_list]
        print(cleaned_content_list)
        # Ask the question and extract the target content
        question = input("Ask a question (or type 'exit' to stop): ")
        if question.lower() == "exit":
            break

        print(f"+++++++++FUNCTION CALLING+++++++++")
        response = retrieval_grader.invoke({"question": question, "page_elements": cleaned_content_list})
        # print(response)
        target_content = response.binary_score  # Extracted content to click on
        print(target_content)

        # Move and click
        print(f"+++++++++ MOVE AND CLICK +++++++++++")
        move_and_click(parsed_content_list, label_coordinates, target_content)
 
        # Delay for a short period before the next iteration
        time.sleep(1)  # Adjust as needed for refresh rate
except KeyboardInterrupt:
    print("Process stopped by the user.")
