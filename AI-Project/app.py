from flask import Flask, request, render_template, jsonify
from vision import detect_text_from_image
from llm import generate, get_dish_info
import os
import uuid

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
  if request.method == 'POST':
    if 'file' not in request.files:
      return render_template('index.html', error='No file part')
    file = request.files['file']
    if file.filename == '':
      return render_template('index.html', error='No selected file')

    if file:
      content = file.read()

      # Save the image to a static file to display it
      image_filename = f"{uuid.uuid4()}{os.path.splitext(file.filename)[1]}"
      image_path = os.path.join('static', image_filename)
      with open(image_path, 'wb') as f:
          f.write(content)
      extracted_text = detect_text_from_image(content)
      
      system_instruction = """You are a highly specialized Chinese cuisine translator with deep knowledge of regional dialects (including Cantonese). Your task is to identify Chinese food items from the given text and return them as a JSON array.

You must follow these rules strictly:
1.  Each object in the array must have two keys: "chinese_name" and "english_name".
2.  For complex dish names, mentally break them down into their components to ensure an accurate translation. For example, '豉油炸脾餐肉飯' breaks down into '豉油' (Soy Sauce), '炸脾' (Fried Chicken Leg), '餐肉' (Luncheon Meat), and '飯' (Rice).
3.  If a line of text is clearly not an edible food item (e.g., a car, a shoe, a landmark, a person's name), you MUST ignore it.
4.  If you cannot find any valid food items in the entire text, you MUST return an empty array: `[]`. Do not attempt to translate non-food items.

**Example 1: Complex Regional Dish**
Input:
'''
豉油炸脾餐肉飯
'''
Output:
```json
[
  {"chinese_name": "豉油炸脾餐肉飯", "english_name": "Soy Sauce with Fried Chicken Leg and Luncheon Meat Rice"}
]
```

**Example 2: Mixed Items**
Input:
'''
宮保雞丁
木頭
麻婆豆腐
'''
Output:
```json
[
  {"chinese_name": "宮保雞丁", "english_name": "Kung Pao Chicken"},
  {"chinese_name": "麻婆豆腐", "english_name": "Mapo Tofu"}
]
```
**Example 2: No Food Items**
Input: `汽車`
Output: `[]`"""
      prompt_with_instruction = f"{system_instruction}\n\nProcess the following text:\n{extracted_text}"

      # Generate content based on extracted text
      dishes = generate(prompt_with_instruction)

      # If no dishes are found, but we have extracted text, we can assume it's not food.
      if not dishes and extracted_text and extracted_text.strip() != "No text found in image.":
        error_message = 'No food items found in the image. Please try another image'
        return render_template('index.html', extracted_text=extracted_text, dishes=[], error=error_message, image_filename=image_filename)

      return render_template('index.html', extracted_text=extracted_text, dishes=dishes, error=None, image_filename=image_filename)

  return render_template('index.html')

@app.route('/get_dish_details', methods=['POST'])
def get_dish_details_route():
  data = request.get_json()
  dish_name = data.get('dish_name')
  force_refresh = data.get('force_refresh', False)
  if not dish_name:
    return jsonify({'error': 'Invalid dish name provided.'}), 400
  details = get_dish_info(dish_name, force_refresh=force_refresh)
  return jsonify({'details': details})

if __name__ == '__main__':
  app.run(debug=True, host='0.0.0.0', port=5000)