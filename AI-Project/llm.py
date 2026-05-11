from google import genai
from google.genai import types
import json

def generate(input_text):
  """Generates content using the generative model with the given text."""
  client = genai.Client(
      vertexai=True,
      project="54847507337",
      location="us-central1",
  )
#Change back model to custom model, need train a new one
  #model = "projects/54847507337/locations/us-central1/endpoints/3538324075694784512"
  model = "gemini-2.5-flash-lite"
  contents = [input_text]

  generate_content_config = types.GenerateContentConfig(
    temperature = 0.3,
    top_p = 0.95,
    max_output_tokens = 1000,
    safety_settings = [types.SafetySetting(
      category="HARM_CATEGORY_HATE_SPEECH",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_DANGEROUS_CONTENT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_HARASSMENT",
      threshold="OFF"
    )],
    # Expecting a JSON response
    response_mime_type="application/json",
  )

  response_text = ""
  for chunk in client.models.generate_content_stream(
    model=model, contents=contents, config=generate_content_config
  ):
    if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
        response_text += chunk.text
  try:
    return json.loads(response_text)
  except json.JSONDecodeError:
    return [] # Return an empty list if the response is not valid JSON

# In-memory cache for dish details
dish_info_cache = {}

def get_dish_info(dish_name, force_refresh=False):
  """
  Gets detailed information about a dish using a specialized model.
  Uses an in-memory cache to avoid repeated API calls for the same dish.
  """
  # Check cache first unless a fresh result was requested.
  if not force_refresh and dish_name in dish_info_cache:
    return dish_info_cache[dish_name]

  # If not in cache, proceed with the API call
  client = genai.Client(
      vertexai=True,
      project="54847507337",
      location="us-central1",
  )

  model = "gemini-2.5-pro"

  system_instruction = """You are an expert Culinary Historian and Food Safety Specialist focused on Chinese cuisine. Your task is to provide comprehensive, culturally insightful, and safety-aware information about Chinese dishes.

When given a dish name (in Chinese or English), respond with a structured report that is easy to read and integrate into a modern web interface. Ensure the output is clean, without extra conversational text or special formatting characters like asterisks or hashtags.

The report must contain these four sections:

1.  **Dish Name**:
    *   Provide the common name in English and Traditional Chinese.
    *   Format: English Name (Chinese Name)

2.  **Cultural Origin**:
    *   A concise 2-3 sentence summary explaining the dish's origin (e.g., Cantonese, Sichuan, Northern) and its cultural significance or typical serving context (e.g., a cha chaan teng staple, banquet dish).

3.  **Key Ingredients**:
    *   List the main components in English, categorized as:
        *   Main: (e.g., Proteins, carbohydrates, vegetables)
        *   Sauce/Seasoning: (e.g., Spices, liquids, aromatics)

4.  **Allergen Alert**:
    *   List common potential allergens based on standard preparation (e.g., "Soy (in soy sauce)," "Gluten," "Shellfish (in oyster sauce)").
    *   Include a disclaimer: "Note: Recipes can vary significantly by restaurant."
"""

  prompt_with_instruction = f"{system_instruction}\n\nPlease provide detailed information about the following Chinese dish: {dish_name}"
  contents = [prompt_with_instruction]

  tools = [
    types.Tool(google_search=types.GoogleSearch()),
  ]

  generate_content_config = types.GenerateContentConfig(
    temperature=0.1, # Lower temperature for more factual responses
    top_p=0.95,
    max_output_tokens=2048,
    safety_settings=[], # Match the configuration of the other model call
    tools=tools,
  )

  response = client.models.generate_content(
      model=model,
      contents=contents,
      config=generate_content_config,
  )

  if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
      details = response.text
      # Store the newly fetched details in the cache before returning
      dish_info_cache[dish_name] = details
      return details
  return "Could not retrieve information for this dish."