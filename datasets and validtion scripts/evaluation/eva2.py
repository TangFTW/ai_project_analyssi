import pandas as pd
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel
import time, json, re
from rapidfuzz import fuzz

aiplatform.init(project="PROJECT_NAME", location="REGION")

model = GenerativeModel("ENDPT")
judge_model = GenerativeModel("MODEL_ENDPOINT")

with open("val_script_wtruth.jsonl", "r") as f:
    lines = f.readlines()

results = []

# synonym that are cultrally correct and accurate.
ALIASES = {
    "spicy dry pot": ["spicy and numbing wok"],
    "ants climbing a tree": ["sautéed cellophane noodles with minced pork"],
    "mouthwatering chicken": ["sichuan saliva chicken", "saliva chicken"],
    "fish-flavored tofu with minced pork": ["yuxiang minced pork tofu"],
    "kung pao shrimp": ["kung pao shrimp balls"],
    "steamed scallops with garlic and glass noodles": ["steamed scallops with garlic and vermicelli"],
    "fish with pickled mustard greens": ["pickled cabbage fish"],
    "steamed fish head with chopped chili": ["steamed fish head with chopped pepper"],
    "braised beef with potatoes": ["potato and beef stew"],
    "scrambled eggs with tomatoes": ["stir-fried tomatoes and eggs"],
    "shredded potatoes with green pepper": ["green pepper and potato threads"],
    "dry-fried green beans": ["stir-fried green beans with minced pork"],
    "lettuce with oyster sauce": ["oyster sauce lettuce"],
    "three-cup chicken": ["three cup chicken"],
    "scallion lamb stir-fry": ["scallion-exploded lamb"],
    "boiled fish in chili oil": ["sichuan boiled fish"],
    "boiled beef in chili oil": ["sichuan boiled beef"],
    "dry-pot cauliflower": ["stir-fried cauliflower in hot wok"],
    "stir-fried cured pork with garlic sprouts": ["stir-fried bacon with garlic sprouts"],
    "chicken with mushrooms": ["mushroom and chicken"],
    "stir-fried chicken cubes in sauce": ["stir-fried chicken cubes with brown bean sauce"],
    "stir-fried celery with dried tofu": ["stir-fried celery with smoked tofu"],
    "fish head and tofu clay pot": ["claypot fish head tofu"],
    "four happiness meatballs": ["braised pork ball in brown sauce"],
    "moo shu pork": ["mushu pork"],
    "leafy greens and tofu soup": ["vegetable and tofu soup"],
    "noodles with soybean paste sauce": ["zhajiangmian"],
    "salted pork and bamboo shoot soup": ["smoked ham and fresh pork soup"],
    "steamed fish with preserved mustard greens": ["steamed fish with preserved vegetables"],
    "spicy tofu": ["sichuan spicy tofu"],
    "poached chicken": ["sliced white cut chicken"],
    "noodles with scallion oil": ["scallion oil noodles"]
}

def safe_parse(row_str):
    try:
        row = json.loads(row_str.strip())
        source = row["request"]["contents"][0]["parts"][0]["text"]
        source_cn = source.split("\n", 1)[-1].strip()
        actual = row.get("ground_truth", "N/A")
        return source_cn, actual
    except:
        return None, None

def normalize(t):
    return re.sub(r"\s+", " ", t.strip().lower())

def canonicalize(t):
    t = normalize(t)
    for canon, vars in ALIASES.items():
        if t == canon or t in vars:
            return canon
    return t

def evaluate_translation(source_cn, actual_en, predicted_en):
    a = canonicalize(actual_en)
    p = canonicalize(predicted_en)

    # 1. Alias Match
    if a == p:
        return True
    
    # 2. Fuzzy Match (catches typos)
    if fuzz.ratio(a, p) >= 85:
        return True

    # 3. LLM Judge with Retry (Fixes 429 Errors)
    prompt = f"""
You are a strict-but-fair translation judge.

Chinese source: "{source_cn}"
Reference English: "{actual_en}"
Model English: "{predicted_en}"

Is the model English a semantically correct translation of the Chinese dish name?
Accept synonyms, common variants, pinyin, and hyphen differences.

Return ONLY JSON: {{"verdict":"YES" or "NO"}}
"""
    for attempt in range(3):
        try:
            resp = judge_model.generate_content(
                prompt,
                generation_config={"temperature": 0.0, "max_output_tokens": 20}
            ).text.strip()
            m = re.search(r'\{.*\}', resp)
            if m:
                verdict = json.loads(m.group())["verdict"].strip().upper()
                return verdict == "YES"
        except Exception:
            time.sleep(5 * (attempt + 1)) # Backoff on 429
            
    return False

for index, line in enumerate(lines):
    print(f"Processing item {index + 1} of {len(lines)}...")

    source_cn, actual = safe_parse(line)
    if source_cn is None:
        continue

    # Retry logic for the main model to prevent 429 ERRORs
    prediction_text = "ERROR"
    for attempt in range(3):
        try:
            response = model.generate_content(
                f"Translate to English:\n{source_cn}",
                generation_config={"temperature": 0.1, "max_output_tokens": 50}
            )
            prediction_text = response.text.strip()
            break
        except Exception as e:
            if attempt == 2:
                print(f" -> Failed to get prediction: {e}")
            time.sleep(10 * (attempt + 1))

    if prediction_text != "ERROR":
        is_correct = evaluate_translation(source_cn, actual, prediction_text)
    else:
        is_correct = False
        
    print(f" -> Match: {is_correct} | Actual: {actual} | Pred: {prediction_text}")

    results.append({
        "input": source_cn,
        "actual": actual,
        "predicted": prediction_text,
        "is_correct": is_correct
    })

    # Sleep longer to respect quota
    time.sleep(15)

output_df = pd.DataFrame(results)
accuracy = output_df['is_correct'].mean()
print(f"\n======================================")
print(f" Validation complete. Semantic Accuracy: {accuracy:.2%}")
print(f"======================================")

output_df.to_csv("eval_results_fixed.csv", index=False)