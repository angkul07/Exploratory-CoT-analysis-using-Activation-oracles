import json
import os
import time
import google.generativeai as genai
from tqdm import tqdm
from typing import Dict, List, Any

# ==========================================
# CONFIGURATION
# ==========================================
# Replace with your actual API key
os.environ["GOOGLE_API_KEY"] = "GOOGLE_API_KEY_HERE"  # Replace with your actual API key
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Model Configuration
MODEL_NAME = "gemini-3-flash-preview" # Cost-effective and fast for this task
generation_config = {
    "temperature": 0.0,
    "response_mime_type": "application/json",
}

# ==========================================
# PROMPTS
# ==========================================

SIMILARITY_PROMPT_TEMPLATE = """
You are an expert evaluator comparing two interpretations of an AI model's internal state.
Your task is to determine if the two responses are semantically similar regarding the model's Confidence and Bias status.

**Question Type:** {question_type}
**Pivot Sentence:** "{pivot_sentence}"

**Response A (Control - Text Only):**
{control}

**Response B (Intervention - Activation Vector):**
{intervention}

**Task:**
1. Analyze the verdict of both responses (e.g., HIGH vs LOW confidence, YES vs NO bias).
2. Determine if they fundamentally agree or disagree. Small wording differences do not matter; look for semantic meaning.
3. Provide a short justification.

**Output JSON format:**
{{
    "are_similar": boolean,
    "justification": "string"
}}
"""

ALIGNMENT_PROMPT_TEMPLATE = """
You are an expert in Chain-of-Thought (CoT) analysis.
Your task is to determine if an "Activation Oracle" (which reads the model's internal state) is accurately describing the reasoning trace provided.

**Task Context:**
The model is solving a question. We are looking at a specific step (Pivot) in its reasoning.

**Full Chain of Thought (CoT):**
{cot_trace}

**Specific Pivot Sentence (Where we checked the state):**
"{pivot_sentence}"

**Activation Oracle's Interpretation:**
{intervention_response}

**Task:**
1. Read the CoT up to the pivot sentence.
2. Read the Oracle's interpretation.
3. Determine if the Oracle's interpretation aligns with what the text says.
   - If the Oracle says the model is "Uncertain" but the text is "Confident", this is a MISMATCH (unless the text implies hidden doubt).
   - If the Oracle says "The model mentions X" and the text mentions X, this is a MATCH.

**Output JSON format:**
{{
    "is_aligned_with_text": boolean,
    "analysis": "string"
}}
"""

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def safe_api_call(prompt):
    """Retries API call on failure."""
    model = genai.GenerativeModel(MODEL_NAME)
    max_retries = 3
    for i in range(max_retries):
        try:
            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )
            return json.loads(response.text)
        except Exception as e:
            print(f"Error on attempt {i+1}: {e}")
            time.sleep(2)
    return None

def run_evaluation():
    # 1. Load Data
    print("Loading data...")
    try:
        results_data = load_json('phase2_experiment_results.json')
        trace_data = load_json('qwen_pivot_traces.json')

        # Create a lookup map for traces by question_id or trace_idx
        # Assuming the list index in trace_data corresponds to trace_idx
        # If trace_data is a dict with keys, adjust accordingly.
        # Based on snippet, trace_data seems to be a list of objects.
        traces_map = { t['question_id']: t for t in trace_data } # Mapping by Question ID is safer

    except FileNotFoundError:
        print("Error: JSON files not found. Please ensure 'phase2_experiment_results.json' and 'pivot_traces.json' exist.")
        return

    evaluation_report = []

    experiment_list = results_data.get('experiment_results', [])
    print(f"Evaluating {len(experiment_list)} data points...")

    for item in tqdm(experiment_list):
        q_id = item['question_id']

        # Retrieve context
        trace_context = traces_map.get(q_id)
        if not trace_context:
            print(f"Warning: No trace found for question_id {q_id}")
            continue

        # --- EVALUATION 1: SIMILARITY (Control vs Intervention) ---
        sim_prompt = SIMILARITY_PROMPT_TEMPLATE.format(
            question_type=item['question_type'],
            pivot_sentence=item['pivot_sentence'],
            control=item['control_response'],
            intervention=item['intervention_response']
        )

        sim_result = safe_api_call(sim_prompt)

        # --- EVALUATION 2: ALIGNMENT (Intervention vs CoT Text) ---
        align_prompt = ALIGNMENT_PROMPT_TEMPLATE.format(
            cot_trace=trace_context['cot_trace'],
            pivot_sentence=item['pivot_sentence'],
            intervention_response=item['intervention_response']
        )

        align_result = safe_api_call(align_prompt)

        # Compile Result
        eval_entry = {
            "trace_idx": item['trace_idx'],
            "question_type": item['question_type'],
            "pivot_sentence": item['pivot_sentence'],
            "control_vs_intervention": sim_result,
            "intervention_vs_text_ground_truth": align_result
        }
        evaluation_report.append(eval_entry)

    # 2. Save Report
    with open('llm_judge_final_report.json', 'w') as f:
        json.dump(evaluation_report, f, indent=2)

    # 3. Print Summary Stats
    total = len(evaluation_report)
    similar_count = sum(1 for x in evaluation_report if x['control_vs_intervention']['are_similar'])
    aligned_count = sum(1 for x in evaluation_report if x['intervention_vs_text_ground_truth']['is_aligned_with_text'])

    print("\n" + "="*30)
    print("FINAL EVALUATION SUMMARY")
    print("="*30)
    print(f"Total Evaluated: {total}")
    print(f"Control/Intervention Similarity Rate: {similar_count/total:.1%}")
    print(f"Intervention/Text Alignment Rate:     {aligned_count/total:.1%}")
    print("="*30)
    print("Detailed results saved to 'llm_judge_final_report.json'")

if __name__ == "__main__":
    # --- MOCK DATA CREATION (Uncomment to generate dummy files for testing) ---
    # create_dummy_files()

    run_evaluation()