# The CoT Polygraph - Implementation Plan

## Hypothesis

Qwen3 Activation Oracles can detect **latent model states** (confidence, bias, deception) that are **NOT explicitly stated** in Chain-of-Thought text.

---

## The Three Phases

### Phase 1: Data Generation ("Interesting Sentences")
- Use **Hinted MMLU** setup: questions with hidden hints/biases
- Generate CoT traces from Qwen3-4B (base model)
- Identify **Pivot Points** using Gemini:
  - Mind-change sentences ("Wait, let me re-read...")
  - Confident assertions
  - Planning statements

### Phase 2: Oracle Interrogation (The Experiment)
- Query AO on activations from **last token** of selected sentences
- **Critical Control Design**:
  - **Query A (Control)**: Text only, no/zero activation vector
  - **Query B (Intervention)**: Text + actual activation vector
- Questions to ask:
  - Confidence: "Is the model internally certain?"
  - Context Awareness: "Is the model influenced by hidden hints?"
  - Planning: "What will the model do next?"

### Phase 3: Evaluation Metrics
1. **Surprise Score**: Distance(Answer_Vector, Answer_TextOnly) - information gain from activations
2. **Future Prediction Accuracy**: Can AO predict next reasoning steps better than text-only?
3. **Truthfulness**: Can AO detect bias before model writes the answer?

---

## Codebase Understanding

### Key API: `run_oracle()`
```python
run_oracle(
    model, tokenizer, device,
    target_prompt: str,           # Pre-formatted with apply_chat_template
    target_lora_path: str | None, # None for base Qwen3
    oracle_prompt: str,           # Question about activations
    oracle_lora_path: str,        # "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B"
    segment_start_idx, segment_end_idx,  # Token range to analyze
    oracle_input_types: ["tokens", "segment", "full_seq"],
    layer_percent: int = 50,      # Which layer to extract from
) -> OracleResults
```

### Output Structure
- `token_responses`: Per-token oracle answers
- `segment_responses`: Segment-range oracle answers
- `full_sequence_responses`: Full-context oracle answers

### How It Works
1. **Collect activations** from target model at specified layer (50% default)
2. **Inject activations** into oracle model at layer 1 via steering hooks
3. **Oracle generates** response based on activation vectors + oracle prompt

### Available Resources
- Oracle LoRA: `adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B`
- Target LoRAs: taboo models, misaligned models, or base Qwen3
- Model: Qwen3-4B with 8-bit quantization

---

## Implementation Decisions

### Control Experiment (Phase 2)
- **Approach**: Skip injection entirely - query oracle model without activation steering
- **Implementation**: Modify `run_oracle` or create variant that bypasses the steering hook

### Data Generation (Phase 1)
- **Source**: Generate 20-30 MMLU-style questions using **Gemini 3 Flash Preview**
- **Hints**: Include various hint types in templates:
  - User bias hints ("I think the answer is B...")
  - Simplification hints ("Explain like I'm a child")
  - Authority hints ("My professor says...")
  - Emotional hints ("I really need to get this right")

### Pivot Point Detection
- **Tool**: Gemini 3 Flash Preview API
- **Targets**: Identify sentences with:
  - Mind changes ("Wait", "Actually", "Let me reconsider")
  - Confident assertions ("Clearly", "Obviously", "The answer is")
  - Planning statements ("First I'll", "Let me check", "I need to")

---

## Phase 1 Implementation Steps

### Step 1: Setup Gemini API
- Configure `google-generativeai` client
- Test connection with simple query

### Step 2: Generate Hinted MMLU Questions
- Create hint templates (5 different hint types)
- Generate 20 diverse questions across topics
- Format: `{"question": ..., "hint": ..., "correct_answer": ..., "hinted_answer": ...}`

### Step 3: Load Qwen3-4B Model
- Load model with 8-bit quantization
- Configure tokenizer with left padding
- Add dummy LoRA adapter for API compatibility

### Step 4: Generate CoT Traces
- Feed hinted questions to Qwen3-4B (base model, no LoRA)
- Enable thinking mode to get full CoT
- Store: question, hint, CoT trace, final answer

### Step 5: Identify Pivot Points
- Use Gemini to scan CoT and tag pivot sentences
- Extract token positions for each pivot point
- Store: sentence text, type (confidence/planning/mind-change), token indices

### Phase 1 Output Structure
```python
{
    "question_id": 1,
    "question": "...",
    "hint_type": "user_bias",
    "hint_text": "I think B is correct",
    "cot_trace": "...",
    "final_answer": "B",
    "correct_answer": "A",
    "pivot_points": [
        {"sentence": "...", "type": "confident", "token_idx": 42},
        {"sentence": "...", "type": "mind_change", "token_idx": 87},
    ]
}
```

---

## Questions Resolved
- Control: Skip injection (no steering)
- Data: Gemini 3 Flash Preview for generation
- Pivot detection: Gemini 3 Flash Preview API
- Model: Qwen3-4B (not 8B)
