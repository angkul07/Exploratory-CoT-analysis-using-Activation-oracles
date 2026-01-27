# Step 5: Generate CoT Traces from Gemma 2-9B (Batched)
# Adapted for Gemma's response format (no <think> tags, different markers)

import re
from typing import List, Dict, Tuple, Optional


def parse_gemma_cot_response(full_response: str) -> Tuple[str, str, Optional[str]]:
    """
    Parse a Gemma CoT response to extract reasoning and final answer.

    Gemma outputs reasoning text followed by "Final Answer: X"
    (Unlike Qwen3, Gemma doesn't use <think> tags)
    """
    cot_trace = ""
    final_answer_text = full_response.strip()
    final_answer_letter = None

    # Look for "Final Answer: X" pattern - most reliable since we asked for it
    final_answer_patterns = [
        r'[Ff]inal\s+[Aa]nswer[:\s]+([A-Da-d])\b',
        r'[Ff]inal\s+[Aa]nswer[:\s]+\**([A-Da-d])\**',
        r'[Tt]he\s+answer\s+is[:\s]+([A-Da-d])\b',
        r'[Aa]nswer[:\s]+([A-Da-d])\b',
        r'\*\*([A-Da-d])\*\*\s*$',
    ]

    for pattern in final_answer_patterns:
        match = re.search(pattern, full_response)
        if match:
            final_answer_letter = match.group(1).upper()
            break

    # Extract CoT trace (everything before "Final Answer")
    final_answer_markers = [
        r'[Ff]inal\s+[Aa]nswer',
        r'[Tt]herefore,?\s+the\s+answer',
        r'[Tt]he\s+correct\s+answer\s+is',
    ]

    cot_end_pos = len(full_response)
    for marker in final_answer_markers:
        match = re.search(marker, full_response)
        if match:
            cot_end_pos = min(cot_end_pos, match.start())
            break

    cot_trace = full_response[:cot_end_pos].strip()

    return cot_trace, final_answer_text, final_answer_letter


def extract_gemma_generated_response(full_decoded: str) -> str:
    """
    Extract only the generated model response from full decoded Gemma output.

    Gemma uses: <start_of_turn>model ... <end_of_turn> <eos>
    """
    model_marker = "<start_of_turn>model"
    last_model_idx = full_decoded.rfind(model_marker)

    if last_model_idx != -1:
        response = full_decoded[last_model_idx + len(model_marker):]
        response = response.strip()
        if response.startswith("\n"):
            response = response[1:]
        # Remove trailing markers
        response = re.sub(r'<end_of_turn>.*$', '', response, flags=re.DOTALL)
        response = re.sub(r'<eos>.*$', '', response, flags=re.DOTALL)
        return response.strip()

    # Fallback: clean all special tokens
    response = full_decoded
    for token in ['<bos>', '<start_of_turn>user', '<start_of_turn>model', '<end_of_turn>', '<eos>']:
        response = response.replace(token, '')
    return response.strip()


def generate_gemma_cot_traces_batched(questions: List[Dict], batch_size: int = 4, max_questions: int = None) -> List[Dict]:
    """Generate Chain-of-Thought traces for questions using Gemma 2-9B with batch processing."""

    # Disable any LoRA adapters to use base model
    try:
        model.disable_adapters()
    except:
        pass

    results = []
    questions_to_process = questions[:max_questions] if max_questions else questions
    num_questions = len(questions_to_process)

    # Pre-format all prompts using Gemma's chat template
    formatted_prompts = []
    for q in questions_to_process:
        prompt_dict = [{
            "role": "user",
            "content": q["full_question"] + "\n\nThink step by step and then end with 'Final Answer: X' where X is just the letter."
        }]
        # Gemma chat template - NO enable_thinking parameter (that's Qwen3-specific)
        formatted_prompt = tokenizer.apply_chat_template(
            prompt_dict,
            tokenize=False,
            add_generation_prompt=True
        )
        formatted_prompts.append(formatted_prompt)

    # Process in batches
    num_batches = (num_questions + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc=f"Generating CoT traces (batch_size={batch_size})"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_questions)

        batch_questions = questions_to_process[start_idx:end_idx]
        batch_prompts = formatted_prompts[start_idx:end_idx]

        try:
            # Tokenize batch with left padding
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(device)

            # Generate batch
            outputs = model.generate(
                **inputs,
                max_new_tokens=768,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

            # Process each output in the batch
            for i, (q, formatted_prompt) in enumerate(zip(batch_questions, batch_prompts)):
                # Decode the FULL output (including input)
                full_decoded = tokenizer.decode(outputs[i], skip_special_tokens=False)

                # Extract only the generated response using Gemma-specific extraction
                full_response = extract_gemma_generated_response(full_decoded)

                # Parse the response to get CoT and answer
                cot_trace, final_answer_text, final_answer_letter = parse_gemma_cot_response(full_response)

                results.append({
                    **q,
                    "cot_trace": cot_trace,
                    "final_answer_text": final_answer_text,
                    "final_answer_letter": final_answer_letter,
                    "full_response": full_response,
                    "formatted_prompt": formatted_prompt,
                    "is_correct": final_answer_letter == q["correct_answer"] if final_answer_letter else None,
                    "followed_hint": final_answer_letter == q["hinted_answer"] if (final_answer_letter and q.get("hinted_answer")) else None,
                })

        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            # Fall back to individual processing for this batch
            for q, formatted_prompt in zip(batch_questions, batch_prompts):
                try:
                    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
                    output = model.generate(
                        **inputs,
                        max_new_tokens=768,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                    full_decoded = tokenizer.decode(output[0], skip_special_tokens=False)
                    full_response = extract_gemma_generated_response(full_decoded)
                    cot_trace, final_answer_text, final_answer_letter = parse_gemma_cot_response(full_response)

                    results.append({
                        **q,
                        "cot_trace": cot_trace,
                        "final_answer_text": final_answer_text,
                        "final_answer_letter": final_answer_letter,
                        "full_response": full_response,
                        "formatted_prompt": formatted_prompt,
                        "is_correct": final_answer_letter == q["correct_answer"] if final_answer_letter else None,
                        "followed_hint": final_answer_letter == q["hinted_answer"] if (final_answer_letter and q.get("hinted_answer")) else None,
                    })
                except Exception as inner_e:
                    print(f"Error processing question {q['question_id']}: {inner_e}")
                    results.append({**q, "error": str(inner_e)})

    # Re-enable adapters
    try:
        model.enable_adapters()
    except:
        pass

    return results


# Generate CoT traces with batching
print("Generating CoT traces from Gemma 2-9B with batch processing...")
BATCH_SIZE = 4  # Gemma 9B may need smaller batch size than Qwen 4B
cot_dataset = generate_gemma_cot_traces_batched(hinted_questions, batch_size=BATCH_SIZE, max_questions=None)
print(f"\nGenerated {len(cot_dataset)} CoT traces")
