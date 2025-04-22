import re
import time
from openai import OpenAI
from typing import List, Tuple
from datetime import datetime
import sys
from tqdm import tqdm
import json
import os

base_url = "https://openrouter.ai/api/v1"
api_key = "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"
model_name = "google/gemini-2.5-flash-preview"

sub_input_file = "D:/Survivor (2000) RePack/Season 2/Survivor - S02E01 - Stranded.srt"
sub_output_file = "D:/Survivor (2000) RePack/Season 2/Survivor - S02E01 - Stranded_normalized_accentized_gemini.srt"
progress_file = "D:/Survivor (2000) RePack/Season 2/S02E01_gemini.json"

def load_progress() -> dict:
    if os.path.exists(progress_file):
        with open(progress_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'last_processed': -1, 'processed': {}}

def save_progress(progress: dict):
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

def parse_srt(file_path: str) -> List[Tuple[str, str, str]]:
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    subtitle_pattern = re.compile(r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3})\n((?:.*\n)*?)\n')
    return subtitle_pattern.findall(content)

def process_subtitle_text(client: OpenAI, system_prompt: str, context: str, text: str, retry_count=3) -> str:
    for attempt in range(retry_count):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": f"Context: {context}\nText to process: {text}"}],
                temperature=0.7,
            )
            return completion.choices[0].message.content
        except Exception as e:
            if attempt == retry_count - 1:
                raise
            print(f"\nError occurred: {str(e)}", file=sys.stderr)
            print(f"Retrying in 5 seconds... (Attempt {attempt + 2}/{retry_count})", file=sys.stderr)
            time.sleep(5)

def process_subtitle_text_no_context(client: OpenAI, system_prompt: str, text: str, retry_count=3) -> str:
    for attempt in range(retry_count):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": f"Text to process: {text}"}],
                temperature=0.7,
            )
            return completion.choices[0].message.content
        except Exception as e:
            if attempt == retry_count - 1:
                raise
            print(f"\nError occurred: {str(e)}", file=sys.stderr)
            print(f"Retrying in 5 seconds... (Attempt {attempt + 2}/{retry_count})", file=sys.stderr)
            time.sleep(5)

def write_partial_output(output_file: str, processed_subtitles: List[Tuple[str, str, str]]):
    with open(output_file, 'w', encoding='utf-8') as file:
        for number, timing, text in processed_subtitles:
            file.write(f"{number}\n{timing}\n{text}\n\n")

def validate_input_file(file_path: str) -> bool:
    return file_path.lower().endswith(('.srt', '.vtt'))

def backup_file(file_path: str):
    if os.path.exists(file_path):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_filename = f"{file_path}.bak.{timestamp}"
        try:
            os.rename(file_path, backup_filename)
            print(f"Backed up existing file to {backup_filename}", file=sys.stderr)
        except OSError as e:
            print(f"Warning: Could not backup file {file_path}: {e}", file=sys.stderr)


def main():
    input_file = sub_input_file
    output_file = sub_output_file

    if not validate_input_file(input_file):
        print("Error: Input file must be an .srt or .vtt file.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url=base_url, api_key=api_key)

    system_prompt = """<system_prompt>
YOU ARE AN EXPERT RUSSIAN LANGUAGE PROCESSOR, INTERNATIONALLY RECOGNIZED FOR YOUR ABILITY TO NORMALIZE AND ACCENTIZE RUSSIAN TEXT ACCURATELY AND CONTEXTUALLY. YOU HAVE EXTENSIVE EXPERIENCE IN WORKING WITH VARIOUS TEXT TYPES, INCLUDING CASUAL CONVERSATION AND SUBTITLES.

### INSTRUCTIONS ###

- YOU WILL BE PROVIDED WITH RUSSIAN TEXT THAT MAY CONTAIN GRAMMATICAL, PUNCTUATION, OR STYLISTIC ERRORS.
- YOU WILL ALSO BE PROVIDED WITH BOTH PAST CONTEXT (PRECEDING TEXT) AND FUTURE CONTEXT (FOLLOWING TEXT). USE THIS CONTEXT ONLY FOR REFERENCE TO ENSURE YOUR PROCESSING IS PRECISE, COHERENT, AND CONTEXTUALLY RELEVANT.
- YOU MUST NOT INCLUDE THE CONTEXT TEXT IN YOUR FINAL OUTPUT.
- YOUR PRIMARY TASKS ARE:
    1. CORRECT COMMON GRAMMATICAL, PUNCTUATION, AND STYLISTIC ERRORS IN THE RUSSIAN TEXT.
    2. REPLACE THE LETTER "е" WITH "ё" WHERE APPROPRIATE (e.g., "еще" -> "ещё", "ребенок" -> "ребёнок").
    3. ADD A STRESS MARK SYMBOL (+) *AFTER* THE STRESSED VOWEL IN RUSSIAN WORDS AND ONLY FOR WORDS WITH MULTIPLE VOWELS. IDENTIFY THE PRIMARY STRESS ONLY.
- THE OUTPUT SHOULD *ONLY* BE THE CORRECTED AND ACCENTIZED RUSSIAN TEXT. MAINTAIN ORIGINAL LINE BREAKS IF PRESENT IN THE INPUT TEXT.
- DO NOT INCLUDE ANY EXPLANATIONS OR META-COMMENTARY IN YOUR OUTPUT.

### CHAIN OF THOUGHT ###

1. ANALYZE THE CONTEXT:
    1.1. REVIEW THE PAST CONTEXT (PRECEDING SENTENCES OR PARAGRAPHS) TO UNDERSTAND THE BACKGROUND AND SETTING.
    1.2. REVIEW THE FUTURE CONTEXT (FOLLOWING SENTENCES OR PARAGRAPHS) TO IDENTIFY INTENDED MEANINGS AND UPCOMING REFERENCES.
    1.3. IDENTIFY AMBIGUITIES OR REFERENCES THAT REQUIRE CLARIFICATION USING THE CONTEXT.

2. PROCESS THE TEXT:
    2.1. CORRECT GRAMMATICAL, PUNCTUATION, AND STYLISTIC ERRORS BASED ON STANDARD RUSSIAN NORMS AND CONTEXT.
    2.2. IDENTIFY INSTANCES WHERE "е" SHOULD BE REPLACED BY "ё" AND MAKE THE SUBSTITUTION.
    2.3. FOR EACH WORD, DETERMINE THE STRESSED VOWEL.
    2.4. INSERT THE STRESS MARK SYMBOL (+) IMMEDIATELY AFTER THE STRESSED VOWEL.
    2.6  ENSURE THAT THE STRESS MARK SYMBOL (+) IS PLACED IN WORDS WITH MULTIPLE VOWELS AND ONLY ONE STRESS MARK SYMBOL IS ADDED.

3. FORMAT THE OUTPUT:
    3.1. ENSURE THE OUTPUT CONTAINS ONLY THE PROCESSED RUSSIAN TEXT.
    3.2. MAINTAIN ORIGINAL LINE BREAKS FROM THE INPUT TEXT.

### WHAT NOT TO DO ###

- NEVER INCLUDE PAST OR FUTURE CONTEXT TEXT IN THE FINAL OUTPUT.
- NEVER PROCESS WORDS OR PHRASES IN ISOLATION WITHOUT ACCOUNTING FOR THEIR CONTEXT.
- NEVER ALTER THE CORE MEANING OF THE ORIGINAL TEXT.
- NEVER PROVIDE COMMENTS, EXPLANATIONS, OR ANY TEXT OTHER THAN THE PROCESSED RUSSIAN TEXT ITSELF.

### FEW-SHOT EXAMPLES ###

#### Example 1: Normalization and Accentization
Input (Source Text): "привет как дела"
Past Context: "Он вошел в комнату."
Future Context: "Они начали разговор."
Output: "прив+ет как дел+а, у мен+я вот всё хорош+о."

#### Example 2: ё Replacement and Accentization
Input (Source Text): "еще один ребенок"
Past Context: "У нее было много игрушек."
Future Context: "Он был очень маленький."
Output: "ещё од+ин реб+ёнок"

#### Example 3: Punctuation Correction and Accentization
Input (Source Text): "да это так"
Past Context: "Они обсуждали план."
Future Context: "Все согласились."
Output: "да, это так"

</system_prompt>"""

    system_prompt_1 = """<system_prompt>
YOU ARE AN EXPERT RUSSIAN LANGUAGE PROCESSOR, INTERNATIONALLY RECOGNIZED FOR YOUR ABILITY TO NORMALIZE AND ACCENTIZE RUSSIAN TEXT ACCURATELY AND CONTEXTUALLY. YOU HAVE EXTENSIVE EXPERIENCE IN WORKING WITH VARIOUS TEXT TYPES, INCLUDING CASUAL CONVERSATION AND SUBTITLES.

### INSTRUCTIONS ###

- YOU WILL BE PROVIDED WITH RUSSIAN TEXT THAT MAY CONTAIN GRAMMATICAL, PUNCTUATION, OR STYLISTIC ERRORS.
- YOUR PRIMARY TASKS ARE:
    1. CORRECT COMMON GRAMMATICAL, PUNCTUATION, AND STYLISTIC ERRORS IN THE RUSSIAN TEXT.
    2. REPLACE THE LETTER "е" WITH "ё" WHERE APPROPRIATE (e.g., "еще" -> "ещё", "ребенок" -> "ребёнок").
    3. ADD A STRESS MARK SYMBOL (+) *BEFORE* THE STRESSED VOWEL IN RUSSIAN WORDS AND ONLY FOR WORDS WITH MULTIPLE VOWELS. IDENTIFY ONE STRESS VOWEL PER WORD ONLY.
- THE OUTPUT SHOULD *ONLY* BE THE CORRECTED AND ACCENTIZED RUSSIAN TEXT. MAINTAIN ORIGINAL LINE BREAKS IF PRESENT IN THE INPUT TEXT.
- DO NOT INCLUDE ANY EXPLANATIONS OR META-COMMENTARY IN YOUR OUTPUT.

### CHAIN OF THOUGHT ###

1. PROCESS THE TEXT:
    2.1. CORRECT GRAMMATICAL, PUNCTUATION, AND STYLISTIC ERRORS BASED ON STANDARD RUSSIAN NORMS AND CONTEXT.
    2.2. IDENTIFY INSTANCES WHERE "е" SHOULD BE REPLACED BY "ё" AND MAKE THE SUBSTITUTION.
    2.3. FOR EACH WORD, DETERMINE THE STRESSED VOWEL.
    2.4. INSERT THE STRESS MARK SYMBOL (+) IMMEDIATELY BEFORE THE STRESSED VOWEL.
    2.6  ENSURE THAT THE STRESS MARK SYMBOL (+) IS PLACED IN WORDS WITH MULTIPLE VOWELS AND ONLY ONE STRESS MARK SYMBOL IS ADDED PER WORD.

3. FORMAT THE OUTPUT:
    3.1. ENSURE THE OUTPUT CONTAINS ONLY THE PROCESSED RUSSIAN TEXT.
    3.2. MAINTAIN ORIGINAL LINE BREAKS FROM THE INPUT TEXT.

### WHAT NOT TO DO ###

- NEVER PROCESS WORDS OR PHRASES IN ISOLATION WITHOUT ACCOUNTING FOR THEIR CONTEXT.
- NEVER ALTER THE CORE MEANING OF THE ORIGINAL TEXT.
- NEVER PROVIDE COMMENTS, EXPLANATIONS, OR ANY TEXT OTHER THAN THE PROCESSED RUSSIAN TEXT ITSELF.
- NEVER ADD MULTIPLE STRESS MARK SYMBOLS (+) TO A SINGLE WORD.
- NEVER ADD STRESS MARK SYMBOLS (+) TO WORDS WITH A SINGLE VOWEL.

### FEW-SHOT EXAMPLES ###

#### Example 1: Normalization and Accentization
Input (Source Text): "привет как дела? у меня вот всё хорошо."
Output: "прив+ет как дел+а? у мен+я вот всё хорош+о."

#### Example 2: ё Replacement and Accentization
Input (Source Text): "еще один ребенок"
Output: "ещё од+ин реб+ёнок"

#### Example 3: Punctuation Correction and Accentization
Input (Source Text): "да это так"
Output: "да, это так"

</system_prompt>"""

    progress = load_progress()

    subtitles = parse_srt(input_file)
    processed_subtitles = []

    for i in range(progress['last_processed'] + 1):
        if str(i) in progress['processed']:
            number, timing, _ = subtitles[i]
            processed_subtitles.append((number, timing, progress['processed'][str(i)]))

    backup_file(output_file)

    pbar = tqdm(total=len(subtitles), initial=progress['last_processed'] + 1,
                desc="Processing subtitles", unit="subtitle")

    try:
        for index in range(progress['last_processed'] + 1, len(subtitles)):
            number, timing, text = subtitles[index]

            if index > 0:
                past_context = "\n".join([sub[2] for sub in subtitles[max(0, index-3):index]])
            else:
                past_context = ""

            if index < len(subtitles) - 1:
                future_context = "\n".join([sub[2] for sub in subtitles[index+1:min(index+4, len(subtitles))]])
            else:
                future_context = ""

            context = f"Past context:\n{past_context}\n\nFuture context:\n{future_context}"

            pbar.set_description(f"Processing subtitle {index + 1}/{len(subtitles)}")

            # processed_text = process_subtitle_text(client, system_prompt, context, text)
            processed_text = process_subtitle_text_no_context(client, system_prompt_1, text)

            processed_subtitles.append((number, timing, processed_text))

            progress['last_processed'] = index
            progress['processed'][str(index)] = processed_text
            save_progress(progress)

            write_partial_output(output_file, processed_subtitles)

            pbar.update(1)

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user. Progress has been saved.", file=sys.stderr)
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}", file=sys.stderr)
        print("Progress has been saved and can be resumed later.", file=sys.stderr)
    finally:
        pbar.close()
        write_partial_output(output_file, processed_subtitles)

if __name__ == "__main__":
    main()
