import sys
import json
from ruaccent import RUAccent
from runorm import RUNorm

cust_dict = {'аганоа': 'аган+оа', 'санаапу': 'сан+аапу', 'вавау': 'вав+ау', 'дэз': 'д+эз', 'кюча': 'к+юча', 'огакор': 'ог+акор' }

accentizer = RUAccent()
accentizer.load(omograph_model_size='turbo3.1', use_dictionary=True, custom_dict=cust_dict, device="CUDA")
normalizer = RUNorm()
normalizer.load(workdir="runorm_cache", model_size="big", device="cpu")

def replace_ellipsis(text):
    return text.replace('...', '..')

def restore_ellipsis(text):
    return text.replace('..', '...')

# text = 'На двери висит замок. Ежик нашел в лесу ягоды. Эти 10 ягод он ежик оставил себе. Зачем ему ягоды?'
# print(accentizer.process_all(text))

def transform_json(obj):
    # Apply transformations to each 'text' field in the JSON data
    transformed_json_data = []
    for item in obj:
        if 'text' in item and isinstance(item['text'], list):
            transformed_texts = []

            for t in item['text']:
                # Apply other transformations here
                no_elipsis_text = replace_ellipsis(t)
                normalized_text = normalizer.norm(no_elipsis_text)
                transformed_text = accentizer.process_all(normalized_text)
                transformed_texts.append(restore_ellipsis(transformed_text))

            new_item = item.copy()
            new_item['text'] = transformed_texts
            transformed_json_data.append(new_item)


    return transformed_json_data

def save_text_to_file(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data:
            if 'text' in item and isinstance(item['text'], list):
                for text_line in item['text']:
                    file.write(text_line + '\n')

if __name__ == "__main__":
    if len(sys.argv) > 1:
        json_file_path = sys.argv[1]

        try:
            with open(json_file_path, 'r', encoding='utf-8-sig') as file:
                data = json.load(file)
                # print(data) # Removed this print as it's not needed

            transformed_data = transform_json(data)

            if len(sys.argv) > 2:
                # If a second argument is provided, save the transformed text to this path
                output_file_path = sys.argv[2]
                save_text_to_file(transformed_data, output_file_path)
            else:
                # If no second argument, print the transformed JSON (keeping original behavior for this case)
                print(json.dumps(transformed_data, ensure_ascii=False, indent=2))

        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        print("Please provide the path to a JSON file")
