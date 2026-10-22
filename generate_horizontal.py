import csv
import os
import pickle
import random
from PIL import Image
from functions import set_deterministic_mode
set_deterministic_mode()
unique_chars = []


def load_char_folders_from_unicode(base_dir):
    """Categorizes flat hex folders (e.g., 0x4e00, 0x3041) into jp, latin, number, and symbol."""
    folder_map = {"jp": [], "latin": [], "number": [], "symbol": []}

    if not os.path.exists(base_dir):
        print(f"Error: Directory '{base_dir}' does not exist.")
        return folder_map

    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        try:
            char_code = int(folder_name, 16)
            char = chr(char_code)
        except ValueError:
            continue

        # Categorize by character range
        if char.isdigit():
            folder_map["number"].append(folder_path)
        elif "a" <= char.lower() <= "z":
            folder_map["latin"].append(folder_path)
        elif (
            "\u3040" <= char <= "\u309f"  # Hiragana
            or "\u30a0" <= char <= "\u30ff"  # Katakana
            or "\u4e00" <= char <= "\u9fff"  # Kanji
        ):
            folder_map["jp"].append(folder_path)
        else:
            folder_map["symbol"].append(folder_path)

    return folder_map


def generate_horizontal_text_line(
    folder_map, output_path, text_length=15, latin_ratio=0.2
):
    selected_chars = []
    label = ""
    spacing = 1

    mix_chars = (
        folder_map["latin"] + folder_map["number"] + folder_map["symbol"]
    )
    num_mixed = int(text_length * latin_ratio) if mix_chars else 0
    mix_positions = (
        set(random.sample(range(text_length), num_mixed))
        if num_mixed > 0
        else set()
    )

    for i in range(text_length):
        if i in mix_positions and mix_chars:
            char_folder = random.choice(mix_chars)
        else:
            if not folder_map["jp"]:
                continue
            char_folder = random.choice(folder_map["jp"])

        image_files = [
            f
            for f in os.listdir(char_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if not image_files:
            continue
        char_image = random.choice(image_files)
        selected_chars.append(os.path.join(char_folder, char_image))

        try:
            character = chr(int(os.path.basename(char_folder), 16))
        except ValueError:
            continue

        label += character
        if character not in unique_chars:
            unique_chars.append(character)

    if not selected_chars:
        return None

    images = [
        Image.open(img_path).convert("L").resize((32, 32))
        for img_path in selected_chars
    ]

    width = sum(img.width for img in images) + (len(images) - 1) * spacing
    height = max(img.height for img in images)
    canvas = Image.new("L", (width + 10, height), color=255)

    x_offset = 5
    for img in images:
        canvas.paste(img, (x_offset, 0))
        x_offset += img.width + spacing

    canvas.save(output_path)
    return label


def generate_dataset(
    folder_map,
    num_samples=1000,
    output_dir="data/generated/horizontal",
    text_length=15,
    latin_ratio=0.2,
):
    os.makedirs(output_dir, exist_ok=True)
    csv_file = os.path.join(output_dir, "labels.csv")

    existing_images = [f for f in os.listdir(output_dir) if f.endswith(".png")]
    existing_indices = [
        int(f.split("_")[1].split(".")[0])
        for f in existing_images
        if f.startswith("line_")
    ]
    start_index = max(existing_indices, default=-1) + 1

    append_mode = os.path.exists(csv_file)
    with open(csv_file, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not append_mode or os.path.getsize(csv_file) == 0:
            writer.writerow(["image_path", "label"])

        for i in range(num_samples):
            index = start_index + i
            output_path = os.path.join(output_dir, f"line_{index}.png")
            label = generate_horizontal_text_line(
                folder_map, output_path, text_length, latin_ratio
            )

            if label:
                writer.writerow([output_path, label])

            if (i + 1) % 500 == 0 or (i + 1) == num_samples:
                print(
                    f"Horizontal Progress: {((i + 1) / num_samples * 100):.1f}%"
                )

    save_variable(unique_chars, os.path.join(output_dir, "vocab.pkl"))


def save_variable(variable, filename: str):
    with open(filename, "wb") as f:
        pickle.dump(variable, f)


if __name__ == "__main__":
    etl_base_dir = "tools/etlcdb-image-extractor/etl_data/images/ETL2"
    folder_map = load_char_folders_from_unicode(etl_base_dir)

    output_directory = "data/generated/horizontal"

    generate_dataset(
        folder_map,
        num_samples=8000,
        output_dir=output_directory,
        text_length=20,
        latin_ratio=0.0,
    )
    generate_dataset(
        folder_map,
        num_samples=4000,
        output_dir=output_directory,
        text_length=20,
        latin_ratio=0.3,
    )

    print(f"Horizontal Vocabulary size: {len(unique_chars)}")