import json
import pandas as pd
from datasets import Dataset, Features, Image as HFImage, Value
from PIL import Image
from tqdm import tqdm
from huggingface_hub import login
# Thêm resize ảnh
# Log in to Hugging Face Hub
login(token="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")

# Helper function to map pandas dtypes to datasets feature types
def pandas_dtype_to_datasets_feature(dtype):
    if pd.api.types.is_integer_dtype(dtype):
        return Value("int64")
    elif pd.api.types.is_float_dtype(dtype):
        return Value("float64")
    elif pd.api.types.is_bool_dtype(dtype):
        return Value("bool")
    elif pd.api.types.is_string_dtype(dtype) or dtype == "object":
        return Value("string")
    else:
        raise ValueError(f"Unsupported pandas dtype: {dtype}")

if __name__ == "__main__":
    # Define chunk size
    CHUNK_SIZE = 500  # Adjust based on your system's memory

    # Load JSON without keeping everything in memory
    with open("new_moe_data.json", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # Define the features explicitly, including the 'image' column as an HFImage type
    features = Features({
        col: pandas_dtype_to_datasets_feature(df[col].dtype) for col in df.columns
    })
    features["image"] = HFImage()  # Explicitly define the image column

    # Process data in chunks
    print("number of chunk:", len(df)//CHUNK_SIZE)
    for i in tqdm(range(0, len(df), CHUNK_SIZE), desc="Processing Chunks"):
        if ((i // CHUNK_SIZE + 1) <= 1663): 
            continue
        print("Chunk:", i // CHUNK_SIZE + 1)
        chunk = df.iloc[i:i + CHUNK_SIZE]  # Extract chunk

        # Convert to Hugging Face Dataset with predefined features (excluding 'image' for now)
        dataset_chunk = Dataset.from_pandas(
            chunk,
            features=Features({k: v for k, v in features.items() if k != "image"})
        )

        # Process images using a for loop
        images = []
        for example in tqdm(dataset_chunk, desc="Loading Images"):
            image_path = "moe_dataset/" + example["image_id"]
            # print(image_path)
            try:
                image = Image.open(image_path).convert("RGB")  # Load image
                # if i == 1661 or i == 1663:
                image = image.resize((224, 224))  # Resize to 224x224
            except Exception as e:
                print(f"Error loading image {example['image_id']}: {e}")
                image = None  # Handle errors gracefully
            images.append(image)

        # Combine all data into a single dictionary
        combined_data = {col: dataset_chunk[col] for col in dataset_chunk.column_names}
        combined_data["image"] = images

        # Create a new dataset with the combined data and full features
        dataset_chunk = Dataset.from_dict(combined_data, features=features)

        # Remove 'image_path' if it exists
        if "image_path" in dataset_chunk.column_names:
            dataset_chunk = dataset_chunk.remove_columns("image_path")

        # Push chunk to Hugging Face Hub
        dataset_chunk.push_to_hub("nhq188/moe-dataset-2", split=f"subset_{i}")
        del dataset_chunk
        del chunk
        del images
        print(f"Uploaded chunk {i // CHUNK_SIZE + 1} to Hugging Face.")
