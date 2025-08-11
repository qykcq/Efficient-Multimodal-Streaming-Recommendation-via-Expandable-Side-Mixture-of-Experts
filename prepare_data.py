import ast
import csv
import pandas as pd
import json
import os
import gc
import requests
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import ViTForImageClassification
torch.manual_seed(123456)


def read_json(input_file):
    objects = []
    # Read and process the file
    with open(input_file, "r", encoding="utf-8") as file:
        for line in file:
            try:
                # Parse the JSON object from the line
                json_object = json.loads(line.strip())
                objects.append(json_object)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line: {line.strip()}\nError: {e}")
    return objects


def get_records_hm(folder, downloaded_images):
    products = []
    asins_in_meta= set()
    # First pass: count ASIN frequencies
    with open(os.path.join(folder, 'meta.csv'), "r", encoding='utf-8') as f:
        next(f)
        for line in f:
            line = line.strip()
            reader = csv.reader([line])
            item_id, _, title, *rest = next(reader)
            desc = rest[-1]
            if title is None or len(title) == 0 or item_id not in downloaded_images:
                continue
            asins_in_meta.add(item_id)
            products.append({
                'asin': item_id,
                'title': title + ' ' + desc
            })
    print('Read {} items.'.format(len(products)))

    reviews = []
    # First pass: collect reviews and count asin/user frequencies
    with open(os.path.join(folder, 'reviews.csv'), "r", encoding='utf-8') as f:
        next(f)
        for line in f:
            timestamp, user, item, _, _ = line.strip().split(',')

            if timestamp is None or timestamp == '' or item not in asins_in_meta:
                continue

            if not (pd.Timestamp('2020-01-01') <= pd.Timestamp(timestamp) <= pd.Timestamp("2020-12-31")):
                continue

            reviews.append({
                'reviewerID': user,
                'asin': item,
                'unixReviewTime': timestamp
            })
    print('Read {} reviews.'.format(len(reviews)))

    return pd.DataFrame(reviews), pd.DataFrame(products)


def get_records_amazon(folder):
    products = []
    with open(os.path.join(folder,'meta.json'), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            json_obj = json.loads(line.strip())  # Parse each JSON object
            title = json_obj.get("title", None)
            description = json_obj.get("description", None)
            if description is not None and len(description) > 0:
                title = title + ' ' + description[0]

            asin = json_obj.get("asin", None)
            image_urls = json_obj.get("imageURL", None)
            if title is None or len(title) == 0 or asin is None or image_urls is None or len(image_urls) == 0 or '{' in title:
                continue

            products.append({
                "asin": asin,
                "title": title,
                "imageURL": image_urls,
            })
    products = pd.DataFrame(products)
    print('number of products', len(products))

    df_iterator = pd.read_json(os.path.join(folder, "reviews.json"), lines=True, chunksize=10000)
    filtered_chunks = []
    for _, chunk in enumerate(df_iterator):
        filtered_chunk = chunk[['asin', 'reviewerID', 'unixReviewTime']]  # Select only required columns
        filtered_chunks.append(filtered_chunk)
    reviews = pd.concat(filtered_chunks, ignore_index=True)
    reviews = pd.DataFrame(reviews)
    print('number of reviews', len(reviews))
    return reviews, products


def get_df(dataset_name, downloaded_images=None):
    dtypes = {'asin': str, 'reviewerID': str, 'unixREviewTime': str, 'title': str}
    try:
        df = pd.read_csv(os.path.join('datasets', dataset_name, 'unfiltered_df.csv'), dtype=dtypes)
    except Exception as e:
        folder = os.path.join('/mnt/d/datasets', dataset_name)
        if 'clothes' in folder or 'home' in folder or 'electronics' in folder:
            reviews, products = get_records_amazon(folder)
        elif 'hm' in folder:
            reviews, products = get_records_hm(folder, downloaded_images)
        else:
            raise ValueError('Invalid choice of dataset:', folder)
        print('Processed products and reviews information.')
        df = pd.merge(products, reviews, on='asin', how='inner')
        print("Number of joined records:", len(df))
        print(df.head(3))
        # drop rows with empty image url and title
        df = df.dropna(ignore_index=True)
        df = df.reset_index(drop=True)
        df.to_csv(os.path.join('datasets', dataset_name, 'unfiltered_df.csv'), index=False)
    print('{} interactions between {} users and {} items'.format(
        len(df), df['reviewerID'].nunique(), df['asin'].nunique()))
    print('Dataframe processing completed.')
    return df


def download_and_save_image(asin, url_list, save_dir, existing):
    if asin in existing:
        return asin

    """Tries to download images from a list of URLs until one succeeds."""
    for url in url_list:
        try:
            # Download the image
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Raise error for HTTP issues

            # Verify the image can be opened
            img = Image.open(BytesIO(response.content))

            # Save the image locally
            os.makedirs(save_dir, exist_ok=True)
            image_path = os.path.join(save_dir, f"{asin}.jpg")
            img.save(image_path)
            return asin  # Return ASIN if successful
        except Exception as e:
            pass
            # print(f"Failed to download image for ASIN: {asin}, URL: {url}. Error: {e}")

    return None  # Return None if all URLs failed


def download_images_parallel(df, save_dir, max_workers=8):
    """Downloads images in parallel, trying multiple URLs per product if needed."""
    def process_chunk(chunk):
        nonlocal failed_downloads
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all download tasks
            futures = {executor.submit(download_and_save_image, asin, url_list, save_dir, existing): asin
                       for asin, url_list in chunk}

            # Use tqdm to show progress bar
            for future in as_completed(futures):
                result = future.result()
                if result:
                    successful_downloads.append(result)
                else:
                    failed_downloads += 1

    # already downloaded
    # Get the list of filenames in the folder
    filenames = os.listdir(save_dir)
    existing = set()
    for filename in filenames:
        existing.add(filename.split('.')[0])

    print(f"Number of previously downloaded images: {len(existing)}")
    successful_downloads = []
    failed_downloads = 0

    # Collect ASIN and their list of image URLs
    chunk_size = 2000
    for start_idx in tqdm(range(0, len(df), chunk_size)):
        chunk = df.iloc[start_idx: start_idx + chunk_size]
        asin_url_list = []
        for row in chunk.itertuples():
            image_urls = row.imageURL
            if isinstance(image_urls, str):
                image_urls = ast.literal_eval(image_urls)

            if len(image_urls) > 0:
                asin_url_list.append((row.asin, image_urls))

        process_chunk(asin_url_list)

    print(f'{len(successful_downloads)} images downloaded; {failed_downloads} downloads failed.')
    return successful_downloads


def load_batch_images(dataset_folder, transform, item_ids_batch):
    images = {}
    for item_id in item_ids_batch:
        item_path = os.path.join(dataset_folder, item_id + '.jpg')
        img = Image.open(item_path).convert('RGB')
        images[item_id] = transform(img)
    return images


def save_outputs(directory, outputs, prefix=''):
    os.makedirs(directory, exist_ok=True)
    for item_id, output in outputs.items():
        file_path = os.path.join(directory, f"{prefix}_{item_id}.pt")
        torch.save(output, file_path)


def load_output(directory, item_id, prefix=''):
    file_path = os.path.join(directory, f"{prefix}_{item_id}.pt")
    if os.path.exists(file_path):
        return torch.load(file_path)
    else:
        return None


def process_items(dataset_folder, texts, transform, bert_model, cv_model, tokenizer, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model.to(device).eval()
    cv_model.to(device).eval()

    item_ids_list = list(texts.keys())
    num_batches = len(item_ids_list) // batch_size + (len(item_ids_list) % batch_size != 0)

    with tqdm(total=num_batches, desc="Processing Batches", unit="batch") as pbar:
        for batch_num in range(0, num_batches):
            bert_outputs = {}
            vit_outputs = {}
            start_idx = batch_num * batch_size
            end_idx = start_idx + batch_size
            batch_item_ids = item_ids_list[start_idx:end_idx]

            # Process text batch
            text_batch = [texts[item_id] for item_id in batch_item_ids]
            encoded_input = tokenizer(text_batch, max_length=40, padding='max_length', truncation=True,
                                      return_tensors="pt").to(device)
       
            with torch.no_grad():
                bert_output = bert_model(**encoded_input)

            for i, item_id in enumerate(batch_item_ids):
                # extracting the [cls] token
                hidden_states_for_item = [hidden_state[i][0, :] for hidden_state in bert_output.hidden_states]
                hidden_states_for_item = torch.stack(hidden_states_for_item)
                bert_outputs[item_id] = hidden_states_for_item.cpu()

            # Load image data for this batch
            image_folder = os.path.join(dataset_folder, 'images')
            image_batch = load_batch_images(image_folder, transform, batch_item_ids)

            image_batch_values = torch.stack(list(image_batch.values())).to(device)

            with torch.no_grad():
                vit_output = cv_model(image_batch_values)
            for i, item_id in enumerate(image_batch):
                hidden_states_for_item = [hidden_state[i][0, :] for hidden_state in vit_output.hidden_states]
                hidden_states_for_item = torch.stack(hidden_states_for_item)
                vit_outputs[item_id] = hidden_states_for_item.cpu()

            # Release memory
            save_outputs(f'datasets/{dataset_name}/stored_vecs/bert_outputs', bert_outputs, prefix='bert')
            save_outputs(f'datasets/{dataset_name}/stored_vecs/vit_outputs', vit_outputs, prefix='vit')
            torch.cuda.empty_cache()
            del encoded_input, text_batch, image_batch, bert_output, vit_output
            gc.collect()
            pbar.update(1)


def load_and_save(dataset_folder, items_with_images, df):
    print("loading bert...")
    bert_model_load = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(bert_model_load)
    config = BertConfig.from_pretrained(bert_model_load, output_hidden_states=True)
    bert_model = BertModel.from_pretrained(bert_model_load, config=config)

    print("loading vit...")
    cv_model_load = 'google/vit-base-patch16-224'
    cv_model = ViTForImageClassification.from_pretrained(cv_model_load, output_hidden_states=True)
    cv_model.classifier = nn.Identity()

    # load data
    texts = df.set_index('asin')['title'].to_dict()
    texts_with_images = {}
    for asin in items_with_images:
        if asin in texts:
            texts_with_images[asin] = texts[asin]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    process_items(dataset_folder, texts_with_images, transform, bert_model, cv_model, tokenizer)


def get_image_names(folder_path):
    names = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".jpg"):  # Ensure it's a JPG file
            name, _ = os.path.splitext(filename)  # Extract NAME part
            names.append(name)
    return set(names)


if __name__ == '__main__':
    # amazon
    dataset_name = input("Choice of dataset: ")
    dataset_name = dataset_name.strip()
    dataset_folder = os.path.join('/mnt/d/datasets', dataset_name)
    if dataset_name in ['clothes', 'sports', 'home', 'electronics']:
        df = get_df(dataset_name)
        # downloaded_images = get_image_names(os.path.join(dataset_folder, 'images'))
        # print('number of images', len(downloaded_images))
        downloaded_images = download_images_parallel(df, os.path.join(dataset_folder, 'images'))
        load_and_save(dataset_folder, downloaded_images, df)
    else:
        # for hm
        downloaded_images = get_image_names(os.path.join(dataset_folder, 'images'))
        df = get_df(dataset_folder, downloaded_images)

        # load_and_save(os.path.join('/mnt/d/datasets', dataset_name), downloaded_images, df)
        
