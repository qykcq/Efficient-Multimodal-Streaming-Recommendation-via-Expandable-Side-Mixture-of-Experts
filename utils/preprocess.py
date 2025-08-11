from collections import defaultdict
from torch.utils.data import DataLoader
import torch
import numpy as np
from dataset import MMDataset, SingleModalDataset
import pandas as pd
import os


class Query:
    def __init__(self, id_to_name_u, name_to_id_u, id_to_name_i, name_to_id_i):
        self.__id_to_name_u = id_to_name_u
        self.__name_to_id_u = name_to_id_u

        self.__id_to_name_i = id_to_name_i
        self.__name_to_id_i = name_to_id_i

        self.num_users = len(self.__name_to_id_u)
        self.num_items = len(self.__name_to_id_i)

    def name_to_id(self, name, kind):
        if kind == 'user':
            if name not in self.__name_to_id_u:
                return -1
            return self.__name_to_id_u[name]
        else:
            assert kind == 'item'
            if name not in self.__name_to_id_i:
                return -1
            return self.__name_to_id_i[name]

    def id_to_name(self, id, kind):
        if kind == 'user':
            return self.__id_to_name_u[id]
        else:
            assert kind == 'item'
            return self.__id_to_name_i[id]


def index_string_columns(dfs):
    reviewer_map, asin_map = {}, {}
    kcore_dfs = []
    reviewer_count, asin_count = 0, 1
    for df in dfs:
        for reviewer in df['reviewerID'].unique():
            if reviewer not in reviewer_map:
                reviewer_map[reviewer] = reviewer_count
                reviewer_count += 1
        for asin in df["asin"].unique():
            if asin not in asin_map:
                asin_map[asin] = asin_count
                asin_count += 1
    # Reverse mappings
    reverse_reviewer_map = {idx: reviewer for reviewer, idx in reviewer_map.items()}
    reverse_asin_map = {idx: asin for asin, idx in asin_map.items()}

    # Replace string-based IDs with integer-based IDs in the DataFrame
    for df in dfs:
        df.loc[:, "reviewerID"] = df["reviewerID"].map(reviewer_map)
        df.loc[:, "asin"] = df["asin"].map(asin_map)
        kcore_dfs.append(df.reset_index(drop=True))
    return kcore_dfs, Query(reverse_reviewer_map, reviewer_map, reverse_asin_map, asin_map)


def index_string_columns_nonstream(df):
    reviewer_map, asin_map = {}, {}
    # user indexed from 0 and item indexed from 1.
    reviewer_count, asin_count = 0, 1
    for reviewer in df['reviewerID'].unique():
        if reviewer not in reviewer_map:
            reviewer_map[reviewer] = reviewer_count
            reviewer_count += 1

    for asin in df["asin"].unique():
        if asin not in asin_map:
            asin_map[asin] = asin_count
            asin_count += 1
    # Reverse mappings
    reverse_reviewer_map = {idx: reviewer for reviewer, idx in reviewer_map.items()}
    reverse_asin_map = {idx: asin for asin, idx in asin_map.items()}

    # Replace string-based IDs with integer-based IDs in the DataFrame
    df.loc[:, "reviewerID"] = df["reviewerID"].map(reviewer_map)
    df.loc[:, "asin"] = df["asin"].map(asin_map)

    return df, Query(reverse_reviewer_map, reviewer_map, reverse_asin_map, asin_map)


def get_train_valid_test_sets(args, query, dfs, t):
    df = dfs[t]
    train_size = int(len(df) * (1 - args.valid_portion))
    train_df = df[:train_size]
    valid_df = df[train_size:]
    del df
    test_df = dfs[t + 1]

    train_df = train_df.sort_values(by=['reviewerID', 'unixReviewTime'], ignore_index=True)
    valid_df = valid_df.sort_values(by=['reviewerID', 'unixReviewTime'], ignore_index=True)
    test_df = test_df.sort_values(by=['reviewerID', 'unixReviewTime'], ignore_index=True)

    # user: item 1, item 2, item 3, ...
    train_user_seqs = train_df.groupby('reviewerID')['asin'].apply(list).to_dict()
    valid_user_seqs = valid_df.groupby('reviewerID')['asin'].apply(list).to_dict()
    test_user_seqs = test_df.groupby('reviewerID')['asin'].apply(list).to_dict()

    # number of items in the training set plus previous sets
    item_num = train_df['asin'].max()
    del train_df, valid_df, test_df

    users_train, users_valid, users_test = {}, {}, {}

    # Build HISTORY up to (but NOT including) CURRENT slice
    history_before_cur = defaultdict(list)
    for past_df in dfs[:t]:
        for uid, items in past_df.groupby('reviewerID')['asin']:
            history_before_cur[uid].extend(items.tolist())

    # hist_for_valid & hist_for_test
    hist_for_valid, hist_for_test = {}, {}
    for uid in set(history_before_cur) | set(train_user_seqs) | set(valid_user_seqs):
        # interactions before current slice
        hist_prev = history_before_cur.get(uid, [])

        # training part of current slice
        cur_train = train_user_seqs.get(uid, [])

        # validation part of current slice
        cur_valid = valid_user_seqs.get(uid, [])

        # mask seen items during VALIDATION
        hist_for_valid[uid] = torch.LongTensor(np.array(hist_prev + cur_train))

        # mask seen items during TEST
        hist_for_test[uid] = torch.LongTensor(np.array(hist_prev + cur_train + cur_valid))

    train_item_counts = [0] * (item_num + 1)
    for user_id in train_user_seqs:
        # some legit items are filtered out of the max seq len
        item_seq = train_user_seqs[user_id][-args.max_seq_len - 1:]
        if len(item_seq) > 1:
            users_train[user_id] = item_seq
            for i in item_seq:
                train_item_counts[i] += 1

    for user_id in valid_user_seqs:
        item_seq = valid_user_seqs[user_id][-args.max_seq_len - 1:]
        if len(item_seq) > 1:
            users_valid[user_id] = item_seq

    for user_id in test_user_seqs:
        item_seq = test_user_seqs[user_id][-args.max_seq_len - 1:]
        if len(item_seq) > 1:
            users_test[user_id] = item_seq

    pop_prob_list = [0] * item_num
    for i in range(1, item_num + 1):
        pop_prob_list[i - 1] = train_item_counts[i]
    pop_prob_list = np.array(pop_prob_list) / sum(np.array(pop_prob_list))
    pop_prob_list = np.append([1], pop_prob_list)

    if args.modality in ['mm']:
        train_set = MMDataset(args, users_train, query, k=1.0)
    elif args.modality == 'cv':
        train_set = SingleModalDataset(args, users_train, query, 'cv', k=1.0)
    elif args.modality == 'text':
        train_set = SingleModalDataset(args, users_train, query, 'text', k=1.0)
    else:
        raise ValueError('Invalid choice of modality')

    train_dl = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    return train_dl, pop_prob_list, users_valid, users_test, hist_for_valid, hist_for_test


def k_core_processing(df, k):
    """
    Perform k-core processing on a DataFrame to ensure that each reviewID has interacted
    with at least k items (asin).

    Parameters:
        df (pd.DataFrame): Input DataFrame with 'reviewID' and 'asin' attributes.
        k (int): Minimum number of interactions required per user.

    Returns:
        pd.DataFrame: Filtered DataFrame after k-core processing.
    """
    while True:
        # Count the number of interactions per reviewID
        user_counts = df['reviewerID'].value_counts()

        # Filter out users with fewer than k interactions
        valid_users = user_counts[user_counts >= k].index
        filtered_df = df[df['reviewerID'].isin(valid_users)]

        # If no more changes, break the loop
        if len(filtered_df) == len(df):
            break

        df = filtered_df

    return df

def list_files_in_folder(folder_path):
    try:
        # Get the list of filenames in the folder
        filenames = os.listdir(folder_path)

        asins = []
        for filename in filenames:
            asin = filename.split('.')[0].split('_')[1]
            asins.append(asin)
        return set(asins)
    except FileNotFoundError:
        print("The folder '{}' does not exist.".format(folder_path))
    except Exception as e:
        print("An error occurred:", str(e))


def even_split_dataframe(dataframe, n_parts):
    split_indices = np.array_split(dataframe.index, n_parts)
    return [dataframe.iloc[indices] for indices in split_indices]


def read_json_to_sorted_kcore(args):
    dtypes = {'asin': str, 'reviewerID': str, 'unixReviewTime': str, 'title': str}
    df = pd.read_csv('datasets/{}/unfiltered_df.csv'.format(args.dataset_name), dtype=dtypes)
    asins_with_images = list_files_in_folder('datasets/{}/stored_vecs/vit_outputs/'.format(args.dataset_name))
    asins_with_texts = list_files_in_folder('datasets/{}/stored_vecs/bert_outputs/'.format(args.dataset_name))
    asins_with_both_modalities = asins_with_texts.intersection(asins_with_images)
    del asins_with_texts, asins_with_images

    df = df[df['asin'].isin(asins_with_both_modalities)]

    print('{} interactions between {} users and {} items'.format(
        len(df), df['reviewerID'].nunique(), df['asin'].nunique()))

    # sort k_core by unixReviewTime
    chrono_df = df.sort_values(by=['unixReviewTime'], ignore_index=True)
    del df

    dfs = even_split_dataframe(chrono_df, args.time_windows)

    kcore_dfs = []
    for df in dfs:
        kcore_dfs.append(k_core_processing(df, k=args.k_core))
    del dfs, chrono_df

    # reindex the users and items
    kcore_dfs, query = index_string_columns(kcore_dfs)

    summarize_dataframes(kcore_dfs)
    return kcore_dfs, query


def summarize_dataframes(df_list):
    """
    Prints stats for each dataframe in the list and overall stats across all dataframes.

    Parameters:
    df_list (list of pd.DataFrame): Each DataFrame should have columns 'asin' and 'reviewerID'.
    """
    import pandas as pd

    print("Per-DataFrame Summary:\n")
    for i, df in enumerate(df_list):
        num_records = len(df)
        num_unique_asins = df['asin'].nunique()
        num_unique_reviewers = df['reviewerID'].nunique()
        print(f"DataFrame {i + 1}:")
        print(f"  Records: {num_records}")
        print(f"  Unique ASINs: {num_unique_asins}")
        print(f"  Unique ReviewerIDs: {num_unique_reviewers}\n")

    # Combine all dataframes
    combined_df = pd.concat(df_list, ignore_index=True)
    total_records = len(combined_df)
    total_unique_asins = combined_df['asin'].nunique()
    total_unique_reviewers = combined_df['reviewerID'].nunique()

    print("Overall Summary:")
    print(f"  Total Records: {total_records}")
    print(f"  Total Unique ASINs: {total_unique_asins}")
    print(f"  Total Unique ReviewerIDs: {total_unique_reviewers}")


