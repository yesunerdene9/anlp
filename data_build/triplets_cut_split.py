import pandas as pd
import numpy as np
import os

IS_UKC_ID = 1

RELATION_MAP = {
    20: 'has_hyponym',
    22: 'meronym_has_part',
    34: 'attribute_has_attribute',
    36: 'meronym_has_substance',
    37: 'meronym_has_member',
    43: 'has_aspect'
}

concepts_df = pd.read_csv('concepts.csv', usecols=['id', 'label'])
relations_df = pd.read_csv('concept_relations.csv', usecols=['relation_type', 'src_con_id', 'trg_con_id'])

relations_df['relation_label'] = relations_df['relation_type']
relations_df.dropna(subset=['relation_label'], inplace=True)

head_concepts = concepts_df.rename(columns={'id': 'src_con_id', 'label': 'head_label'})
tail_concepts = concepts_df.rename(columns={'id': 'trg_con_id', 'label': 'tail_label'})

triples_df = relations_df.merge(head_concepts, on='src_con_id', how='left')
triples_df = triples_df.merge(tail_concepts, on='trg_con_id', how='left')

triples_df.dropna(subset=['head_label', 'tail_label'], inplace=True)

if IS_UKC_ID == 1:
    final_triples = triples_df[['src_con_id', 'relation_label', 'trg_con_id']]
else:
    final_triples = triples_df[['head_label', 'relation_label', 'tail_label']]

final_triples.columns = ['head', 'relation', 'tail']

MIN_FREQ = 2
VALID_RATIO = 0.04
TEST_RATIO = 0.04
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

df = final_triples.copy()
df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

all_entities = pd.concat([df["head"], df["tail"]])
entity_freq = all_entities.value_counts()

filtered_df = df.copy()

# for freq in [1, 2, 3, 4, 5]:
for freq in [1]:
    entities_to_filter = set(entity_freq[entity_freq == freq].index)
    if len(entities_to_filter) == 0:
        continue

    mask = ((filtered_df["head"].isin(entities_to_filter) | filtered_df["tail"].isin(entities_to_filter)) &
            (filtered_df["relation"] == 20))
    triples_to_consider = filtered_df[mask]

    dev = 1
    n_remove = int(len(triples_to_consider) * dev)
    remove_indices = triples_to_consider.sample(n=n_remove, random_state=RANDOM_SEED).index

    filtered_df = filtered_df.drop(remove_indices).reset_index(drop=True)

    all_entities = pd.concat([filtered_df["head"], filtered_df["tail"]])
    entity_freq = all_entities.value_counts()

    print(f"Removed {n_remove} triples containing entities with freq={freq}. Remaining triples: {len(filtered_df)}")

print("filtered_df len:", len(filtered_df))
n = len(filtered_df)
n_valid = int(n * VALID_RATIO)
n_test = int(n * TEST_RATIO)

valid_df = filtered_df.iloc[:n_valid]
test_df = filtered_df.iloc[n_valid:n_valid + n_test]
train_df = filtered_df.iloc[n_valid + n_test:]

def move_unseen_to_train(train_df, valid_df, test_df):
    seen = set(train_df["head"]).union(set(train_df["tail"]))

    def process_eval_split(eval_df):
        keep_rows = []
        move_rows = []
        for _, row in eval_df.iterrows():
            h, t = row["head"], row["tail"]
            if h in seen and t in seen:
                keep_rows.append(row)
            else:
                move_rows.append(row)
        return pd.DataFrame(keep_rows), pd.DataFrame(move_rows)

    valid_keep, valid_move = process_eval_split(valid_df)
    test_keep, test_move = process_eval_split(test_df)
    new_train = pd.concat([train_df, valid_move, test_move], ignore_index=True)
    return new_train, valid_keep, test_keep, len(valid_move) + len(test_move)

while True:
    train_df, valid_df, test_df, moved = move_unseen_to_train(train_df, valid_df, test_df)
    if moved == 0:
        break

data_dir = "../UKC_CUT_1_hyp_t"
os.makedirs(data_dir, exist_ok=True)
train_df.to_csv(f"{data_dir}/train", sep="\t", index=False, header=False)
valid_df.to_csv(f"{data_dir}/valid", sep="\t", index=False, header=False)
test_df.to_csv(f"{data_dir}/test", sep="\t", index=False, header=False)

print("Train:", len(train_df))
print("Valid:", len(valid_df))
print("Test:", len(test_df))
