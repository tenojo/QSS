import json
import random


# Data should be in format:
# {col, col_id, text, under_col, under_col_id}
def format_datasets(dss:list[dict[str]]):
    # Dicts are in format: {ds_path, ds_name, under_ds_name}
    ds_items = []
    for x in dss:
        ds_path = x['ds_path']
        ds_name = x['ds_name']
        ds_under = x['under_ds_name']
        with open(ds_path, 'r', encoding='utf-8') as reader:
            for i,l in enumerate(reader):
                if len(l)>0:
                    line = json.loads(l.strip())
                    if ds_under:
                        ds_items.append({'collection':ds_name, 'collection_id':ds_name+"_"+str(i), 'text':line['text'], 'under_collection':ds_under, 'under_collection_id':line['custom_id']})
                    else:
                        ds_items.append({'collection':ds_name, 'collection_id':ds_name+"_"+str(i), 'text':line['text'], 'under_collection':None, 'under_collection_id':None})



    return ds_items


def sample_reference_corpus(dict_list, reference_name, reference_size):
    """
    Sample dictionaries from a list where the 'collection' field equals reference_name,
    and remove both sampled dictionaries and related dictionaries based on collection_id.
    
    Args:
        dict_list: List of dictionaries with format {'collection', 'collection_id', 'text', 
                  'under_collection', 'under_collection_id'}.
        reference_name: The value to match with the 'collection' field.
        reference_size: Number of dictionaries to sample.
    
    Returns:
        A tuple of (sampled_list, remaining_list) where:
        - sampled_list: List of sampled dictionaries from the specified collection
        - remaining_list: Original list with sampled dictionaries and related dictionaries removed
    """
    # Filter the list to find all dictionaries where 'collection' equals reference_name
    candidates = [d for d in dict_list if d.get('collection') == reference_name]
    
    # Ensure we don't try to sample more than what's available
    sample_size = min(reference_size, len(candidates))
    
    # Randomly sample from the candidates
    sampled = random.sample(candidates, sample_size)
    
    # Get collection_ids of all sampled items
    sampled_collection_ids = {d.get('collection_id') for d in sampled}
    
    # Create a new list that excludes:
    # 1. The sampled dictionaries
    # 2. Dictionaries where 'under_collection_id' equals 'collection_id' of any sampled dictionary
    remaining = [d for d in dict_list if d not in sampled and 
                 d.get('under_collection_id') not in sampled_collection_ids]
    
    return sampled, remaining

