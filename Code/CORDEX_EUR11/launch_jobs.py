import os
import json

def json_loads_dictionary(json_array_having_key_and_value_as_keys):
    list_of_dicts_having_key_and_value_as_keys = json.loads(json_array_having_key_and_value_as_keys)
    if not all(['key' in diz for diz in list_of_dicts_having_key_and_value_as_keys]) and all(['value' in diz for diz in list_of_dicts_having_key_and_value_as_keys]):
        raise Exception('Error using json_loads_dictionary_split_into_key_and_value_as_keys_and_underwent_json_dumps: at least one dictionary in list_of_dicts_having_key_and_value_as_keys ismissing key "key" or key "value".')
    dict_having_tuple_as_key = {}
    for dict_having_key_and_value_as_keys in list_of_dicts_having_key_and_value_as_keys:
        dict_having_tuple_as_key[ tuple(dict_having_key_and_value_as_keys['key']) ] = dict_having_key_and_value_as_keys['value']
    return dict_having_tuple_as_key

with open('/home/tmandonnet/CORDEX/CORDEX_pairs_path_dict.json', 'r') as f:
    data = json.load(f)

loaded_dict = json_loads_dictionary(data)

for k,v in loaded_dict.items() :
    os.system(f"sbatch cordex_job.sh {k[0]} {k[1]} {k[2]} {k[3]} {v['historical']} {v['rcp']}")  