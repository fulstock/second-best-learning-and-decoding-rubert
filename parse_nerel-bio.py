from typing import Dict, Optional, Tuple, List
import json
from tokenizers.pre_tokenizers import BertPreTokenizer, Whitespace

from tqdm.auto import tqdm

class Stat:

    def __init__(self) -> None:
        self.total: int = 0
        self.layer: List[int] = []
        self.ignored: int = 0
        self.num_labels: int = 0
            
def parse_nerelbio() -> None:
    
    CORPUS_FILE_PATH: str = "./nerel-bio-v1.0/"
    
    TAG_SET: Dict[str, Stat] = {
        'ACTIVITY' : Stat(), 
        'ADMINISTRATION_ROUTE' : Stat(), 
        'AGE' : Stat(), 
        'ANATOMY' : Stat(), 
        'CHEM' : Stat(), 
        'CITY' : Stat(), 
        'COUNTRY' : Stat(), 
        'DATE' : Stat(),
        'DEVICE' : Stat(),
        'DISO' : Stat(), 
        'FACILITY' : Stat(),
        'FINDING' : Stat(), 
        'FOOD' : Stat(),
        'GENE' : Stat(), 
        'HEALTH_CARE_ACTIVITY' : Stat(), 
        'INJURY_POISONING' : Stat(), 
        'LABPROC' : Stat(), 
        'LIVB' : Stat(),
        'LOCATION' : Stat(),
        'MEDPROC' : Stat(),
        'MENTALPROC' : Stat(),
        'NUMBER': Stat(),
        'ORDINAL': Stat(),
        'ORGANIZATION': Stat(),
        'PERCENT': Stat(),
        'PERSON': Stat(),
        'PHYS' : Stat(),
        'PRODUCT' : Stat(),
        'PROFESSION' : Stat(),
        'SCIPROC' : Stat(),
        'STATE_OR_PROVINCE' : Stat(),
        'TIME' : Stat()
    }

    output_dir_path = "./nerel-bio-v1.0/"
    # os.makedirs(output_dir_path, mode=0o755, exist_ok=True)

    dataset_name = "nerel-bio"

    output_file_list = ["train", "dev", "test"]
    dataset_size_list = [-1, -1, -1]

    tokenizer = BertPreTokenizer()

    for output_type, dataset_size in zip(output_file_list, dataset_size_list):

        with open(CORPUS_FILE_PATH + output_type + ".json", 'r', encoding="UTF-8") as f:

            all_samples = json.load(f)
            existing_samples = [s for s in all_samples if s["exists"]]

            # merge contexts

            unique_samples = list(set([(sample["context"], sample["filename"], sample["id"].split('.')[0]) for sample in all_samples]))

            output_file = dataset_name + "." + output_type
            
            sent_count = 0
            token_count = 0
            error_count = 0

            outputs = []
            output_lines = []

            for tag in TAG_SET:
                TAG_SET[tag] = Stat()

            for context, filename, sent_id in tqdm(unique_samples):
                
                if len(context) == 0:
                    continue

                same_context_samples = [s for s in existing_samples if s["context"] == context]

                pret = tokenizer.pre_tokenize_str(context)
                words = [w for w, s in pret]
                spans = [s for w, s in pret]
                # print(pret)

                labels = []
                for sample in same_context_samples:

                    start_poses = sample["start_positions"]
                    end_poses = sample["end_positions"]

                    token_level_poses = []

                    for start, end in zip(start_poses, end_poses):

                        try:
                            first_word = [idx for idx, (s,e) in enumerate(spans) if s == start][0]
                            last_word = [idx for idx, (s,e) in enumerate(spans) if e == end][0] + 1

                            token_level_poses.append((first_word, last_word))
                        except:
                            error_count += 1
                            # print(f"Tokenizing error at tag {sample['tag']}:")
                            # print(context)
                            # print(pret)
                            # print(f"{start}, {end}, {context[start : end]}")
                            # print("\n")

                    # print(token_level_poses)
                    # print([words[first : last] for first, last in token_level_poses])

                    labels.extend([f"{start},{end} {sample['tag']}" for start, end in token_level_poses])

                words = " ".join(words) 
                labels = "|".join(labels)
                
                outputs.append((filename, int(sent_id)+1, words, labels))
    
                sent_count += 1
                token_count += len(words.split(' '))

                if sent_count == dataset_size:
                    break
        
        outputs = sorted(outputs, key = lambda out: (out[0], out[1]))
        for out in outputs:
            
            filename, sent_id, words, labels = out
            output_lines.append(filename + "\n")
            output_lines.append("Предложение #" + str(sent_id) + "\n")
            output_lines.append(words + "\n")
            output_lines.append(labels + "\n")
            output_lines.append("\n")
        
        with open(output_dir_path + output_file, 'w', encoding="UTF-8") as f2:
            f2.writelines(output_lines)

        print("")
        print("--- {}".format(output_file))
        print("# of sentences:\t{:6d}".format(sent_count))
        print("# of tokens:\t{:6d}".format(token_count))
        print("# of errors:\t{:6d}".format(error_count))

if __name__ == '__main__':
    parse_nerelbio()