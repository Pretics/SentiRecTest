from os import path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import csv
import random
from dataclasses import dataclass

@dataclass
class PrepBehaviorArgs:
    train_behavior_path: str
    test_behavior_path: str
    train_out_dir: str
    test_out_dir: str
    user2int_path: str
    split_test_size: float
    n_negative: int

class PrepBehavior:
    args: PrepBehaviorArgs

    def __init__(self, args):
        self.args = args
    
    # helper to load map (e.g. user-index) as dict
    @staticmethod
    def load_idx_map_as_dict(file_path):
        with open(file_path, 'r') as file:
            dictionary = {}
            lines = file.readlines()
            for line in tqdm(lines):
                key, value = line.strip().split("\t")
                dictionary[key] = value
            return dictionary

    # for each positive candidate sample N negative canidates
    # <uid>,<news ids of clickhistory>,<pos. candidate id + N negative candidate ids><mask: 1 + 0*N>
    def generate_training_data(self, behavior, out_dir):
        print("preparing training data")
        random.seed(1234)
        with open(path.join(out_dir, "train_behavior.tsv"), 'w', newline='') as train_out_file:
            train_writer = csv.writer(train_out_file, delimiter='\t')
            user2int = {}
            for b in tqdm(behavior): 
                imp_id, userid, imp_time, click, imps = b.strip().split("\t")
                if userid not in user2int:
                    user2int[userid] = len(user2int) + 1
                positive = [x[:-2] for x in imps.strip().split(" ") if x.endswith("1")]
                negative = [x[:-2] for x in imps.strip().split(" ") if x.endswith("0")]
                if (len(positive) < 1 or len(negative) < self.args.n_negative):
                    continue
                for p in positive:
                    ns  = random.sample(negative, self.args.n_negative)
                    pair = " ".join([p] + ns)
                    mask = " ".join(["1"]+["0"]*self.args.n_negative)
                    out = [user2int[userid], click, pair, mask]
                    train_writer.writerow(out)
            with open(path.join(out_dir, 'user2int.tsv'), 'w', newline='') as file:  
                user_writer = csv.writer(file, delimiter='\t')
                for key, value in user2int.items():
                    user_writer.writerow([key, value])
                return user2int
    # eval data is not balanced
    # <uid>,<news ids of clickhistory>,<candidate ids><click mask>
    @staticmethod
    def generate_eval_data(behavior, out_dir, out_file_name, user2int):
        print("preparing eval data")
        with open(path.join(out_dir, out_file_name), 'w', newline='') as eval_out_file:
            eval_writer = csv.writer(eval_out_file, delimiter='\t')
            for b in tqdm(behavior): 
                imp_id, userid, imp_time, click, imps = b.strip().split("\t")
                impressions =  " ".join([x[:-2] for x in imps.strip().split(" ")])
                impressions_mask = " ".join(["1" if x.endswith('1') else "0" for x in imps.strip().split(" ")])
                out = [user2int.get(userid, 0), click, impressions, impressions_mask]
                eval_writer.writerow(out)

    def prep_behavior(self):
        with open(self.args.train_behavior_path, 'r') as train_behaviors_file:
            behavior = train_behaviors_file.readlines()

        if (self.args.split_test_size == 0):
            self.generate_training_data(behavior, self.args.train_out_dir)
        else:
            train_behavior, val_behavior = train_test_split(behavior,test_size=self.args.split_test_size, random_state=1234)
            user2int = self.generate_training_data(train_behavior, self.args.train_out_dir)
            self.generate_eval_data(val_behavior, self.args.train_out_dir, "val_behavior.tsv", user2int)

        user2int = self.load_idx_map_as_dict(self.args.user2int_path)

        with open(self.args.test_behavior_path, 'r') as test_behaviors_file:
            behavior = test_behaviors_file.readlines()
            self.generate_eval_data(behavior, self.args.test_out_dir, "test_behavior.tsv", user2int)
