import os
import pickle
from collections import OrderedDict

from dassl.data.datasets import DATASET_REGISTRY, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

from .oxford_pets import OxfordPets
import numpy as np
import math
import glob

CLASS_LABELS = ["basket", "bed", "bench", "bin", "blanket", "blinds", "book", "bottle", "box", "bowl", "camera", "cabinet", "candle", "chair", "clock",
"cloth", "comforter", "cushion", "desk", "desk-organizer", "door", "indoor-plant", "lamp", "monitor", "nightstand",
"panel", "picture", "pillar", "pillow", "pipe", "plant-stand", "plate", "pot", "sculpture", "shelf", "sofa", "stool", "switch", "table",
"tablet", "tissue-paper", "tv-screen", "tv-stand", "vase", "vent", "wall-plug", "window", "rug"]
VALID_CLASS_IDS = np.asarray([3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 26, 29, 34, 35, 37, 44, 47, 52, 54, 56, 59, 60, 61, 62, 63, 64, 65, 70, 71, 76, 78, 79, 80, 82, 83, 87, 88, 91, 92, 95, 97, 98])
ID_TO_LABEL = {}
LABEL_TO_ID = {}
PRED_ID_TO_ID = {}
ID_TO_PRED_ID = {}
for pred_id, i in enumerate(range(len(VALID_CLASS_IDS))):
    ID_TO_PRED_ID[VALID_CLASS_IDS[i]] = pred_id
    PRED_ID_TO_ID[pred_id] = VALID_CLASS_IDS[i]
    LABEL_TO_ID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
    ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]
    
SKIPPED_CLASSES = ["undefined", "floor", "ceiling", "wall"]
from collections import Counter

def check_labels(dataset):
    labels = []
    for data in dataset:
        label = data.label  # データセット構造に応じて調整
        labels.append(label)
    
    print(f"Label range: {min(labels)}-{max(labels)}")
    print(f"Unique labels: {set(labels)}")




@DATASET_REGISTRY.register()
class Replica(DatasetBase):

    dataset_dir = "replica_features"
    # dataset_dir = "instance_features"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        # mkdir_if_missing(self.split_fewshot_dir)


        
        # text_file = os.path.join(self.dataset_dir, "classnames.txt")
        # classnames = self.read_classnames(text_file)
        train = self.read_data("train")
        # Follow standard practice to perform evaluation on the val set
        # Also used as the val set (so evaluate the last-step model)
        val = self.read_data("val")
        test = self.read_data("test")
        


        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train = data["train"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                data = {"train": train}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
        print("-----------------------------------------------------------")
        print("train:", train)
        print("val:", val)
        print("test:", test)
        print("-----------------------------------------------------------")
        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = self.subsample_classes(train, val, test, subsample=subsample)
        print("-----------------------------------------------------------")
        print("train:", train)
        print("val:", val)
        print("test:", test)
        print("-----------------------------------------------------------")
        check_labels(train)
        check_labels(val)
        check_labels(test)

        super().__init__(train_x=train, val=val, test=test)

    
    @staticmethod
    def read_classnames(text_file):
        """Return a dictionary containing
        key-value pairs of <folder name>: <class name>.
        """
        classnames = OrderedDict()
        for pred_id, i in enumerate(range(len(VALID_CLASS_IDS))):
            classnames[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]
        
        return classnames

    def read_data(self, split):
        split_dir = os.path.join(self.dataset_dir, split)
        # scenes = sorted(f.name for f in os.scandir(split_dir) if f.is_dir() and not f.name.startswith('.'))
        scene_features = glob.glob(os.path.join(split_dir, "*_features.npy"))
        items = []
        for scene_path in scene_features:
            scene = scene_path.replace("_features.npy", "")
            feature_path = os.path.join(split_dir, scene + "_features.npy")
            label_path = os.path.join(split_dir, scene + "_labels.npy")

            if not os.path.exists(feature_path) or not os.path.exists(label_path):
                print(feature_path)
                print(label_path)
                raise FileNotFoundError(f"Missing data files in {split_dir}")

            # with open(feature_path, "rb") as f:
            features = np.load(feature_path)  # ndarray(N_instance, 768)
            # with open(label_path, "rb") as f:
            labels = np.load(label_path)  # ndarray(N_instance)
        
            for i, label_id in enumerate(labels):
                if not label_id in VALID_CLASS_IDS:
                    continue
                item = Datum_feature(feature=features[i], label=int(ID_TO_PRED_ID[label_id]), classname=str(ID_TO_LABEL[label_id]))
                items.append(item)
        return items
    
    def subsample_classes(self, *args, subsample="all"):
        """Divide classes into two groups. The first group
        represents base classes while the second group represents
        new classes.

        Args:
            args: a list of datasets, e.g. train, val and test.
            subsample (str): what classes to subsample.
        """
        assert subsample in ["all", "base", "new"]

        # if subsample == "all":
        #     return args
        
        dataset = args[0]  # train
        labels = set()
        for item in dataset:
            labels.add(item.label)
        dataset = args[1]  # val
        for item in dataset:
            labels.add(item.label)
        dataset = args[2]  # test
        for item in dataset:
            labels.add(item.label)
        labels = list(labels)
        labels.sort()
        n = len(labels)
        # Divide classes into two halves
        m = math.ceil(n / 2)

        print(f"SUBSAMPLE {subsample.upper()} CLASSES!")
        if subsample == "base":
            selected = labels[:m]  # take the first half
        elif subsample == "new":
            selected = labels[m:]  # take the second half
        else:
            selected = labels  # take all
        relabeler = {y: y_new for y_new, y in enumerate(selected)}
        
        output = []
        for dataset in args:
            dataset_new = []
            for item in dataset:
                if item.label not in selected:
                    continue
                item_new = Datum_feature(
                    feature=item.feature,
                    label=relabeler[item.label],
                    classname=item.classname
                )
                dataset_new.append(item_new)
            output.append(dataset_new)
        
        return output



class Datum_feature:
    """Data instance which defines the basic attributes.

    Args:
        feature (ndarray): openscene feature.
        label (int): class label.
        domain (int): domain label.
        classname (str): class name.
    """

    def __init__(self, feature=np.zeros((1, 728)), label=0, domain=0, classname=""):

        self._feature = feature
        self._label = label
        self._domain = domain
        self._classname = classname

    @property
    def feature(self):
        return self._feature

    @property
    def label(self):
        return self._label

    @property
    def domain(self):
        return self._domain

    @property
    def classname(self):
        return self._classname
