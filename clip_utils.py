import clip
import torch
import torch.nn.functional as F
import numpy as np

MATTERPORT_LABELS_21 = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
                    'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator',
                    'shower curtain', 'toilet', 'sink', 'bathtub', 'other', 'ceiling')
REPLICA_LABELS = ("basket", "bed", "bench", "bin", "blanket", "blinds", "book", "bottle", "box", "bowl", "camera", "cabinet", "candle",
                  "chair", "clock", "cloth", "comforter", "cushion", "desk", "desk-organizer", "door", "indoor-plant", "lamp", "monitor",
                  "nightstand", "panel", "picture", "pillar", "pillow", "pipe", "plant-stand", "plate", "pot", "sculpture", "shelf", "sofa", 
                  "stool", "switch", "table", "tablet", "tissue-paper", "tv-screen", "tv-stand", "vase", "vent", "wall-plug", "window",
                  "rug")
VALID_CLASS_IDS = np.asarray([3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 26, 29, 34, 35, 37, 44, 47, 52, 54, 56, 59, 60, 61, 62, 63, 64, 65, 70, 71, 76, 78, 79, 80, 82, 83, 87, 88, 91, 92, 95, 97, 98])
ID_TO_LABEL = {}
LABEL_TO_ID = {}
PRED_ID_TO_ID = {}
ID_TO_PRED_ID = {}
for pred_id, i in enumerate(range(len(VALID_CLASS_IDS))):
    PRED_ID_TO_ID[pred_id] = VALID_CLASS_IDS[i]
    ID_TO_PRED_ID[VALID_CLASS_IDS[i]] = pred_id
    LABEL_TO_ID[REPLICA_LABELS[i]] = VALID_CLASS_IDS[i]
    ID_TO_LABEL[VALID_CLASS_IDS[i]] = REPLICA_LABELS[i]

# PRED_ID_TO_ID[-1] = -1
PRED_ID_TO_ID[48] = -1 

SKIPPED_CLASSES = ["undefined", "floor", "ceiling", "wall"]

def gt_ids_to_label(gt_id):
    transformed_id = int(gt_id // 1000)
    return ID_TO_LABEL[transformed_id]

def gt_ids_to_id(gt_id):
    transformed_id = int(gt_id // 1000)
    if transformed_id in ID_TO_PRED_ID:
        return ID_TO_PRED_ID[transformed_id]
    else:
        return -1
    
def get_label(index):
    if 0 <= index <= len(REPLICA_LABELS):
        return REPLICA_LABELS[index]
    elif index == -1:
        return "undefined"
    else:
        raise IndexError("Index out of bounds.")

def get_clip_model(model_name):
    print("Loading CLIP {} model...".format(model_name))
    clip_pretrained, _ = clip.load(model_name, device='cuda', jit=False)
    print("Finish loading")
    return clip_pretrained
    
    
def extract_text_feature(labelset):
    '''extract CLIP text features.'''

    # a bit of prompt engineering
    print('Use prompt engineering: a XX in a scene')
    labelset = [ "a " + label + " in a scene" for label in labelset]
    
    model_name="ViT-B/32"
    # "ViT-L/14@336px" # the big model that OpenSeg uses
    clip_model = get_clip_model(model_name)

    if isinstance(labelset, str):
        lines = labelset.split(',')
    elif isinstance(labelset, list):
        lines = labelset
    else:
        raise NotImplementedError

    labels = []
    for line in lines:
        label = line
        labels.append(label)
    text = clip.tokenize(labels)
    text = text.cuda()
    text_features = clip_model.encode_text(text)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features.detach().cpu().float(), labelset


def classify_features(text_features, instance_features, normalize=True):
    """
    Classify instance feature vectors based on similarity to text feature vectors.

    Args:
        text_features (torch.Tensor): A tensor containing feature vectors for text (e.g., CLIP-encoded class embeddings).
                                      Shape: (C, D), where C is the number of classes and D is the feature dimension.
        instance_features (torch.Tensor): A tensor containing feature vectors for instances/groups.
                                          Shape: (N, D), where N is the number of instances and D is the feature dimension.
        normalize (bool): Whether to normalize the feature vectors before computing similarities. Default is True.

    Returns:
        predicted_classes (torch.Tensor): Predicted class indices for each instance. Shape: (N,)
        confidence_scores (torch.Tensor): Confidence scores (max probability) for each instance. Shape: (N,)
    """
    if normalize:
        text_features= F.normalize(text_features, dim=1)
        instance_features = F.normalize(instance_features, dim=1)
    
    cosine_similarities = torch.matmul(instance_features, text_features.T)
    
    predicted_probs = F.softmax(cosine_similarities, dim=1)  # Softmax over classes
    predicted_classes = torch.argmax(predicted_probs, dim=1)  # Predicted class indices
    confidence_scores = torch.max(predicted_probs, dim=1)[0]  # Confidence scores
    
    return predicted_classes, confidence_scores