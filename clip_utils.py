import clip
import torch
import torch.nn.functional as F

MATTERPORT_LABELS_21 = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
                    'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator',
                    'shower curtain', 'toilet', 'sink', 'bathtub', 'other', 'ceiling')

def extract_text_feature(labelset):
    '''extract CLIP text features.'''

    # a bit of prompt engineering
    print('Use prompt engineering: a XX in a scene')
    labelset = [ "a " + label + " in a scene" for label in labelset]
    
    model_name="ViT-B/32"
    # "ViT-L/14@336px" # the big model that OpenSeg uses
    print("Loading CLIP {} model...".format(model_name))
    clip_pretrained, _ = clip.load(model_name, device='cuda', jit=False)
    print("Finish loading")

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
    text_features = clip_pretrained.encode_text(text)
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