
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from clip_benchmark.models import load_clip, MODEL_TYPES

def run_classification(model, classifier, dataloader, device, amp=True):
    """
    Run zero-shot classifcation

    model: torch.nn.Module
        CLIP-like model with `encode_image` and `encode_text`
    
    classifier: torch.Tensor
        obtained from the function `zero_shot_classifier`
    
    dataloader: torch.utils.data.Dataloader 
    
    Returns
    -------
    (pred, true)  where
        - pred (N, C) are the logits
        - true (N,) are the actual classes
    """
    autocast = torch.cuda.amp.autocast if amp else suppress
    pred = []
    true = []
    nb = 0
    with torch.no_grad():
        for images, target in tqdm(dataloader):
            images = images.to(device)
            target = target.to(device)

            with autocast():
                # predict
                image_features = model.encode_image(images)
                image_features = F.normalize(image_features, dim=-1)
                logits = 100. * image_features @ classifier
            
            true.append(target.cpu())
            pred.append(logits.float().cpu())

    pred = torch.cat(pred)
    true = torch.cat(true)
    return pred, true

def zero_shot_classifier(model, tokenizer, classnames, templates, device, amp=True, cupl=False):
    """
    This function returns zero-shot vectors for each class in order
    to use it for zero-shot classification.


    model:
        CLIP-like model with `encode_text`

    tokenizer:
        text tokenizer, i.e. convert list of strings to torch.Tensor of integers

    classnames: list of str
        name of classes

    templates: list of str
        templates to use.

    Returns
    -------

    torch.Tensor of shape (N,C) where N is the number
    of templates, and C is the number of classes.
    """
    autocast = torch.cuda.amp.autocast if amp else suppress
    with torch.no_grad(), autocast():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            if cupl:
                texts = templates[classname]
            else:
                texts = [template.format(c=classname) for template in templates]
            texts = tokenizer(texts).to(device)  # tokenize
            class_embeddings = model.encode_text(texts)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights


def reject_based_on_softmax_response(logits, target):
    # Sort logits in descending order and get the corresponding indice
    max_logits, max_indices = torch.max(logits, dim=1)

    sorted_logits, indices_sort = torch.sort(max_logits, descending=False, stable=True)
    sorted_indices = max_indices[indices_sort]
    sorted_targets = target[indices_sort]

    return sorted_logits, sorted_indices, sorted_targets

def reject_based_on_least_confidence(logits, target):

    max_logits, max_indices = torch.max(logits, dim=1)
    max_logits = 100 - max_logits
    # Append logits, indices, and targets to lists
    sorted_logits, indices_sort = torch.sort(max_logits, descending=True, stable=True)
    sorted_indices = max_indices[indices_sort]
    sorted_targets = target[indices_sort]

    return sorted_logits, sorted_indices, sorted_targets

def reject_based_on_predictive_entropy(logits, target):
    ## REVISAR
    values, indices = torch.max(logits, dim=1)
    # Calculate the logarithm of values
    log_values = torch.log2(logits)

    num_classes = len(logits[0].size())
    entropy = -torch.sum(logits * log_values,  dim=1) / torch.log2(torch.tensor(num_classes, dtype=torch.float32))

    # Append logits, indices, and targets to lists
    sorted_logits, indices_sort = torch.sort(entropy, descending=False, stable=True)
    sorted_indices = indices[indices_sort]
    sorted_targets = target[indices_sort]

    return sorted_logits, sorted_indices, sorted_targets

def reject_based_on_margin_confidence(logits, target):
    
    # Calculate the top two values and indices for each sample in the batch
    top_values, top_indices = torch.topk(logits, k=2, dim=1)

    # Calculate the difference between the top two values for each sample in the batch
    difference = top_values[:, 0] - top_values[:, 1]

    # Retrieve the index of the top value
    top_index = top_indices[:, 0]

    # Append logits, indices, and targets to lists
    sorted_logits, indices_sort = torch.sort(difference, descending=False, stable=True)
    sorted_indices = top_index[indices_sort]
    sorted_targets = target[indices_sort]

    return sorted_logits, sorted_indices, sorted_targets



def reject_based_on_ratio_confidence(logits, target):

    # Calculate the top two values and indices for each sample in the batch
    top_values, top_indices = torch.topk(logits, k=2, dim=1)

    # Calculate the difference between the top two values for each sample in the batch
    ratio = top_values[:, 0] / top_values[:, 1]

    # Retrieve the index of the top value
    top_index = top_indices[:, 0]

    # Append logits, indices, and targets to lists
    sorted_logits, indices_sort = torch.sort(ratio, descending=False, stable=True)
    sorted_indices = top_index[indices_sort]
    sorted_targets = target[indices_sort]

    return sorted_logits, sorted_indices, sorted_targets

def reject_based_on_montecarlo_dropout(model, classifier, dataloader, device, amp, data, N=10):
    all_logits = []
    model.train()
    for resblock in model.visual.transformer.resblocks:
        resblock.attn.dropout = 0.1
    target = None
    for i in range(N):
         logits, target = run_classification(model, classifier, dataloader, device, amp=amp)
         all_logits.append(logits.cpu().numpy())

    # all_logits = np.array(all_logits)
    # # Calculate averaged prediction
    # mean_prediction = np.mean(all_logits, axis=0)
    # # Calculate averaged prediction
    # predicted_class = np.argmax(mean_prediction, axis=1)
    # # Calculate variance of predicted class
    # predicted_class_variance = torch.tensor([np.var(all_logits[:,i,predicted_class[i]]) for i in range(predicted_class.shape[0])])


    # # Sort logits in descending order and get the corresponding indice
    # sorted_logits, indices_sort = torch.sort(predicted_class_variance, descending=True, stable=True)
    # sorted_preds   = torch.tensor(predicted_class[indices_sort])
    # sorted_targets = target[indices_sort]

    sorted_Logits, sorted_preds, sorted_targets = calculate_variance_and_sort_logits_montecarlo(all_logits, target)

    rejection_percentages = np.arange(0. , 1., 0.05)
    non_rejected_accuracies, classification_qualities, rejection_qualities = compute_accuracy(sorted_logits, sorted_preds, sorted_targets, rejection_percentages)
    data['montecarlo_dropout'] = {}
    data['montecarlo_dropout']['non-rejected-accuracy'] = non_rejected_accuracies
    data['montecarlo_dropout']['classification-quality'] = classification_qualities
    data['montecarlo_dropout']['rejection-quality'] = rejection_qualities

    return data
   
    

def reject_based_on_montecarlo_patch_dropout(args, classifier, dataloader, device, amp, data, N=3):
    model, transform, tokenizer = load_clip(
            model_type=args.model_type,
            model_name=args.model,
            pretrained=args.pretrained,
            cache_dir=args.model_cache_dir,
            device=args.device,
            force_patch_dropout=0.5
        )

    target = None
    all_logits = []
    for i in range(N):
         logits, target = run_classification(model, classifier, dataloader, device, amp=amp)
         all_logits.append(logits.cpu().numpy())
  
    sorted_logits, sorted_preds, sorted_targets = calculate_variance_and_sort_logits_montecarlo(all_logits, target)

    rejection_percentages = np.arange(0. , 1., 0.05)
    non_rejected_accuracies, classification_qualities, rejection_qualities = compute_accuracy(sorted_logits, sorted_preds, sorted_targets, rejection_percentages)
    data['montecarlo_patch_dropout'] = {}
    data['montecarlo_patch_dropout']['non-rejected-accuracy'] = non_rejected_accuracies
    data['montecarlo_patch_dropout']['classification-quality'] = classification_qualities
    data['montecarlo_patch_dropout']['rejection-quality'] = rejection_qualities

    return data


def calculate_variance_and_sort_logits_montecarlo(all_logits, target):
    all_logits = np.array(all_logits)
    # Calculate averaged prediction
    mean_prediction = np.mean(all_logits, axis=0)
    # Calculate averaged prediction
    predicted_class = np.argmax(mean_prediction, axis=1)
    # Calculate variance of predicted class
    predicted_class_variance = torch.tensor([np.var(all_logits[:,i,predicted_class[i]]) for i in range(predicted_class.shape[0])])

    # Sort logits in descending order and get the corresponding indice
    sorted_logits, indices_sort = torch.sort(predicted_class_variance, descending=True, stable=True)
    sorted_preds   = torch.tensor(predicted_class[indices_sort])
    sorted_targets = target[indices_sort]

    return sorted_logits, sorted_preds, sorted_targets




def compute_classification_quality(A, N, M, R):
    numerator = A + M
    denominator = N + R
    cq = numerator / denominator
    return cq


def compute_rejection_quality(A, AR, M, MR):
    if AR != 0:
        numerator = MR / AR
    else:
        return float('inf')
    denominator = M / A
    if denominator != 0:

        rq = numerator / denominator
        return rq
    else:
        return 1

def compute_accuracy(sorted_logits, pred, sorted_targets, rejection_percentages):
    non_rejected_accuracies = []
    classification_qualities = []
    rejection_qualities = []

    for percentage_rejection in rejection_percentages:
        # Calculate the number of elements to reject
        num_reject = int(len(sorted_logits) * percentage_rejection)

        # Discard the first 'num_reject' elements
        not_rejected_pred = pred[num_reject:]
        not_rejected_targets = sorted_targets[num_reject:]

        rejected_pred = pred[:num_reject]
        rejected_targets = sorted_targets[:num_reject]

        # Compute accuracy in the not rejected dataset
        correct_not_rejected = (not_rejected_pred == not_rejected_targets).sum().item()
        total = len(not_rejected_pred)
        accuracy = correct_not_rejected / total * 100.0

        #Compute missclassified samples in the rejected dataset
        correct_rejected = (rejected_pred == rejected_targets).sum().item()
        missclassified_rejected = len(rejected_pred) - correct_rejected
        non_rejected_accuracies.append(accuracy)

        #Compute accuracy in all dataset
        accuracy_all_dataset =  (pred == sorted_targets).sum().item()
        missclassified_all_dataset = len(pred) - accuracy_all_dataset

        classification_quality = compute_classification_quality(correct_not_rejected, total, missclassified_rejected, num_reject) * 100.0
        classification_qualities.append(classification_quality)

        rejection_quality = compute_rejection_quality(accuracy_all_dataset, correct_rejected, missclassified_all_dataset, missclassified_rejected)  * 100.0
        rejection_qualities.append(rejection_quality)

    return non_rejected_accuracies, classification_qualities, rejection_qualities

def compute_ensembles():
    pass
