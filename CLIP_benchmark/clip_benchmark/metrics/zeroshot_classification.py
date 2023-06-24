"""
Code adapated from https://github.com/mlfoundations/open_clip/blob/main/src/training/zero_shot.py
Thanks to the authors of OpenCLIP
"""
import logging
from contextlib import suppress
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from sklearn.metrics import classification_report, balanced_accuracy_score


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


def accuracy(output, target, topk=(1,)):
    """
    Compute top-k accuracy

    output: torch.Tensor
        shape (N, C) where N is the number of examples, C the number of classes.
        these are the logits.
    
    target: torch.Tensor
        shape (N,) where N is the number of examples. Groundtruth class id of each example.
    
    topk: tuple
        which topk to compute, e.g., topk=(1,5) will compute top-1 and top-5 accuracies
    
    Returns
    -------
    
    list of top-k accuracies in the same order as `topk`
    """
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    n = len(target)
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) / n for k in topk]


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



def average_precision_per_class(scores, targets):
    """
    Compute average precision  for each class
    this metric is used for multi-label classification
    see explanations here https://fangdahan.medium.com/calculate-mean-average-precision-map-for-multi-label-classification-b082679d31be
    Code is adapted from https://github.com/pytorch/tnt/blob/master/torchnet/meter/meter.py, thanks to the authors of `tnt`.

    Parameters
    ----------

    scores: torch.Tensor
        logits, of shape (N,C) where N is the number of examples, C the number of classes
    
    targets: torch.Tensor
        one-hot vectors of groundtruth targets (N, C), where N is the number of examples, C is the
        number of classes
    
    Returns
    -------

    torch.Tensor of shape (C,) of avereage precision for each class, where C is     
    the number of classes.
    
    """
    ap = torch.zeros(scores.size(1))
    rg = torch.arange(1, scores.size(0) + 1).float()
    # compute average precision for each class
    for k in range(scores.size(1)):
        # sort scores
        scores_k = scores[:, k]
        targets_k = targets[:, k]
        _, sortind = torch.sort(scores_k, 0, True)
        truth = targets_k[sortind]
        tp = truth.float().cumsum(0)
        # compute precision curve
        precision = tp.div(rg)
        # compute average precision
        ap[k] = precision[truth.bool()].sum() / max(float(truth.sum()), 1)
    return ap


def evaluate(model, dataloader, tokenizer, classnames, templates, device, amp=True, verbose=False, cupl=False, save_clf=None, load_clfs=[]):
    """
    Run zero-shot classification and evaluate the metrics

    Parameters
    ----------

    model: torch.nn.Module
        CLIP-like model with `encode_image` and `encode_text`
    
    dataloader: torch.utils.data.Dataloader

    tokenizer: text tokenizer

    classnames: list of str
        class names
    
    templates: list of str
        templates to use for zero-shot classification
    
    device: cpu/cuda

    amp: whether to use automatic mixed precision

    verbose: whether to use verbose model

    Returns
    -------

    dict of classification metrics
    """
    if len(load_clfs) > 0:
        n = len(load_clfs)
        classifier = torch.load(load_clfs[0], map_location='cpu') / n
        for i in range(1, n):
            classifier = classifier + torch.load(load_clfs[i], map_location='cpu') / n
        classifier = classifier.to(device)
    else:
        classifier = zero_shot_classifier(model, tokenizer, classnames, templates, device, cupl=cupl)
    
    if save_clf is not None:
        torch.save(classifier, save_clf)
        # exit() - not sure if we want to exit here or not.
    logits, target = run_classification(model, classifier, dataloader, device, amp=amp)
    is_multilabel = (len(target.shape) == 2)

    if is_multilabel:
        if verbose:
            print("Detected a multi-label classification dataset")
        # Multiple labels per image, multiple classes on the dataset
        ap_per_class = average_precision_per_class(logits, target)
        if verbose:
            for class_name, ap in zip(dataloader.dataset.classes, ap_per_class.tolist()):
                print(f"Class: {class_name}, AveragePrecision: {ap}")
        return {"mean_average_precision": ap_per_class.mean().item()}
    else:
        # Single label per image, multiple classes on the dataset
        # just compute accuracy and mean_per_class_recall

        pred = logits.argmax(axis=1)
        # measure accuracy
        if len(dataloader.dataset.classes) >= 5:
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        else:
            acc1, = accuracy(logits, target, topk=(1,))
            acc5 = float("nan") 
        mean_per_class_recall = balanced_accuracy_score(target, pred)
        if verbose:
            print(classification_report(target, pred, digits=3))

        # TODO: add different rejection strategies here

        compute_rejection(logits, target)
        reject_based_on_montecarlo_dropout(model, classifier, dataloader, device, amp=True)
        return {"acc1": acc1, "acc5": acc5, "mean_per_class_recall": mean_per_class_recall}


def compute_rejection(logits, target):

    methods = [reject_based_on_softmax_response, reject_based_on_least_confidence, reject_based_on_predictive_entropy, reject_based_on_margin_confidence, reject_based_on_ratio_confidence]

    for method in methods:
         # Call the process_function to get sorted 
        sorted_logits, pred, sorted_targets = method(logits, target)
        rejection_percentages = np.arange(0. , 1., 0.05)
        accuracies  = compute_accuracy(sorted_logits, pred, sorted_targets, rejection_percentages)
        print("Method:", method.__name__)
        print(accuracies)


def compute_accuracy(sorted_logits, pred, sorted_targets, rejection_percentages):
    accuracies = []

    for percentage_rejection in rejection_percentages:
        # Calculate the number of elements to reject
        num_reject = int(len(sorted_logits) * percentage_rejection)
        # Discard the first 'num_reject' elements
        filtered_logits = sorted_logits[num_reject:]
        filtered_pred = pred[num_reject:]
        filtered_targets = sorted_targets[num_reject:]
        # Compute accuracy between filtered indices and targets
        correct = (filtered_pred == filtered_targets).sum().item()
        total = len(filtered_pred)
        accuracy = correct / total * 100.0

        accuracies.append(accuracy)

    return accuracies


def reject_based_on_softmax_response(logits, target):

    # Sort logits in descending order and get the corresponding indice
    max_logits, max_indices = torch.max(logits, dim=1)

    sorted_logits, indices_sort = torch.sort(max_logits, descending=False)
    sorted_indices = max_indices[indices_sort]
    sorted_targets = target[indices_sort]

    return sorted_logits, sorted_indices, sorted_targets

def reject_based_on_least_confidence(logits, target):

    max_logits, max_indices = torch.max(logits, dim=1)
    max_logits = 100 - max_logits
    # Append logits, indices, and targets to lists
    sorted_logits, indices_sort = torch.sort(max_logits, descending=True)
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
    sorted_logits, indices_sort = torch.sort(entropy, descending=False)
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
    sorted_logits, indices_sort = torch.sort(difference, descending=False)
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
    sorted_logits, indices_sort = torch.sort(ratio, descending=False)
    sorted_indices = top_index[indices_sort]
    sorted_targets = target[indices_sort]

    return sorted_logits, sorted_indices, sorted_targets

def reject_based_on_montecarlo_dropout(model, classifier, dataloader, device, amp, N=3):
    all_pred = []
    model.train()
    for resblock in model.visual.transformer.resblocks:
        resblock.attn.dropout = 0.1
    
    for i in range(N):
         logits, target = run_classification(model, classifier, dataloader, device, amp=amp)
         pred = logits.argmax(axis=1)
         all_pred.append(pred)

    index_freq = {}
    print(tensor)
    for tensor in all_pred:
         for index, value in enumerate(tensor):
            if index not in index_freq:
                index_freq[index] = {value: 1}
            else:
                if value not in index_freq[index]:
                    index_freq[index][value] = 1
                else:
                    index_freq[index][value] += 1

    # Step 2: Find the element with the highest frequency at each index
    max_frequent_elements_list = [max(freq, key=freq.get) for index, freq in index_freq.items()]
    max_frequent_elements = torch.tensor(max_frequent_elements_list)
    print(max_frequent_elements)
    # Step 3: Compute the frequency ratio of the maximum frequent element
    num_tensors = len(all_pred)
    ratio_max_frequent_elements_list = [max(freq, key=freq.get) for index, freq in index_freq.items()]
    max_frequent_values_list = [max(freq.values()) for freq in index_freq.values()]

    frequency_ratios = torch.tensor([freq[max_frequent_elements_list[index]] / num_tensors for index, freq in index_freq.items()])
    print(frequency_ratios)



