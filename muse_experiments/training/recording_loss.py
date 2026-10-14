"""Recording-level L1 loss matching official W-bMAE.

Each recording contributes one score: the weighted mean of its present
bin MAEs. Empty bins are left out of both the numerator and the
denominator. A training step averages those recording scores, so a
recording with more windows does not get a larger share of the update.
"""

import numpy as np
import torch


def recording_groups(dataset):
    """Dataset positions grouped into whole recordings, in table order."""
    rows = np.asarray(dataset.indices)
    subjects = dataset.subjects
    sessions = dataset.sessions
    if len(rows) == 0:
        raise ValueError("dataset has no windows")
    groups = []
    start = 0
    for index in range(1, len(rows) + 1):
        if index < len(rows):
            current = int(rows[index])
            previous = int(rows[start])
            same = (
                subjects[current] == subjects[previous]
                and sessions[current] == sessions[previous]
            )
            if same:
                continue
        groups.append(list(range(start, index)))
        start = index
    return groups


def chunk_recordings(groups, order, size):
    """Yield successive groups of recordings.

    ``size`` is the number of recordings in one optimizer update. A
    short final chunk is still returned. Every recording in ``order``
    appears once.
    """
    if size < 1:
        raise ValueError("recordings per update must be at least 1")
    chunk = []
    for position in order:
        chunk.append(groups[int(position)])
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _bin_index(target, edges):
    """Bin ids matching ``np.digitize(..., right=False)``."""
    index = torch.zeros(
        target.shape, dtype=torch.long, device=target.device
    )
    for edge in list(edges)[1:-1]:
        index = index + (target >= float(edge)).long()
    return index


def recording_wbmae_loss(prediction, target, edges, weights):
    """Official per-recording W-bMAE for one recording, as a tensor.

    ``prediction`` and ``target`` are one window each, in seconds.
    """
    estimate = prediction.reshape(-1)
    actual = target.reshape(-1).to(dtype=estimate.dtype)
    if estimate.shape != actual.shape:
        raise ValueError("prediction and target lengths differ")
    edge_values = [float(edge) for edge in edges]
    weight_values = [float(weight) for weight in weights]
    if len(weight_values) != len(edge_values) - 1:
        raise ValueError("one weight is required per bin")
    in_range = (actual >= edge_values[0]) & (actual <= edge_values[-1])
    if not bool(in_range.any()):
        raise ValueError("recording has no targets inside the bins")
    index = _bin_index(actual, edge_values)
    error = torch.abs(estimate - actual)
    numerator = estimate.new_zeros(())
    denominator = estimate.new_zeros(())
    for bin_id, weight in enumerate(weight_values):
        mask = in_range & (index == bin_id)
        if not bool(mask.any()):
            continue
        numerator = numerator + error[mask].mean() * weight
        denominator = denominator + weight
    return numerator / denominator


def collate_positions(dataset, positions):
    """Stack one recording's windows in dataset order."""
    sequences = []
    masks = []
    targets = []
    subjects = []
    sessions = []
    for position in positions:
        sequence, mask, target, subject, session = dataset[position]
        sequences.append(sequence)
        masks.append(mask)
        targets.append(target)
        subjects.append(subject)
        sessions.append(session)
    return (
        torch.stack(sequences),
        torch.stack(masks),
        torch.stack(targets),
        subjects,
        sessions,
    )


def chunk_loss(model, dataset, chunk, device, edges, weights):
    """Mean of the recording losses in one optimizer update."""
    losses = []
    for positions in chunk:
        sequence, mask, target, _subject, _session = collate_positions(
            dataset, positions
        )
        estimate = model(sequence.to(device), mask.to(device))
        losses.append(
            recording_wbmae_loss(
                estimate, target.to(device), edges, weights
            )
        )
    return torch.stack(losses).mean()
