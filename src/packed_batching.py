from typing import Iterable, List, Tuple

def drop(
    current: List[int],
    current_mask: List[int],
    packed_rows: List[List[int]],
    masks: List[List[int]],
    max_length: int,
    pad_id: int
):
    """Сбрасывает текущий батч в общую коллекцию."""
    if not current:
        return current, current_mask, packed_rows, masks
    
    pad_size = max_length - len(current)
    packed_rows.append(current + [pad_id] * pad_size)
    masks.append(current_mask + [0] * pad_size)
    
    return [], [], packed_rows, masks

def packing(
    sequences: Iterable[List[int]], 
    max_length: int = 512, 
    pad_id: int = 0
) -> Tuple[List[List[int]], List[List[int]]]:
    """Упаковывает последовательности в packed batches."""
    packed_rows: List[List[int]] = []
    masks: List[List[int]] = []
    current: List[int] = []
    current_mask: List[int] = []
    segment_id = 1

    for sequence in sequences:
        start = 0
        while start < len(sequence):
            remaining = max_length - len(current)
            if remaining == 0:
                current, current_mask, packed_rows, masks = drop(
                    current, current_mask, packed_rows, masks, max_length, pad_id
                )
                remaining = max_length
            
            chunk = sequence[start:start + remaining]
            current.extend(chunk)
            current_mask.extend([segment_id] * len(chunk))
            start += len(chunk)
            
            if start < len(sequence) or len(current) == max_length:
                current, current_mask, packed_rows, masks = drop(
                    current, current_mask, packed_rows, masks, max_length, pad_id
                )
            else:
                segment_id += 1
    
    current, current_mask, packed_rows, masks = drop(
        current, current_mask, packed_rows, masks, max_length, pad_id
    )
    
    return packed_rows, masks