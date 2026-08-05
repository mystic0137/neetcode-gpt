import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        sentences = positive + negative

        words = [word for sentence in sentences for word in sentence.split()]
        
        token_map = {
            word: i
            for i, word in enumerate(sorted(set(words)), start=1)
        }

        padded_token_ids = []
        for sentence in sentences:
            temp = []
            for word in sentence.split():
                temp.append(token_map[word])
            padded_token_ids.append(torch.tensor(temp, dtype=float))
        
        padded_token_ids = nn.utils.rnn.pad_sequence(
            padded_token_ids,
            batch_first=True,
            padding_value=0
        )
        return padded_token_ids