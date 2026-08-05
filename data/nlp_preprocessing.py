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
        words = []
        for string in (positive + negative):
            words.extend(string.split())
        words = sorted(set(words))

        token_map = {}
        for i, word in enumerate(words, start=1):
            token_map[word] = i
        
        embeddings = []
        for string in (positive + negative):
            temp = []
            for word in string.split():
                temp.append(token_map[word])
            embeddings.append(torch.tensor(temp, dtype=float))
        
        embedding_tensors = nn.utils.rnn.pad_sequence(
            embeddings,
            batch_first=True,
            padding_value=0
        )
        return embedding_tensors