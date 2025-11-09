import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import json
from typing import List

class KbertDataset(Dataset):
    """將 kbert_train_data.jsonl 轉換為 PyTorch Dataset，支援多標籤分類。"""
    def __init__(self, data_list: list, tokenizer: BertTokenizer, max_len: int):
        self.data = data_list
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        text = sample['text']
        # ⚠️ 假設 label 是一個包含 4 個元素的列表，例如 [0, 1, 0, 0]
        label: List[int] = sample['label'] 
        # triplets = sample['triplets'] # 知識三元組數據（目前未使用）

        # 1. Tokenization：標準 BERT 文本處理
        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_attention_mask=True,
            return_token_type_ids=True
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        token_type_ids = encoding['token_type_ids'].squeeze(0)

        # 2. K-BERT 核心：生成 Visible Matrix (簡化版)
        visible_matrix = torch.ones((self.max_len, self.max_len), dtype=torch.long)
        
        # 3. 標籤轉換：多標籤分類 (BCEWithLogitsLoss) 需要浮點數標籤
        labels_tensor = torch.tensor(label, dtype=torch.float)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels': labels_tensor, # 使用 float 類型
            'visible_matrix': visible_matrix 
        }

def create_kbert_dataloader(data: list, tokenizer: BertTokenizer, max_len: int, batch_size: int):
    """建立 K-BERT 專用的 DataLoader"""
    dataset = KbertDataset(data, tokenizer, max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)