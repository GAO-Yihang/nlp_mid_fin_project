import json
from typing import Dict, Any, List
from torch.utils.data import Dataset

class JsonlIdsDataset(Dataset):
    """
    Reads preprocessed data.ids.jsonl
    Each line: {"src_ids":[...], "tgt_ids":[...], "index": ...}
    """
    def __init__(self, ids_jsonl_path: str):
        self.items: List[Dict[str, Any]] = []
        with open(ids_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.items.append(obj)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]
