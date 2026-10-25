"""Client-side fixed encoders and lazy URL datasets. Never sent to the server."""
import torch


class LocalURLDataset:
    def __init__(self, split, enrichment=None):
        self.local = split
        self.split, self.labels = split.split, split.labels
        self.enrichment = enrichment

    def __len__(self):
        return len(self.local)

    def __getitem__(self, index):
        url = self.local.urls[index]
        features = self.enrichment.read(self.local.source_rows[index], self.split, url) if self.enrichment else url
        return features, int(self.labels[index])


def make_collator(model_cfg, enrichment=None):
    if model_cfg["family"] == "character_cnn":
        alphabet = {char: index for index, char in enumerate(model_cfg["characters"])}
        length, unknown = model_cfg["sequence_length"], len(alphabet)

        def encode(urls):
            if enrichment:
                from src.data.enrichment.inputs import character_input
                urls = [character_input(record, length) for record in urls]
            x = torch.zeros(len(urls), length, unknown + 1)
            for row, url in enumerate(urls):
                for column, char in enumerate(url[:length]):
                    x[row, column, alphabet.get(char, unknown)] = 1
            return {"x": x}
    else:
        from src.models.deberta_v3_base.model import load_tokenizer, tokenize_urls
        tokenizer = load_tokenizer(model_cfg["source"], revision=model_cfg["revision"])

        def encode(urls):
            if enrichment:
                from src.data.enrichment.inputs import token_inputs
                return token_inputs(urls, tokenizer, model_cfg['max_length'])
            return tokenize_urls(tokenizer, urls, max_length=model_cfg["max_length"])

    def collate(rows):
        urls, labels = zip(*rows)
        return encode(urls), torch.tensor(labels, dtype=torch.long)

    return collate
