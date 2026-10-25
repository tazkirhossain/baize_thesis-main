"""Fixed field budgets, never fitted on evaluation data or learned from labels."""


def sections(record):
    return [('url', record['url']),
            ('title', record.get('title', '')),
            ('meta', ' '.join(record.get(k, '') for k in ('rating', 'language', 'description', 'keywords'))),
            ('page', record.get('text', '')),
            ('site', ' '.join(record.get(k, '') for k in ('page_status', 'robots_status', 'sitemap_status', 'sitemap_paths')))]


def character_input(record, length):
    # Each field is guaranteed space; metadata is not appended after a full URL.
    groups = sections(record)
    remaining = length-sum(len(name)+2 for name, _ in groups)
    if remaining < 8:
        raise ValueError('enriched Character CNN requires sequence_length >= 64')
    shares = [.25, .15, .25, .20, .15]
    budgets = [int(remaining*x) for x in shares[:-1]]
    budgets.append(remaining-sum(budgets))
    return ''.join(f'{name}:{text[:budget]}\n' for (name, text), budget in zip(groups, budgets))


def token_inputs(records, tokenizer, max_length):
    special = tokenizer.num_special_tokens_to_add(pair=False)
    # Four independent token budgets prevent a URL/body from consuming all tokens.
    remaining = max_length-special
    if remaining < 16:
        raise ValueError('enriched DeBERTa requires max_length >= 32')
    shares = [.25, .15, .25, .20, .15]
    budgets = [int(remaining*x) for x in shares[:-1]]
    budgets.append(remaining-sum(budgets))
    features = []
    for record in records:
        ids = []
        for (name, text), budget in zip(sections(record), budgets):
            # Character cap also bounds tokenization of adversarial pages.
            ids.extend(tokenizer.encode(f'{name}: {text or "missing"}'[:max_length*16],
                add_special_tokens=False, truncation=True, max_length=budget))
        ids = tokenizer.build_inputs_with_special_tokens(ids)
        features.append({'input_ids': ids, 'attention_mask': [1]*len(ids)})
    return tokenizer.pad(features, padding=True, return_tensors='pt')
