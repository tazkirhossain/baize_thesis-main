"""Network boundaries, frozen data integrity and actual enriched FL/model paths."""
from copy import deepcopy
from dataclasses import asdict, replace
import gzip
import hashlib
import json
from pathlib import Path
import socket
import sqlite3

import pytest

from src.data.enrichment.collect import collect_client
from src.data.enrichment.config import CollectionConfig, digest, pooled_digest, resolve_enrichment
from src.data.enrichment.extract import RobotsRules, extract_page, parse_sitemap
from src.data.enrichment.fetch import FetchResult, PublicFetcher, checked_url, public_addresses
from src.data.enrichment.inputs import character_input, token_inputs
from src.data.enrichment.store import CollectionStore, EnrichmentReader, check_snapshot

ROOT = Path(__file__).parents[1]


def settings():
    return {'mode': 'url_plus_metadata', 'snapshot_id': 'fixture-v1', 'collection': asdict(CollectionConfig())}


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_client(root, *, flip=False):
    directory = root/'client_000'
    splits = directory/'splits'
    splits.mkdir(parents=True)
    (directory/'manifest.json').write_text(json.dumps({'client_id': 0}))
    preprocessing = splits/'preprocessing.json'
    preprocessing.write_text(json.dumps({'partitioning': {'strategy': 'iid'}}))
    manifest = {'preprocessing_sha256': file_hash(preprocessing), 'splits': {}}
    for index, split in enumerate(('train', 'val', 'test')):
        path = splits/f'{split}.jsonl'
        rows = [{'source_row': index*10+i, 'url': f'https://example.org/{split}/{i}',
                 'label': int((i % 2) == (0 if flip else 1)), 'original_category': 'must-never-enter-input'} for i in range(3)]
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')
        manifest['splits'][split] = {'n_records': len(rows), 'sha256': file_hash(path)}
    (splits/'manifest.json').write_text(json.dumps(manifest))
    return directory


class FixtureFetcher:
    def __init__(self, *, blocked=False, robots_code=200):
        self.calls = []
        self.blocked, self.robots_code = blocked, robots_code

    def get(self, url, *, domain, limit, allow=None, delay=None):
        checked_url(url, domain)
        if allow:
            reason = allow(url)
            if reason:
                return FetchResult(reason, url)
        self.calls.append(url)
        if url.endswith('/robots.txt'):
            text = 'User-agent: *\nDisallow: /test/0\nSitemap: https://example.org/index.xml\n'
            if self.blocked:
                text = 'User-agent: *\nDisallow: /\n'
            return FetchResult('ok' if self.robots_code == 200 else 'http_error', url,
                               self.robots_code, text.encode(), 'text/plain')
        if url.endswith('/index.xml'):
            return FetchResult('ok', url, 200, b'<sitemapindex><sitemap><loc>https://example.org/child.xml.gz</loc></sitemap></sitemapindex>', 'application/xml')
        if url.endswith('/child.xml.gz'):
            body = b'<urlset><url><loc>https://example.org/health/articles</loc></url><url><loc>https://outside.net/secret</loc></url></urlset>'
            return FetchResult('ok', url, 200, gzip.compress(body), 'application/gzip')
        if url.endswith('/sitemap.xml'):
            return FetchResult('http_error', url, 404)
        html = b'<html lang="en"><head><title>Independent website</title><meta name="description" content="Example page summary"><meta name="rating" content="general"></head><body><h1>Page heading</h1><p>Useful content here.</p><script>PRIVATE SCRIPT TOKEN</script></body></html>'
        return FetchResult('ok', url, 200, html, 'text/html; charset=utf-8', len(html))


def test_streamed_collection_resume_freeze_keeps_every_row_and_original_split(tmp_path):
    directory = local_client(tmp_path)
    original = {p: p.read_bytes() for p in (directory/'splits').iterdir()}
    first = FixtureFetcher()
    report = collect_client(directory, settings(), limit=2, fetcher=first)
    assert report['status'] == 'partial' and report['n_records'] == 2
    with pytest.raises(ValueError, match='incomplete'):
        EnrichmentReader(directory, settings())
    second = FixtureFetcher()
    report = collect_client(directory, settings(), fetcher=second)
    assert report['status'] == 'complete' and report['n_records'] == 9
    assert report['excluded_records'] == 0
    assert report['page_status_counts'] == {'ok': 8, 'robots_disallowed': 1}
    assert len([u for u in first.calls+second.calls if u.endswith('robots.txt')]) == 1
    assert not any(u.endswith('/test/0') for u in first.calls+second.calls)
    assert not any('outside.net' in u for u in first.calls+second.calls)
    reader = EnrichmentReader(directory, settings())
    data = reader.read(0, 'train', 'https://example.org/train/0')
    assert data['title'] == 'Independent website' and 'Useful content' in data['text']
    assert '/health/articles' in data['sitemap_paths']
    assert 'PRIVATE SCRIPT' not in json.dumps(data) and 'must-never-enter-input' not in json.dumps(data)
    assert not {'label', 'original_category', 'source_row', 'collected_at'} & data.keys()
    with pytest.raises(ValueError, match='mismatched'):
        reader.read(0, 'test', 'https://example.org/train/0')
    reader.close()
    with pytest.raises(ValueError, match='frozen'):
        collect_client(directory, settings(), fetcher=FixtureFetcher())
    assert original == {p: p.read_bytes() for p in original}


def test_collection_is_label_independent_and_cannot_mix_source_or_settings(tmp_path):
    a, b = local_client(tmp_path/'a'), local_client(tmp_path/'b', flip=True)
    ra = collect_client(a, settings(), fetcher=FixtureFetcher())
    rb = collect_client(b, settings(), fetcher=FixtureFetcher())
    assert ra['input_digest'] == rb['input_digest']
    changed = settings()
    changed['collection']['max_text_chars'] = 1000
    with pytest.raises(ValueError, match='does not match'):
        check_snapshot(a, changed)
    path = a/'splits/train.jsonl'
    path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError, match='fingerprint'):
        CollectionStore(a, changed)


def test_missing_or_blocked_websites_are_retained_and_mutated_cache_is_rejected(tmp_path):
    directory = local_client(tmp_path)
    result = collect_client(directory, settings(), fetcher=FixtureFetcher(robots_code=503))
    assert result['n_records'] == 9 and result['page_status_counts'] == {'robots_unreachable': 9}
    assert result['missing_fields']['title'] == 9
    reader = EnrichmentReader(directory, settings())
    database = directory/'enrichment/fixture-v1/features.sqlite3'
    with database.open('ab') as stream:
        stream.write(b'tampered')
    with pytest.raises(ValueError, match='changed'):
        reader.assert_unchanged()
    reader.close()
    with pytest.raises(ValueError, match='checksum'):
        EnrichmentReader(directory, settings())


def test_collection_writer_is_exclusive_and_released_for_resume(tmp_path):
    directory = local_client(tmp_path)
    store = CollectionStore(directory, settings())
    with pytest.raises(ValueError, match='another collector'):
        CollectionStore(directory, settings())
    store.close()
    again = CollectionStore(directory, settings())
    again.close()


@pytest.mark.parametrize('code,allowed', [(404,True),(403,True),(429,False),(503,False),(0,False)])
def test_robots_unavailable_and_unreachable_policies(code, allowed):
    rule = RobotsRules({'status': 'http_error', 'http_status': code}, 'BaizeResearchBot/1.0')
    assert (rule.blocked('https://example.org/a') is None) == allowed


def test_robots_longest_group_merging_wildcards_allow_ties_and_unicode():
    rule = RobotsRules({'status':'ok', 'http_status':200, 'text': '''User-agent: *
Disallow: /
User-agent: BaizeResearchBot
Disallow: /private
Allow: /private/public
Disallow: /*.pdf$
Disallow: /café
User-agent: BaizeResearchBot
Allow: /same
Disallow: /same
'''}, 'BaizeResearchBot/1.0')
    assert rule.blocked('https://example.org/news') is None
    assert rule.blocked('https://example.org/private/file') == 'robots_disallowed'
    assert rule.blocked('https://example.org/private/public/file') is None
    assert rule.blocked('https://example.org/report.pdf') == 'robots_disallowed'
    assert rule.blocked('https://example.org/report.pdf/child') is None
    assert rule.blocked('https://example.org/caf%C3%A9') == 'robots_disallowed'
    assert rule.blocked('https://example.org/same') is None


def test_public_dns_checks_reject_private_mixed_and_mapped_addresses(monkeypatch):
    for addresses in [['127.0.0.1'], ['10.0.0.1'], ['169.254.169.254'], ['8.8.8.8','192.168.0.1'], ['::ffff:8.8.8.8']]:
        monkeypatch.setattr(socket, 'getaddrinfo', lambda *a, **kw: [(socket.AF_INET, socket.SOCK_STREAM, 6, '', (ip,443)) for ip in addresses])
        with pytest.raises(ValueError, match='non_public'):
            public_addresses('example.org', 443)


@pytest.mark.parametrize('target', ['http://localhost/','http://127.0.0.1/','https://outside.net/',
    'file:///tmp/private', 'https://name:password@example.org/', 'https://example.org:8888/'])
def test_redirect_policy_rechecks_targets_before_connecting(monkeypatch, target):
    calls=[]
    fetcher = PublicFetcher(CollectionConfig())
    def request(url, *args):
        calls.append(url)
        return 302, {'location':target}, b''
    monkeypatch.setattr(fetcher, '_request', request)
    result = fetcher.get('https://example.org/', domain='example.org', limit=1000)
    assert result.status == 'unsafe_or_cross_domain_url' and len(calls) == 1


def test_redirect_checks_robots_on_the_destination(monkeypatch):
    fetcher = PublicFetcher(CollectionConfig())
    calls=[]
    monkeypatch.setattr(fetcher, '_request', lambda url,*a: (calls.append(url) or (302, {'location':'/private'}, b'')))
    result=fetcher.get('https://example.org/', domain='example.org', limit=1000,
                       allow=lambda u: 'robots_disallowed' if u.endswith('/private') else None)
    assert result.status == 'robots_disallowed' and len(calls) == 1


def test_sitemaps_reject_entities_gzip_bombs_and_cross_domain_entries():
    cfg = replace(CollectionConfig(), max_response_bytes=200)
    for body in [b'<!DOCTYPE a [<!ENTITY x "data">]><urlset/>', '<!DOCTYPE a><urlset/>'.encode('utf-16'),
                 gzip.compress(b'x'*10000)]:
        with pytest.raises(ValueError):
            parse_sitemap(body, domain='example.org', config=cfg)
    kind, urls = parse_sitemap(b'<urlset><url><loc>https://example.org/a?token=private</loc></url><url><loc>https://outside.net/b</loc></url></urlset>', domain='example.org',config=cfg)
    assert kind == 'urlset' and urls == ['https://example.org/a']


def test_metadata_parsing_is_bounded_handles_missing_attributes_and_does_not_execute():
    cfg = replace(CollectionConfig(), max_title_chars=8, max_text_chars=12)
    result = extract_page(b'<html lang=en><title>Long website title</title><meta name=description><style>hidden</style><body>Visible words for page<script>hidden()</script></body>',cfg)
    assert len(result['title']) <= 8 and len(result['text']) <= 12
    assert 'hidden' not in json.dumps(result) and result['language'] == 'en'


def test_character_budgets_keep_every_field_even_with_a_very_long_url():
    text=character_input({'url':'https://example.org/'+'u'*2000,'title':'TITLEVALUE',
        'description':'DESCVALUE','text':'BODYVALUE','sitemap_paths':'/SITEVALUE',
        'page_status':'ok','robots_status':'ok','sitemap_status':'ok'},512)
    assert len(text) <= 512
    for marker in ('TITLEVALUE','DESCVALUE','BODYVALUE','SITEVALUE'):
        assert marker in text


def tokenizer_fixture(tmp_path):
    from transformers import BertTokenizer
    vocabulary = ['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]', 'url','title','meta','page','site',':',
                  'urlvalue','titlevalue','descvalue','bodyvalue','sitevalue','ok','missing']
    path = tmp_path/'vocab.txt'
    path.write_text('\n'.join(vocabulary)+'\n')
    return BertTokenizer(vocab_file=str(path), do_lower_case=True)


def test_deberta_token_budgets_preserve_metadata_and_attention_masks(tmp_path):
    pytest.importorskip('transformers')
    tokenizer = tokenizer_fixture(tmp_path)
    record = {'url': 'urlvalue '*500, 'title':'titlevalue', 'description':'descvalue',
              'text':'bodyvalue', 'sitemap_paths':'sitevalue', 'page_status':'ok'}
    inputs = token_inputs([record, {'url':'urlvalue'}], tokenizer, 64)
    assert inputs['input_ids'].shape[1] <= 64
    decoded = tokenizer.decode(inputs['input_ids'][0])
    for word in ('titlevalue','descvalue','bodyvalue','sitevalue'):
        assert word in decoded
    assert inputs['input_ids'][0,0] == tokenizer.cls_token_id
    assert inputs['attention_mask'][1].sum() < inputs['attention_mask'][0].sum()


def frozen_fixture(directory, enrichment):
    store = CollectionStore(directory, enrichment)
    for split in ('train','val','test'):
        for line in (directory/'splits'/f'{split}.jsonl').read_text().splitlines():
            record = json.loads(line)
            store.add_record(record['source_row'], split, {'url':record['url'], 'title':'titlevalue',
                'description':'descvalue', 'text':'bodyvalue', 'sitemap_paths':'sitevalue',
                'page_status':'ok', 'robots_status':'ok', 'sitemap_status':'ok'})
    store.write_manifest(complete=True)
    store.close()


@pytest.mark.parametrize('family', ['character_cnn','deberta_v3_base'])
@pytest.mark.parametrize('method', ['undersampling','oversampling','smote','ensemble_undersampling'])
@pytest.mark.parametrize('partition', ['iid','label_skew','category_skew'])
def test_enriched_five_client_fl_all_models_methods_and_scenarios(tmp_path, monkeypatch, family, method, partition):
    torch=pytest.importorskip('torch')
    pytest.importorskip('flwr')
    from scripts.federated import synthetic_clients
    from src.client.flower_client import URLFlowerClient
    from src.models.character_cnn import CharacterCNNConfig, CharacterLevelCNN
    from src.server.flower_strategy import FiveClientFedAvg
    from src.server.local_transport import run_local
    from src.evaluation.reporting import federated_rows
    from src.models.federated_state import StateCodec
    from src.server.checkpoints import load_checkpoint
    cfg=json.loads((ROOT/'config/federated/character_cnn.json').read_text())
    cfg.update(rounds=2,batch_size=16,evaluate_training=True,evaluate_final_test=True,progress_interval_seconds=0,
               model_selection={'source':'training_selection','fraction':.1,'metric':'adult_f1'}, website_enrichment=settings())
    cfg['model']['sequence_length']=64
    cfg['model']['characters']+=' \n'
    cfg['balancing'].update(method=method, ensemble_members=2, k_neighbors=3)
    cfg['experiment'].update(partition_strategy=partition,data_kind='synthetic')
    if method=='smote':
        cfg['representation']={'mode':'encoder_embeddings','warmup_epochs':1}
    synthetic_clients(tmp_path/'clients',selection_ready=True)
    for cid in range(5):
        frozen_fixture(tmp_path/'clients'/f'client_{cid:03d}',settings())
    before={p:file_hash(p) for p in (tmp_path/'clients').rglob('*.jsonl')}
    previous=torch.get_num_threads()
    torch.set_num_threads(2)
    if family=='deberta_v3_base':
        from transformers import DebertaV2Config
        from src.models.deberta_v3_base.model import DebertaV3BaseClassifier
        import src.models.deberta_v3_base.model as module
        tokenizer=tokenizer_fixture(tmp_path)
        monkeypatch.setattr(module,'load_tokenizer',lambda *a,**k: tokenizer)
        cfg['model']={'family':family,'source':'fixture-only','revision':'fixture-only','max_length':64}
        tiny=DebertaV2Config(vocab_size=tokenizer.vocab_size,hidden_size=12,num_hidden_layers=1,
            num_attention_heads=2,intermediate_size=24,max_position_embeddings=128,relative_attention=True,
            position_biased_input=False)
        def build():
            torch.manual_seed(42)
            return [DebertaV3BaseClassifier(tiny) for _ in range(2 if method=='ensemble_undersampling' else 1)]
    else:
        class TinyCNN(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc=torch.nn.Linear(len(cfg['model']['characters'])+1,2)
            def extract_features(self,x): return x.mean(dim=1)
            def classify_features(self,x): return self.fc(x)
            def feature_head(self): return self.fc
            def forward(self,x): return self.classify_features(self.extract_features(x))
        def build():
            torch.manual_seed(42)
            return [TinyCNN() for _ in range(2 if method=='ensemble_undersampling' else 1)]
    try:
        strategy=FiveClientFedAvg(cfg,build(),output=tmp_path/'server')
        clients=[URLFlowerClient(cid,tmp_path/'clients'/f'client_{cid:03d}',cfg,models=build()) for cid in range(5)]
        # Collection must never be invoked by fit/evaluate/selection, for any method.
        monkeypatch.setattr(PublicFetcher,'get',lambda *a,**k:pytest.fail('training attempted network access'))
        run_local(strategy,clients)
        assert len(strategy.round_reports)==2 and all('selection' in x and 'val' in x for x in strategy.round_reports.values())
        assert 'test' not in strategy.round_reports[1] and 'test' in strategy.round_reports[2]
        assert len(strategy.client_data_contracts)==5
        assert all('enrichment_digest' in item for item in strategy.client_data_contracts.values())
        assert before=={p:file_hash(p) for p in before}
        load_checkpoint(tmp_path/'server/best_model.npz',StateCodec(build()))
        report_settings=json.loads((ROOT/'config/reporting/comparison.json').read_text())
        rows,_=federated_rows(tmp_path/'server',report_settings)
        assert all(row['input_mode']=='url_plus_metadata' for row in rows)
        for client in clients:
            client.enrichment.close()
    finally:
        torch.set_num_threads(previous)


def test_resolved_config_preflight_and_corpus_digest_are_explicit(tmp_path):
    from src.server.experiment_config import read_json, resolve_experiment, check_plan
    cfg = read_json(ROOT/'config/experiment_website.json')
    plan = resolve_experiment(cfg, ROOT)
    assert plan.configuration['website_enrichment']['collection'] == settings()['collection']
    assert ' ' in plan.configuration['model']['characters']
    check_plan(plan, mode='server')  # Server never opens client snapshots or raw inputs.
    with pytest.raises(ValueError, match='snapshot missing'):
        check_plan(plan, mode='client', client_id=0, client_directory=local_client(tmp_path))
    check_plan(plan,mode='smoke')  # Explicit frozen synthetic website fixtures, never a hidden URL-only run.
    assert pooled_digest({'a':{'enrichment_digest':'0'*63+'2'},'b':{'enrichment_digest':'0'*63+'3'}})==pooled_digest({'c':{'enrichment_digest':'0'*63+'5'}})
    assert pooled_digest({}) is None


def test_fake_website_features_cannot_be_installed_on_a_real_client(tmp_path):
    from src.data.enrichment.synthetic import freeze_synthetic_client
    with pytest.raises(ValueError, match='synthetic'):
        freeze_synthetic_client(local_client(tmp_path), settings())


@pytest.mark.parametrize('headers,body,status', [
    ({'Content-Type':'text/html'}, b'hello', 'ok'),
    ({'Content-Length':'999999'}, b'hello', 'response_too_large'),
    ({'Content-Type':'text/html'}, b'x'*100, 'response_too_large'),
    ({'Content-Encoding':'gzip'}, b'compressed', 'unsupported_content_encoding')])
def test_public_transport_pins_dns_preserves_tls_and_caps_streams(monkeypatch, headers, body, status):
    import io
    import http.client
    import ssl
    resolved, connected, tls_names, requests = [], [], [], []
    def resolve(host, port, **kwargs):
        resolved.append(host)
        return [(socket.AF_INET,socket.SOCK_STREAM,6,'',('8.8.8.8',port))]
    class FakeSocket:
        def settimeout(self,value): pass
        def connect(self,address): connected.append(address)
        def shutdown(self,how): pass
        def close(self): pass
    class Response:
        status=200
        def __init__(self): self.stream=io.BytesIO(body)
        def getheaders(self): return list(headers.items())
        def read1(self,size): return self.stream.read(size)
    class Connection:
        def __init__(self,*a,**k): self.sock=None
        def request(self,method,url,headers): requests.append((method,url,headers))
        def getresponse(self): return Response()
        def close(self): pass
    class Context:
        def wrap_socket(self,sock,server_hostname):
            tls_names.append(server_hostname)
            return sock
    monkeypatch.setattr(socket,'getaddrinfo',resolve)
    monkeypatch.setattr(socket,'socket',lambda *a,**k:FakeSocket())
    monkeypatch.setattr(http.client,'HTTPConnection',Connection)
    monkeypatch.setattr(ssl,'create_default_context',Context)
    result=PublicFetcher(CollectionConfig()).get('https://example.org/a',domain='example.org',limit=16)
    assert result.status==status and resolved==['example.org'] and connected==[('8.8.8.8',443)]
    assert tls_names==['example.org'] and requests[0][2]['Accept-Encoding']=='identity'


def test_frozen_cli_is_idempotent_and_never_fetches_again(tmp_path,monkeypatch,capsys):
    from scripts.enrich_client import main
    directory=local_client(tmp_path)
    collect_client(directory,settings(),fetcher=FixtureFetcher())
    request={'website_enrichment':{'mode':'url_plus_metadata','snapshot_id':'fixture-v1',
                                 'collection_config':str(ROOT/'config/website_collection.json')}}
    path=tmp_path/'experiment.json'
    path.write_text(json.dumps(request))
    monkeypatch.setattr(PublicFetcher,'get',lambda *a,**k:pytest.fail('frozen CLI fetched again'))
    assert main(['--config',str(path),'--client-dir',str(directory)])==0
    assert '"network_used": false' in capsys.readouterr().out


def test_comparison_rejects_changed_enriched_inputs_even_with_same_snapshot_name():
    from src.evaluation.reporting import verify_comparability
    row={'scope':'federated global model','family':'character_cnn','data_kind':'real','seed':42,
         'partition':'iid','method':'undersampling','split':'val','round':1,'experiment_id':'a'}
    second=dict(row,partition='label_skew',experiment_id='b')
    cfg=json.loads((ROOT/'config/federated/character_cnn.json').read_text())
    cfg['website_enrichment']=settings()
    metadata={'a':{'configuration':cfg,'data_contracts':{'0':{'enrichment_digest':'1'*64}}},
              'b':{'configuration':deepcopy(cfg),'data_contracts':{'0':{'enrichment_digest':'2'*64}}}}
    with pytest.raises(ValueError,match='incompatible'):
        verify_comparability([row,second],metadata)


def reuse_target(source_root, destination):
    """Move records to different owners while preserving every URL's fixed split."""
    gathered={split:[] for split in ('train','val','test')}
    for cid in range(5):
        directory=source_root/f'client_{cid:03d}'
        for split in gathered:
            gathered[split].extend(json.loads(line) for line in (directory/'splits'/f'{split}.jsonl').read_text().splitlines())
    for cid in range(5):
        directory=destination/f'client_{cid:03d}'
        splits=directory/'splits'
        splits.mkdir(parents=True)
        (directory/'manifest.json').write_text(json.dumps({'client_id':cid,'synthetic':True}))
        preprocessing=splits/'preprocessing.json'
        preprocessing.write_text(json.dumps({'partitioning':{'strategy':'label_skew'}}))
        manifest={'preprocessing_sha256':file_hash(preprocessing),'splits':{}}
        for split,rows in gathered.items():
            path=splits/f'{split}.jsonl'
            selected=rows[(cid+1)%5::5]
            path.write_text(''.join(json.dumps(row)+'\n' for row in selected))
            manifest['splits'][split]={'n_records':len(selected),'sha256':file_hash(path)}
        (splits/'manifest.json').write_text(json.dumps(manifest))


def test_offline_reuse_preserves_cross_scenario_inputs_and_only_target_ownership(tmp_path,monkeypatch):
    from scripts.federated import synthetic_clients
    from src.data.enrichment.reuse import reuse_client_snapshot
    source, target=tmp_path/'iid',tmp_path/'label'
    synthetic_clients(source)
    source_manifests={}
    for cid in range(5):
        directory=source/f'client_{cid:03d}'
        frozen_fixture(directory,settings())
        source_manifests[str(cid)]=dict(enrichment_digest=check_snapshot(directory,settings())['input_digest'])
    reuse_target(source,target)
    monkeypatch.setattr(PublicFetcher,'get',lambda *a,**k:pytest.fail('offline reuse made a network request'))
    target_manifests={}
    for cid in range(5):
        directory=target/f'client_{cid:03d}'
        report=reuse_client_snapshot(source,directory,settings(),batch_size=2)
        assert report['status']=='complete'
        target_manifests[str(cid)]=dict(enrichment_digest=report['input_digest'])
        db=sqlite3.connect(directory/'enrichment/fixture-v1/features.sqlite3')
        stored={row[0] for row in db.execute('SELECT source_row FROM records')}
        expected={json.loads(line)['source_row'] for split in ('train','val','test')
                  for line in (directory/'splits'/f'{split}.jsonl').read_text().splitlines()}
        assert stored==expected
        db.close()
    assert pooled_digest(source_manifests)==pooled_digest(target_manifests)


def test_offline_reuse_rejects_changed_urls_or_holdout_membership(tmp_path):
    from scripts.federated import synthetic_clients
    from src.data.enrichment.reuse import reuse_client_snapshot
    source,target=tmp_path/'iid',tmp_path/'category'
    synthetic_clients(source)
    for cid in range(5):
        frozen_fixture(source/f'client_{cid:03d}',settings())
    reuse_target(source,target)
    directory=target/'client_000'
    path=directory/'splits/train.jsonl'
    rows=path.read_text().splitlines()
    first=json.loads(rows[0]); first['url']='https://different.example/record'
    rows[0]=json.dumps(first)
    path.write_text('\n'.join(rows)+'\n')
    manifest=json.loads((directory/'splits/manifest.json').read_text())
    manifest['splits']['train']['sha256']=file_hash(path)
    (directory/'splits/manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='same assigned URL'):
        reuse_client_snapshot(source,directory,settings())
    partial=json.loads((directory/'enrichment/fixture-v1/manifest.json').read_text())
    assert partial['status']=='partial' and partial['n_records']==0
