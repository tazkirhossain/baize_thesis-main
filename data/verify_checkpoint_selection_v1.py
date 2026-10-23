"""Independent synthetic artifact audit; reads no real corpus or private URLs aloud."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from src.client.data import read_local_split
from src.client.selection import selection_mask,selection_views,membership_summary
from src.data.preprocessing import _file_hash
from src.models.federated_state import StateCodec,make_models,state_hash
from src.server.config import contract_hash
from src.server.checkpoints import load_checkpoint
from src.evaluation.reporting import checked_selection

run_root=ROOT/'data/checkpoint-selection-smoke-v1'
server=run_root/'server'
run=json.loads((server/'run.json').read_text())
cfg=run['configuration']
assert run['status']=='PASS' and cfg['rounds']==2 and cfg['model_selection']['source']=='training_selection'
assert not cfg['evaluate_final_test']
codec=StateCodec(make_models(cfg))
history=[]
previous=run['initial_global_hash']
for rnd in range(1,3):
    report=json.loads((server/f'round_{rnd:04d}.json').read_text())
    arrays,metadata=load_checkpoint(server/f'checkpoints/round_{rnd:04d}.npz',codec,expected_contract=contract_hash(cfg))
    assert metadata['status']=='evaluated' and all(client['base_hash']==previous for client in report['clients'])
    assert report['global_hash']==state_hash(arrays)==metadata['global_hash']
    assert report['train']['n_records']+report['selection']['n_records']==200
    assert report['optimization']['examples_processed']==242
    assert sum(row['used_adult'] for row in report['clients'])==121
    assert sum(row['used_non_adult'] for row in report['clients'])==121
    assert 'test' not in report
    previous=report['global_hash']
    history.append(report)
checked_selection(cfg,run,history)
best,metadata=load_checkpoint(server/'best_model.npz',codec,expected_contract=contract_hash(cfg))
assert state_hash(best)==run['selection']['global_hash']==metadata['global_hash']
with np.load(server/'global_final.npz',allow_pickle=False) as stored:
    final=[stored[f'tensor_{i:04d}'] for i in range(len(codec.entries))]
codec.validate(final)
assert state_hash(final)==run['global_hash']==previous
for cid in range(5):
    directory=run_root/'synthetic-clients'/f'client_{cid:03d}'
    for name in ('train','val','test'):
        manifest=json.loads((directory/'splits/manifest.json').read_text())
        assert _file_hash(directory/f'splits/{name}.jsonl')==manifest['splits'][name]['sha256']
    original=read_local_split(directory,'train')
    train,selection=selection_views(original,selection_mask(original,fraction=.1,seed=cfg['seed'],synthetic=True))
    assert len(train)+len(selection)==40 and not set(train.source_rows)&set(selection.source_rows)
    summary=membership_summary(train,selection)
    for report in history:
        row=next(item for item in report['clients'] if item['client_id']==cid)
        assert row['assigned_train']==40 and row['original_train']==len(train) and row['selection_train']==len(selection)
        assert row['aggregation_weight']==len(train)
        assert run['client_data_contracts'][str(cid)]['selection_membership_sha256']==summary['selection']['membership_sha256']
report=json.loads((run_root/'report/report.json').read_text())
assert len(report['rows'])==4 and len(report['plots'])==6
for name,digest in report['plots'].items():
    assert _file_hash(run_root/'report'/name)==digest
result={'status':'PASS','data_kind':'synthetic','model':'actual Character CNN',
        'rounds':2,'clients':5,'round_checkpoints':2,'best_checkpoint_round':run['selection']['round'],
        'best_selection_adult_f1':run['selection']['score'],'selection_only':True,
        'all_client_files_hash_verified':True,'trained_and_selection_records_disjoint':True,
        'assigned_training_per_round':200,'optimization_per_round':history[0]['train']['n_records'],
        'selection_per_round':history[0]['selection']['n_records'],'test_evaluated':False,
        'balanced_adult_per_round':121,'balanced_non_adult_per_round':121,
        'best_and_final_checksums_verified':True,'automatic_report_rows':4,'plot_files':6,
        'quality_claim':'execution/integrity evidence only; not real-corpus model quality'}
(run_root/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
