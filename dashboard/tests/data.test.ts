import { strict as assert } from 'node:assert';
import { test } from 'node:test';
import { compatible, latestRound, resultsCsv, roundsIn } from '../src/data.ts';
import type { Run } from '../src/types.ts';

const run = {experiment_id:'a', name:'=unsafe, name', model_name:'character_cnn', partition_type:'iid',
  imbalance_method:'oversampling', data_kind:'real', training_protocol:'end_to_end',status:'completed',
  configuration:{batch_size:64,local_epochs:1,rounds:10,seed:42},comparison_group:'matched',partition_data_signature:'same-inputs',
  rounds:[1,2,3].map(n=>({round_number:n,metrics:{val:{adult_precision:.5,adult_recall:1,adult_f1:2/3,
    non_adult_precision:0,non_adult_recall:0,non_adult_f1:0,macro_f1:1/3,balanced_accuracy:.5,
    accuracy:.5,n_records:20,confusion_matrix:[[0,10],[0,10]]}},timing:{round_wall_seconds:3},communication:{}}))} as unknown as Run;

test('range filtering uses true rounds and preserves unavailable test data',()=>{
  assert.deepEqual(roundsIn(run,[2,3]).map(r=>r.round_number),[2,3]);
  assert.equal(latestRound(run,[1,2])?.round_number,2);
  assert.equal(latestRound(run,[4,5]),undefined);
  assert.equal(resultsCsv([run],[1,2],'test').split('\r\n').length,1);
});
test('CSV scope, nulls and spreadsheet formula safety',()=>{
  const csv=resultsCsv([run],[2,2],'val');
  assert.equal(csv.split('\r\n').length,2);
  assert.ok(csv.includes('"\'=unsafe, name"'));
  assert.ok(csv.includes('"[[0,10],[0,10]]"'));
  assert.ok(!csv.includes('"undefined"'));
});
test('mixed cohorts are never a compatible scientific comparison',()=>{
  assert.equal(compatible([run,{...run,experiment_id:'b'}]),true);
  assert.equal(compatible([run,{...run,comparison_group:'different'}]),false);
  assert.equal(compatible([run,{...run,comparison_group:null}]),false);
  assert.equal(compatible([run,{...run,partition_data_signature:'different-inputs'}]),false);
});
