"""Read-only HTTP boundary check against this task's local preview."""
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

base='http://127.0.0.1:8765'
checks=[]
for route,expected in [('/api/health',200),('/api/experiments',200),('/',200),('/favicon.svg',200),
                       ('/data/raw/url.csv',404),('/api/files',404),('/%2e%2e/data/raw/url.csv',404),
                       ('/..%5cdata%5craw%5curl.csv',404)]:
    try:
        with urlopen(base+route,timeout=10) as response:
            code=response.status
            if route=='/api/experiments':
                snapshot=json.load(response)
                assert snapshot['capabilities']['read_only'] is True
                assert snapshot['capabilities']['raw_records'] is False
                assert len(snapshot['runs'])==7
                assert snapshot['dataset']['n_records']==1468243
                assert all(r['monitoring']=='recorded_snapshot' for r in snapshot['runs'])
    except HTTPError as error:
        code=error.code
    assert code==expected,(route,code,expected)
    checks.append({'route':route,'expected':expected,'actual':code})
try:
    urlopen(Request(base+'/api/experiments',data=b'{}',method='POST'),timeout=10)
    raise AssertionError('unexpected write route')
except HTTPError as error:
    assert error.code==501
    checks.append({'route':'POST /api/experiments','expected':501,'actual':error.code})
Path('data/dashboard-http-verification-v1.json').write_text(json.dumps({'status':'PASS','checks':checks},indent=2)+'\n')
print('PASS: read-only API, actual result counts, static assets, missing raw-data paths, traversal rejection, and no POST endpoint.')
