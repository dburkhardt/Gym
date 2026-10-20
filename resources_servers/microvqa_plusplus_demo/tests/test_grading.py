from resources_servers.microvqa_plusplus_demo.grading import grade
from resources_servers.microvqa_plusplus_demo.app import Config

def test_controls():
 assert grade('2',correct_index=2)['reward']==1
 assert grade('1',correct_index=2)['reward']==0
 for value in ['', '1 or 2','The answer is 2']:
  assert grade(value,correct_index=2)['reward']==0

def test_full_config():
 assert Config(host='127.0.0.1',port=40101,entrypoint='app.py',name='microvqa_plusplus_demo',domain='knowledge').domain=='knowledge'

def test_actual_native_verify():
 from fastapi.testclient import TestClient
 from unittest.mock import MagicMock
 from nemo_gym.server_utils import ServerClient
 from resources_servers.microvqa_plusplus_demo.app import Server
 from resources_servers.superchem.tests.test_native_contracts import response
 server=Server(config=Config(host='127.0.0.1',port=40101,entrypoint='app.py',name='microvqa_plusplus_demo',domain='knowledge'),server_client=MagicMock(spec=ServerClient))
 with TestClient(server.setup_webserver()) as client:
  for text,score in [('2',1),('1',0),('',0)]:
   result=client.post('/verify',json={'responses_create_params':{'input':[{'role':'user','content':'fixture'}]},'response':response(text),'verifier_metadata':{'correct_index':2}})
   assert result.status_code==200,result.text
   assert result.json()['reward']==score
