import json

pagerank_result_path = 'models/Ranking/pagerank_result.json'
hits_result_path = 'models/Ranking/hits_result.json'
hits_result_hubs_path = 'models/Ranking/hits_result_hubs.json'

print('Loading Ranking...')
pagerank_result = json.load(open(pagerank_result_path))
hits_result = json.load(open(hits_result_path))
hits_result_hubs = json.load(open(hits_result_hubs_path))
