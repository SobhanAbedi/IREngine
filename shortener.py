# This program is meant for sampling of the original json file in order to have smaller files for testing

import ujson
samples = 500
with open('IR_data_news_12k.json', 'r') as f:
	data = ujson.load(f)
	short_data = {}
	count = 0
	for key, value in data.items():
		short_data[key] = value
		count += 1
		if count >= samples:
			break
	with open("data_" + str(samples) + ".json", "w") as nf:
		ujson.dump(short_data, nf)
