import ujson as json


with open('IR_data_news_12k.json', 'r') as f:
	data = json.load(f)
	short_data = {}
	count = 0
	for key, value in data.items():
		short_data[key] = value
		count += 1
		if count > 500:
			break
	with open("data_500.json", "w") as nf:
		json.dump(short_data, nf)
