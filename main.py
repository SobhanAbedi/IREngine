import ujson as json
import hazm as h


def nmz(cnt: str) -> str:
    return normalizer.normalize(cnt)


def main(filename: str):
    f = open(filename, "r")
    data = json.load(f)

    for key, value in data.items():
        raw_cnt = value['content']
        # Normalize content
        nmz_cnt = nmz(raw_cnt)

        print(raw_cnt)
        print(nmz_cnt)
        break

    f.close()


if __name__ == '__main__':
    normalizer = h.Normalizer()
    main("data_500.json")
