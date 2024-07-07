
import json,csv
def csv_json():
    json_fp = open("gossipcop_v3-4_story_based_fake.json", "r",encoding='utf-8')
    csv_fp = open("gossipcop_v3-4_story_based_fake.csv", "w",encoding='utf-8',newline='')
    writer = csv.writer(csv_fp)
    data_list = json.load(json_fp)
    for data in data_list:
        writer.writerow([data_list[data]['origin_label'],data_list[data]['origin_text'].replace("\n", "")])

    json_fp.close()
    csv_fp.close()

csv_json()

