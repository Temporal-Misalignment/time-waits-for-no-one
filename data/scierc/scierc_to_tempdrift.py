import json
import os

all_types_to_idx = {
    'Task': 0,
    'Method': 1,
    'Material': 2,
    'Metric': 3,
    'OtherScientificTerm': 4,
    'Generic': 5
}

outfiles = {}

for split in ['test', 'dev', 'train']:
    sep1 = ''
    sep2 = ''
    if split == 'train':
        sep1 = '/indivs'
        sep2 = 'indivs.'
    outfiles[split + '1980'] = open(f'/Users/danielk/ideaProjects/temporal_drift/sciERC_temporal/{split}{sep1}/1980-1989.{split}.{sep2}forlabels.scores', 'w+')
    outfiles[split + '1990'] = open(f'/Users/danielk/ideaProjects/temporal_drift/sciERC_temporal/{split}{sep1}/1990-1999.{split}.{sep2}forlabels.scores', 'w+')
    outfiles[split + '2000'] = open(f'/Users/danielk/ideaProjects/temporal_drift/sciERC_temporal/{split}{sep1}/2000-2004.{split}.{sep2}forlabels.scores', 'w+')
    outfiles[split + '2005'] = open(f'/Users/danielk/ideaProjects/temporal_drift/sciERC_temporal/{split}{sep1}/2005-2009.{split}.{sep2}forlabels.scores', 'w+')
    outfiles[split + '2010'] = open(f'/Users/danielk/ideaProjects/temporal_drift/sciERC_temporal/{split}{sep1}/2010-2004.{split}.{sep2}forlabels.scores', 'w+')
    outfiles[split + '2015'] = open(f'/Users/danielk/ideaProjects/temporal_drift/sciERC_temporal/{split}{sep1}/2015-2016.{split}.{sep2}forlabels.scores', 'w+')

for subdir, dirs, files in os.walk("/Users/danielk/ideaProjects/temporal_drift/sciERC"):
    per_year_stats = {}
    all_data = {
        1980: [],
        1990: [],
        2000: [],
        2005: [],
        2010: [],
        2015: [],
    }

    for file in files:
        #print os.path.join(subdir, file)
        filepath = subdir + os.sep + file

        if file.endswith(".txt"):
            print(file)
            if "-" in file:
                year = file.split("-")[0]
                year = year[-2:]
                if year[0] == '9' or year[0] == '8':
                    year = '19' + year
                else:
                    year = '20' + year
            else:
                year = file.split("_")[1]
            year = int(year)
            print(year)
            if year not in per_year_stats:
                per_year_stats[year] = 0
                # all_data[year] = []

            with open(filepath, 'r') as f:
                text = f.read().replace("\n", " ").replace("\t", " ").strip()

            with open(filepath.replace('.txt', '.ann'), 'r') as f:
                for line in f.readlines():
                    line = line.replace('\n', '')
                    line_split = line.split('\t')
                    if len(line_split) == 3:
                        type = line_split[1].split(" ")[0]
                        type_idx = all_types_to_idx[type]
                        mention = line_split[2]
                        # print(mention, type)
                        row = {
                            'text': f"{mention} [SEP] {text}",
                            'labels': type_idx,
                            'year': year
                        }

                        per_year_stats[year] += 1
                        row = json.dumps(row)

                        if year < 1990:
                            all_data[1980].append(row)
                        elif year < 2000:
                            all_data[1990].append(row)
                        elif year < 2005:
                            all_data[2000].append(row)
                        elif year < 2010:
                            all_data[2005].append(row)
                        elif year < 2015:
                            all_data[2010].append(row)
                        elif year < 2020:
                            all_data[2015].append(row)

    print(per_year_stats)

    for year, rows in all_data.items():
        if len(rows) > 800:
            train_size = 600
        else:
            train_size = 400

        train = rows[:train_size]
        split_sizes = int((len(rows) - train_size)/2)
        test = rows[train_size:train_size + split_sizes]
        dev = rows[train_size + split_sizes:train_size + 2*split_sizes]
        outfiles['train' + str(year)].write('\n'.join(train))
        outfiles['test' + str(year)].write('\n'.join(test))
        outfiles['dev' + str(year)].write('\n'.join(dev))

