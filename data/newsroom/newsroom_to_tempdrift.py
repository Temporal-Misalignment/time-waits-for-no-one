import random
from collections import Counter
from urllib.parse import urlparse
import json
import os
import re

from tqdm import tqdm

urls_counts = {}
years_count = {}

classes = {
    'www.nytimes.com': 0,
    'www.washingtonpost.com': 1,
    'www.foxnews.com': 2
}

combined_years = {
    '2009': '2009_2010',
    '2010': '2009_2010',
    '2011': '2011_2012',
    '2012': '2011_2012',
    '2013': '2013_2014',
    '2014': '2013_2014',
    '2015': '2015_2016',
    '2016': '2015_2016',
}

adaptation_per_year = {}


def process_file(file, split):
    infile = open(file, 'r')
    count_total = 0
    count_filtered = 0
    summarization_rows_per_year = {}
    classification_rows_per_year_and_class = {}
    for line in tqdm(infile.readlines()):
        jsonline = json.loads(line)
        url = jsonline['url']
        o = urlparse(url)
        url_cleaned = o.netloc
        text = jsonline['text']
        summary = jsonline['summary']
        date = jsonline['date']
        year = date[:4]

        count_total += 1
        if year not in url:
            count_filtered += 1
            continue

        # find mentions of years in the summary
        years_mentioned = re.findall('[ .,\/]\d{4}[ .,\/]', summary) + re.findall('[ .,\/]\d{4}[ .,\/]', text)
        if len(years_mentioned) > 0:
            years_mentioned = [x.replace(".", "").replace("/", "").replace(",", "") for x in years_mentioned]
            valid_years = [x for x in years_mentioned if int(x) > int(year) and int(year) <= 2016]
            # print(valid_years)
            # if there are any "future" years in the article, it is very likely invalid date
            if len(valid_years) > 0:
                continue

        if int(year) < 2009:
            continue

        year = combined_years[year]

        if year not in years_count:
            years_count[year] = 0
        years_count[year] += 1

        if url_cleaned not in urls_counts:
            urls_counts[url_cleaned] = 0
        urls_counts[url_cleaned] += 1

        if year not in summarization_rows_per_year:
            summarization_rows_per_year[year] = []
            classification_rows_per_year_and_class[f"{year}_0"] = []
            classification_rows_per_year_and_class[f"{year}_1"] = []
            classification_rows_per_year_and_class[f"{year}_2"] = []
            # adaptation_per_year[year] = ""

        summarization_rows_per_year[year].append(
            {"year": year, "score": 0, "text": text, "summary": summary, "label": 0}
        )

        # adaptation_per_year[year] += text

        if url_cleaned in classes.keys():
            cls = classes[url_cleaned]
            classification_rows_per_year_and_class[f"{year}_{cls}"].append(
                {"year": year, "score": 0, "text": text, "label": classes[url_cleaned]}
            )

    print(urls_counts)
    print(years_count)

    # summarization
    if False:
        summarization_min_count = min([len(summarization_rows_per_year[str(year)]) for year in combined_years.values()])
        print(f" * min count: {summarization_min_count}")
        for year in combined_years.values():
            year = str(year)
            name = f'{year}.{split}.forlabels.scores'
            if split == 'train':
                name = 'indivs/' + name
            outfile = open(f"/Users/danielk/ideaProjects/temporal_drift/newsroom/summarization/{split}/{name}", '+w')
            for row in summarization_rows_per_year[year][:summarization_min_count]:
                outfile.write(json.dumps(row) + "\n")

    # unlabeled text
    if True:
        summarization_min_count = min([len(summarization_rows_per_year[str(year)]) for year in combined_years.values()])
        print(f" * min count: {summarization_min_count}")
        for year in combined_years.values():
            year = str(year)
            name = f'{year}.{split}.txt'
            if split == 'train':
                name = 'indivs/' + name
            outfile = open(f"/Users/danielk/ideaProjects/temporal_drift/newsroom/newsroom_dapt/{split}/{name}", '+w')
            for row in summarization_rows_per_year[year][:summarization_min_count]:
                outfile.write(row['text'] + "\n")

    # source classification
    if False:
        # balance for each classes
        print(classification_rows_per_year_and_class.keys())
        print([len(classification_rows_per_year_and_class[f"{year}_0"]) for year in combined_years.values()])
        print([len(classification_rows_per_year_and_class[f"{year}_1"]) for year in combined_years.values()])
        print([len(classification_rows_per_year_and_class[f"{year}_2"]) for year in combined_years.values()])

        classification_min_count_across_years_and_cats = min(
            [len(classification_rows_per_year_and_class[f"{year}_0"]) for year in combined_years.values()] +
            [len(classification_rows_per_year_and_class[f"{year}_1"]) for year in combined_years.values()] +
            [len(classification_rows_per_year_and_class[f"{year}_2"]) for year in combined_years.values()]
        )

        print(f" * min count: {classification_min_count_across_years_and_cats}")
        for year in combined_years.values():
            year = str(year)
            name = f'{year}.{split}.forlabels.scores'
            if split == 'train':
                name = 'indivs/' + name
            outfile = open(f"/Users/danielk/ideaProjects/temporal_drift/newsroom/newsroom_source_classification/{split}/{name}", '+w')

            rows = classification_rows_per_year_and_class[f"{year}_0"][:classification_min_count_across_years_and_cats]
            rows += classification_rows_per_year_and_class[f"{year}_1"][:classification_min_count_across_years_and_cats]
            rows += classification_rows_per_year_and_class[f"{year}_2"][:classification_min_count_across_years_and_cats]
            random.shuffle(rows)
            for row in rows:
                outfile.write(json.dumps(row) + "\n")


process_file('/Users/danielk/ideaProjects/temporal_drift/newsroom/release/dev.jsonl', 'dev')
process_file('/Users/danielk/ideaProjects/temporal_drift/newsroom/release/test.jsonl', 'test')
process_file('/Users/danielk/ideaProjects/temporal_drift/newsroom/release/train.jsonl', 'train')

