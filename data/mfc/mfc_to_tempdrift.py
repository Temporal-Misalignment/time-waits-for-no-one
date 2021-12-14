import json as js
import sys
from collections import defaultdict
import random 

TRN_SIZE = 1600
DEV_SIZE = 400
#TEST_SIZE = 400

fns = sys.argv[1:]
by_years = defaultdict(list)
#by_frames = defaultdict(int)
for fn in fns:
    with open(fn) as f:
        jso = js.load(f)
        for key in jso.keys():
            doc = jso[key]      
            yr = int(doc['year'])
            
            frame = doc['primary_frame']
            if frame is None:
                frame = 0
            else:
                frame = int(frame)  


            new_dict = {'year':yr, 'label': frame, 'text': doc['text'] }
            #print (doc.keys())

            by_years[yr].append(new_dict)


year_buckets = [ (2015, 2016), (2013, 2014), (2011, 2012), (2009, 2010), (2007, 2008), (2005, 2006) ]


def to_jsonfile( lst, outname ):
    with open( outname, 'w+' ) as f:
        for item in lst:
            f.write( js.dumps(item) + '\n' )

for (y1, y2) in year_buckets:
    data = by_years[y1] + by_years[y2]
    print( y1, y2, len(data) ) 
    random.shuffle( data )
    
    train = data[:TRN_SIZE]
    dev = data[TRN_SIZE:TRN_SIZE+DEV_SIZE]
    test = data[DEV_SIZE+TRN_SIZE:]

    
    outname = "{y1}-{y2}.{data}.forlabels.scores"
    to_jsonfile( train, outname.format(y1=y1, y2=y2, data='train.indivs' ))
    to_jsonfile( dev, outname.format(y1=y1, y2 = y2, data='dev' ))
    to_jsonfile( test, outname.format(y1=y1, y2=y2, data='test' ) )

