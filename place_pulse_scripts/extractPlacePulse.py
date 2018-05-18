# extracts a categorical subset from the place pulse 2.0 csv file

# You need to first download the placepulse votes csv file from: http://pulse.media.mit.edu/data/

place_pulse_file = 'C:/users/laggi/downloads/votes.csv'
category = 'wealthy'
output_file = 'C:/users/laggi/downloads/' + category + '.csv'

with open(output_file,'w') as fout:
    with open(place_pulse_file,'r') as f:
        fout.write(f.readline())
        for line in f:
            if category in line:
                if 'equal' not in line:
                    items=line.split(",")
                    fout.write(line)