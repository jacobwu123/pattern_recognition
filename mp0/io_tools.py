"""IO tools for mp0.
"""


def read_data_from_file(filename):
    """
    Read txt data from file.
    Each row is in the format article_id\ttitle\tpositivity_score\n.
    Store this information in a python dictionary. Key: article_id(int),
    value: [title(str), score(float)].

    Args:
        filename(string): Location of the file to load.
    Returns:
        out_dict(dict): data loaded from file.
    """
    out_dict = {}
    file = open(filename,'r')f
        key = int(line.strip().split('\t')[0])
        title = line.strip().split('\t')[1]
        score = line.strip().split('\t')[2]
        out_dict[key] = [title,float(score)] 
    file.close()
    return out_dict


def write_data_to_file(filename, data):
    """
    Writes data to file in the format article_id\ttitle\tpositivity_score\n.

    Args:
        filename(string): Location of the file to save.
        data(dict): data for writting to file.
    """
    keys = data.keys()
    file = open(filename,'w')
    
    for key in keys:
        file.write(str(key)+'\t' + data[key][0] + '\t' + str(data[key][1]) +'\n')
    file.close()

    return
