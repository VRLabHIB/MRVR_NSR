import numpy as np

def create_dict_from_lists(keys, values):
    # to convert lists to dictionary
    res = {}
    for key in keys:
        for value in values:
            res[key] = value
            values.remove(value)
            break
    return res

# Function to split a string into a list of letters
def split(word):
    return [char for char in word]

def calculate_centroid(x2D, y2D):
    x = np.array(x2D)
    y = np.array(y2D)
    mx = np.nanmean(x)
    my = np.nanmean(y)

    dist = np.sqrt((x - mx)**2 + (y - my)**2)
    max_dist = np.max(dist)

    return mx, my, max_dist

def most_frequent(List):
    return max(set(List), key = List.count)

if __name__ == '__main__':
    #Unit Test
    res = create_dict_from_lists(['A','B','C'],[1,2,3])
    print(res)