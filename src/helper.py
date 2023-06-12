
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


if __name__ == '__main__':
    #Unit Test
    res = create_dict_from_lists(['A','B','C'],[1,2,3])
    print(res)
