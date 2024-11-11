from .volleyball import *
from .nba_modified import *


TRAIN_SEQS_VOLLEY = [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54]
VAL_SEQS_VOLLEY = [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51]
TEST_SEQS_VOLLEY = [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]


def read_dataset(args):
    # data_path = args.data_path + 'NBA_dataset'
    # image_path = data_path + "/videos"
    image_path = '/PATH/TO/Players'  #

    train_playerid_path = '/PATH/TO/D_train.json'
    test_playerid_path = '/PATH/TO/D_test.json'

    train_actionid_path = '/PATH/TO/E_action_train.json'
    test_actionid_path = '/PATH/TO/E_action_test.json'
    #test_id_path = '/PATH/TO/D_all.json'

    train_ids = read_ids(train_playerid_path)
    test_ids = read_ids(test_playerid_path)

    train_data = nba_read_annotations(train_playerid_path, train_actionid_path, train_ids)
    #print(train_data)
    #train_data = nba_read_annotations(image_2022_path, train_ids)
    train_frames = nba_all_frames(train_data)

    test_data = nba_read_annotations(test_playerid_path, test_actionid_path, test_ids)
    test_frames = nba_all_frames(test_data)

    train_set = NBADataset(train_frames, train_data, image_path, args)
    test_set = NBADataset(test_frames, test_data, image_path, args)


    print("%d train samples and %d test samples" % (len(train_frames), len(test_frames)))

    return train_set, test_set
