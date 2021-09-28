from glob import glob
import json


dataset = 'test'

all_image_files = sorted(glob(f'data/{dataset}/*.jpg'))
all_label_files = sorted(glob(f'data/{dataset}/*.json'))

for image_file in all_image_files:
    if image_file.replace('jpg', 'json') in all_label_files:
        continue
    print(image_file.split('/')[-1])
    card_data = []
    key_data = []

    for label in ['First', 'Second', 'Third', 'Fourth', 'Fifth']:
        next_data = input(f'{label} card row: ')
        while len(next_data.split()) != 5:
            print('Try again.')
            next_data = input(f'{label} card row: ')
        card_data.append(next_data.split())
    first_turn = ''
    while first_turn.lower() not in ['red', 'blue']:
        first_turn = input('Whose turn is next? ')
    for label in ['First', 'Second', 'Third', 'Fourth', 'Fifth']:
        next_data = input(f'{label} key row: ')
        while len(next_data.split()) != 5:
            print('Try again.')
            next_data = input(f'{label} key row: ')
        key_data.append(next_data.split())

    with open(image_file.replace('jpg', 'json'), 'w') as out_file:
        obj = {'cards': card_data, 'first_turn': first_turn, 'key': key_data}
        json.dump(obj, out_file)
