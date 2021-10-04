I made this labeling script to help me quickly label data and save labels in a consistent JSON format.

To use, first add any images you want to `data/test/` or `data/validation/`.

Then, set the `dataset` variable in `labeling_script.py` to either `test` or `validation` depending on which set you're labeling.

Finally, run `labeling_script.py`

The script will grab all the unlabeled images from the specified directory and print out image file names one at a time. For each file, you should

1. Pull up the image in your favorite image viewing app
2. Enter the words you see, space separated, one row at a time. If a word includes a space (e.g. 'New York') replace the space with an underscore. If a card has already been guessed, enter `r`, `b`, `y`, or `k` for red cards, blue cards, civilian cards, and the assassin respectively.
3. After all 25 cards have been entered, type either `red` or `blue` depending on the color on the outside of the key card.
4. The last step is to enter the colors in the key, space separated, one row at a time. Enter `r`, `b`, `y`, or `k` for red cards, blue cards, civilian cards, and the assassin respectively.

When you're done you can check that a new JSON file has been created in `data/test/` or `data/validation` and that it has the proper values. 
